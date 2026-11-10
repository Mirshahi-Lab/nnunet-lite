"""Sliding-window prediction on synthetic bundles (no nnunetv2 needed)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_convert import make_model_dir

import nnunet_lite as nl
from nnunet_lite import sliding_window
from nnunet_lite.sliding_window import (
    _tiles,
    compute_steps,
    gaussian_importance_map,
    pad_to_patch,
    predict_logits,
)

NO_GAUSSIAN = nl.InferenceConfig(use_gaussian=False)


@pytest.fixture
def bundle(tmp_path: Path) -> nl.Bundle:
    return nl.convert(make_model_dir(tmp_path, "3d_fullres")[0])


def test_compute_steps() -> None:
    assert compute_steps((16, 32, 40), (16, 16, 16), 0.5) == [[0], [0, 8, 16], [0, 8, 16, 24]]
    assert compute_steps((20,), (16,), 0.5) == [[0, 4]]
    assert compute_steps((40,), (16,), 1.0) == [[0, 12, 24]]
    with pytest.raises(ValueError, match="padded"):
        compute_steps((8,), (16,), 0.5)


def test_tile_step_size_comes_from_the_config() -> None:
    assert len(_tiles((32, 32, 32), (16, 16, 16), 0.5)) == 27
    assert len(_tiles((32, 32, 32), (16, 16, 16), 1.0)) == 8


def test_pad_to_patch() -> None:
    image = torch.arange(1 * 10 * 16 * 21, dtype=torch.float32).reshape(1, 10, 16, 21)
    padded, revert = pad_to_patch(image, (16, 16, 16))
    assert padded.shape == (1, 16, 16, 21)
    assert torch.equal(padded[revert], image)
    assert torch.equal(padded[0, 3:13], image[0])  # 6 voxels of padding: 3 before, 3 after
    odd, revert = pad_to_patch(torch.ones(1, 11, 16, 16), (16, 16, 16))
    assert odd[0, :, 0, 0].tolist() == [0, 0] + [1] * 11 + [0, 0, 0]  # the extra voxel goes last


def test_gaussian_importance_map() -> None:
    gaussian = gaussian_importance_map((16, 16, 16), 0.125, 10.0)
    assert gaussian.shape == (16, 16, 16)
    assert gaussian.dtype == torch.float16
    assert gaussian.max() == 10
    assert gaussian.min() > 0
    assert gaussian[8, 8, 8] == gaussian.max()


def test_single_patch_equals_a_direct_forward(bundle: nl.Bundle) -> None:
    config = replace(bundle.config, inference=NO_GAUSSIAN)
    data = torch.randn(1, 16, 16, 16)
    with torch.inference_mode():
        expected = bundle.network(data[None])[0].half()
    assert torch.equal(predict_logits(bundle.network, data, config), expected)


def test_gaussian_weighting_changes_overlapping_tiles_only(bundle: nl.Bundle) -> None:
    data = torch.randn(1, 16, 16, 24)
    weighted = predict_logits(bundle.network, data, bundle.config)
    uniform = predict_logits(bundle.network, data, replace(bundle.config, inference=NO_GAUSSIAN))
    assert weighted.shape == uniform.shape == (3, 16, 16, 24)
    assert torch.isfinite(weighted).all()
    assert not torch.equal(weighted, uniform)


def test_batch_size_does_not_change_the_result(bundle: nl.Bundle) -> None:
    data = torch.randn(1, 24, 20, 30)
    single = predict_logits(bundle.network, data, bundle.config, batch_size=1)
    batched = predict_logits(bundle.network, data, bundle.config, batch_size=4)
    assert torch.allclose(single.float(), batched.float(), atol=2e-3)


def test_image_smaller_than_the_patch(bundle: nl.Bundle) -> None:
    data = torch.randn(1, 8, 10, 12)
    logits = predict_logits(bundle.network, data, bundle.config)
    assert logits.shape == (3, 8, 10, 12)
    assert torch.isfinite(logits).all()


def test_mirroring(bundle: nl.Bundle) -> None:
    # Without Gaussian weights, whose float16 tails add noise at the tile corners.
    config = replace(bundle.config, inference=NO_GAUSSIAN)
    data = torch.randn(1, 16, 16, 16)
    plain = predict_logits(bundle.network, data, config)
    mirrored = predict_logits(bundle.network, data, config, mirror=True)
    assert not torch.equal(plain, mirrored)
    # Averaging over every flip makes the prediction equivariant to flipping the input.
    flipped = predict_logits(bundle.network, data.flip(1, 3), config, mirror=True)
    assert torch.allclose(flipped.flip(1, 3).float(), mirrored.float(), atol=2e-3)
    # A model trained without mirroring ignores the request.
    disabled = replace(config, mirror_axes=())
    assert torch.equal(predict_logits(bundle.network, data, disabled, mirror=True), plain)


def test_2d_network_predicts_slice_by_slice(tmp_path: Path) -> None:
    bundle = nl.convert(make_model_dir(tmp_path, "2d")[0])
    data = torch.randn(1, 3, 20, 24)
    logits = predict_logits(bundle.network, data, bundle.config)
    assert logits.shape == (3, 3, 20, 24)
    for index in range(3):
        alone = predict_logits(bundle.network, data[:, index : index + 1], bundle.config)
        assert torch.allclose(alone[:, 0].float(), logits[:, index].float(), atol=2e-3)


def test_rejects_bad_input(bundle: nl.Bundle) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        predict_logits(bundle.network, torch.zeros(1, 16, 16, 16), bundle.config, batch_size=0)
    with pytest.raises(ValueError, match="Expected preprocessed data"):
        predict_logits(bundle.network, torch.zeros(2, 16, 16, 16), bundle.config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu_and_falls_back_when_out_of_memory(
    bundle: nl.Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(bundle.config, inference=NO_GAUSSIAN)
    data = torch.randn(1, 24, 20, 30)
    cpu = predict_logits(bundle.network, data, config)
    bundle.network.to("cuda")
    gpu = predict_logits(bundle.network, data.cuda(), config)
    assert gpu.device.type == "cuda"
    assert torch.allclose(gpu.cpu().float(), cpu.float(), atol=2e-2)

    accumulate = sliding_window._accumulate
    devices = []

    def out_of_memory_on_gpu(*args: object) -> torch.Tensor:
        devices.append(args[5])
        if args[5].type == "cuda":  # type: ignore[attr-defined]
            raise torch.OutOfMemoryError("CUDA out of memory")
        return accumulate(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(sliding_window, "_accumulate", out_of_memory_on_gpu)
    with pytest.warns(RuntimeWarning, match="accumulating on CPU"):
        fallback = predict_logits(bundle.network, data.cuda(), config)
    assert [d.type for d in devices] == ["cuda", "cpu"]
    assert fallback.device.type == "cpu"
    assert torch.allclose(fallback.float(), cpu.float(), atol=2e-2)
