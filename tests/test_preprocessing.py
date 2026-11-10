"""Preprocessing behaviour that can be checked without nnunetv2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_convert import make_model_dir

import nnunet_lite as nl
from nnunet_lite.config import ChannelConfig, PreprocessingConfig, ResamplingConfig

CT = ChannelConfig(
    name="CT",
    normalization="CTNormalization",
    use_mask_for_norm=False,
    lower=-100.0,
    upper=200.0,
    mean=10.0,
    std=50.0,
)
CONFIG = PreprocessingConfig(
    image_reader="NibabelIOWithReorient",
    target_spacing=(1.5, 1.5, 1.5),
    transpose_forward=(0, 1, 2),
    transpose_backward=(0, 1, 2),
    channels=(CT,),
    resampling_data=ResamplingConfig(
        "resample_data_or_seg_to_shape",
        {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
    ),
    resampling_probabilities=ResamplingConfig(
        "resample_data_or_seg_to_shape",
        {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
    ),
)


def preprocess_cpu(*args: Any) -> tuple[torch.Tensor, nl.PreprocessingProperties]:
    return nl.preprocess(*args, device="cpu")


def volume(shape: tuple[int, ...] = (1, 20, 24, 28), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 300, shape).astype(np.float32)


def test_ct_normalization_without_resampling() -> None:
    data = volume()
    output, properties = preprocess_cpu(data, (1.5, 1.5, 1.5), CONFIG)
    expected = (np.clip(data, -100.0, 200.0) - np.float32(10.0)) / np.float32(50.0)
    assert output.dtype == torch.float32
    assert np.array_equal(output.numpy(), expected)
    assert properties.shape_before_cropping == (20, 24, 28)
    assert properties.crop_bbox == ((0, 20), (0, 24), (0, 28))
    assert properties.shape_after_cropping == (20, 24, 28)
    assert properties.spacing == (1.5, 1.5, 1.5)


def test_crops_to_nonzero_region_and_resamples_to_target_spacing() -> None:
    data = np.zeros((1, 20, 24, 28), dtype=np.float32)
    data[:, 4:16, 2:20, 6:26] = volume((1, 12, 18, 20)) + 5000
    output, properties = preprocess_cpu(data, (3.0, 0.75, 1.5), CONFIG)
    assert properties.crop_bbox == ((4, 16), (2, 20), (6, 26))
    assert properties.shape_after_cropping == (12, 18, 20)
    assert tuple(output.shape) == (1, 24, 9, 20)
    assert torch.isfinite(output).all()


def test_all_zero_image_is_not_cropped() -> None:
    output, properties = preprocess_cpu(np.zeros((1, 6, 8, 10)), (1.5, 1.5, 1.5), CONFIG)
    assert properties.crop_bbox == ((0, 6), (0, 8), (0, 10))
    assert tuple(output.shape) == (1, 6, 8, 10)


def test_transpose_forward_is_applied_to_data_and_spacing() -> None:
    config = replace(CONFIG, transpose_forward=(2, 0, 1), transpose_backward=(1, 2, 0))
    output, properties = preprocess_cpu(volume(), (1.5, 1.5, 3.0), config)
    assert properties.spacing == (3.0, 1.5, 1.5)
    assert properties.shape_before_cropping == (28, 20, 24)
    assert tuple(output.shape) == (1, 56, 20, 24)


def test_2d_configuration_keeps_slice_spacing() -> None:
    config = replace(CONFIG, target_spacing=(1.5, 1.5))
    output, _ = preprocess_cpu(volume(), (4.0, 0.75, 0.75), config)
    assert tuple(output.shape) == (1, 20, 12, 14)


def test_tensor_and_array_inputs_match_and_are_not_modified() -> None:
    data = volume().astype(np.int16)
    tensor = torch.from_numpy(data.copy())
    from_array, _ = preprocess_cpu(data, (2.0, 1.0, 1.0), CONFIG)
    from_tensor, _ = preprocess_cpu(tensor, (2.0, 1.0, 1.0), CONFIG)
    assert torch.equal(from_array, from_tensor)
    assert np.array_equal(tensor.numpy(), data)
    assert tensor.dtype == torch.int16


def test_masked_zscore_leaves_outside_at_zero() -> None:
    channel = replace(CT, normalization="ZScoreNormalization", use_mask_for_norm=True)
    data = volume() + 1000
    data[:, :, :, :5] = 0
    data[:, :3] = 0
    output, _ = preprocess_cpu(data, (1.5, 1.5, 1.5), replace(CONFIG, channels=(channel,)))
    inside = output[output != 0]
    assert abs(float(inside.mean())) < 1e-4
    assert abs(float(inside.std()) - 1) < 1e-3


@pytest.mark.parametrize(
    ("data", "spacing", "match"),
    [
        (np.zeros((8, 8, 8)), (1.0, 1.0, 1.0), "shape"),
        (np.zeros((2, 8, 8, 8)), (1.0, 1.0, 1.0), "channel"),
        (np.zeros((1, 8, 8, 8)), (1.0, 1.0), "spacing"),
    ],
)
def test_rejects_bad_input(data: np.ndarray, spacing: tuple[float, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        preprocess_cpu(data, spacing, CONFIG)


def test_rejects_unsupported_schemes() -> None:
    torch_resampling = ResamplingConfig("resample_torch_fornnunet", {"is_seg": False})
    with pytest.raises(NotImplementedError, match="resampling function"):
        preprocess_cpu(volume(), (1.0, 1.0, 1.0), replace(CONFIG, resampling_data=torch_resampling))
    channel = replace(CT, normalization="MyNormalization")
    with pytest.raises(NotImplementedError, match="normalization scheme"):
        preprocess_cpu(volume(), (1.0, 1.0, 1.0), replace(CONFIG, channels=(channel,)))


def test_load_config_reads_bundle_without_weights(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(tmp_path, "3d_fullres")
    bundle = nl.convert(model_dir, output_dir=tmp_path / "bundle")
    (tmp_path / "bundle" / "weights.safetensors").unlink()
    config = nl.load_config(tmp_path / "bundle")
    assert config == bundle.config
    output, _ = preprocess_cpu(volume(), (1.5, 1.5, 1.5), config.preprocessing)
    assert tuple(output.shape) == (1, 20, 24, 28)


# --- torch path (GPU) against the exact numpy path ----------------------------

ZSCORE = replace(CT, normalization="ZScoreNormalization")
TORCH_CONFIGS = {
    "ct": CONFIG,
    "ct_order1_z1": replace(
        CONFIG,
        resampling_data=ResamplingConfig(
            "resample_data_or_seg_to_shape", {"order": 1, "order_z": 1, "force_separate_z": True}
        ),
    ),
    "ct_anisotropic_target": replace(CONFIG, target_spacing=(5.0, 0.9, 0.9)),
    "ct_transposed": replace(CONFIG, transpose_forward=(2, 0, 1), transpose_backward=(1, 2, 0)),
    "ct_2d": replace(CONFIG, target_spacing=(1.25, 1.25)),
    "zscore_masked_and_plain": replace(
        CONFIG, channels=(replace(ZSCORE, use_mask_for_norm=True), ZSCORE)
    ),
    "rescale_and_none": replace(
        CONFIG,
        channels=(
            replace(CT, normalization="RescaleTo01Normalization"),
            replace(CT, normalization="NoNormalization"),
        ),
    ),
}
TORCH_GEOMETRIES = {
    "isotropic_fine": ((30, 44, 40), (1.0, 1.0, 1.0)),
    "matches_target": ((30, 44, 40), (1.5, 1.5, 1.5)),
    "thick_slices": ((12, 50, 46), (5.0, 0.8, 0.8)),
    "thick_middle_axis": ((40, 10, 44), (0.9, 4.5, 0.9)),
    "thick_last_axis": ((44, 40, 11), (0.7, 0.7, 4.0)),
    "odd_spacing": ((33, 37, 41), (1.37, 0.91, 2.6)),
}
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def bordered_volume(channels: int, shape: tuple[int, ...]) -> np.ndarray:
    data = volume((channels, *shape), seed=channels + sum(shape)) + 400
    data[:, :3] = 0
    data[:, :, -5:] = 0
    centre = tuple(s // 2 for s in shape)
    data[:, centre[0], centre[1] - 2 : centre[1] + 2, centre[2] - 2 : centre[2] + 2] = 0
    return data


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("geometry", sorted(TORCH_GEOMETRIES))
@pytest.mark.parametrize("name", sorted(TORCH_CONFIGS))
def test_torch_path_matches_numpy_path(name: str, geometry: str, device: str) -> None:
    from nnunet_lite.preprocessing import _preprocess_torch, resampling_kwargs

    config = TORCH_CONFIGS[name]
    shape, spacing = TORCH_GEOMETRIES[geometry]
    data = bordered_volume(len(config.channels), shape)
    before = data.copy()
    expected, expected_properties = preprocess_cpu(data, spacing, config)
    # Called directly so the torch code is exercised on CPU too, where preprocess() uses numpy.
    output, properties = _preprocess_torch(
        data,
        tuple(float(spacing[a]) for a in config.transpose_forward),
        config,
        resampling_kwargs(config.resampling_data),
        torch.device(device),
    )
    assert np.array_equal(data, before)
    assert properties == expected_properties
    assert output.dtype == torch.float32
    assert output.device.type == device
    assert output.shape == expected.shape
    # Same operations in a different summation order: a few float32 rounding steps at most.
    scale = max(1.0, float(expected.abs().max()))
    assert float((output.cpu() - expected).abs().max()) <= 2e-6 * scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_uses_gpu_by_default_and_keeps_input_tensor_unchanged() -> None:
    tensor = torch.from_numpy(volume()).cuda()
    before = tensor.clone()
    output, _ = nl.preprocess(tensor, (1.0, 1.0, 1.0), CONFIG)
    assert output.device.type == "cuda"
    assert torch.equal(tensor, before)
    expected, _ = preprocess_cpu(tensor, (1.0, 1.0, 1.0), CONFIG)
    assert torch.allclose(output.cpu(), expected, atol=1e-5)


def test_defaults_to_cpu_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output, _ = nl.preprocess(volume(), (1.0, 1.0, 1.0), CONFIG)
    expected, _ = preprocess_cpu(volume(), (1.0, 1.0, 1.0), CONFIG)
    assert output.device.type == "cpu"
    assert torch.equal(output, expected)


def test_falls_back_to_cpu_when_gpu_path_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def out_of_memory(*args: Any) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 9.00 GiB")

    monkeypatch.setattr("nnunet_lite.preprocessing._preprocess_torch", out_of_memory)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        output, properties = nl.preprocess(volume(), (1.0, 1.0, 1.0), CONFIG, device="cuda")
    expected, expected_properties = preprocess_cpu(volume(), (1.0, 1.0, 1.0), CONFIG)
    assert output.device.type == "cpu"
    assert torch.equal(output, expected)
    assert properties == expected_properties


def test_non_finite_input_falls_back_to_cpu() -> None:
    from nnunet_lite.preprocessing import _preprocess_torch, resampling_kwargs

    data = volume()
    data[0, 5, 5, 5] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _preprocess_torch(
            data, (1.0, 1.0, 1.0), CONFIG, resampling_kwargs(CONFIG.resampling_data), torch.device("cpu")
        )
