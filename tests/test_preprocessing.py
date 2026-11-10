"""Preprocessing behaviour that can be checked without nnunetv2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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


def volume(shape: tuple[int, ...] = (1, 20, 24, 28), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 300, shape).astype(np.float32)


def test_ct_normalization_without_resampling() -> None:
    data = volume()
    output, properties = nl.preprocess(data, (1.5, 1.5, 1.5), CONFIG)
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
    output, properties = nl.preprocess(data, (3.0, 0.75, 1.5), CONFIG)
    assert properties.crop_bbox == ((4, 16), (2, 20), (6, 26))
    assert properties.shape_after_cropping == (12, 18, 20)
    assert tuple(output.shape) == (1, 24, 9, 20)
    assert torch.isfinite(output).all()


def test_all_zero_image_is_not_cropped() -> None:
    output, properties = nl.preprocess(np.zeros((1, 6, 8, 10)), (1.5, 1.5, 1.5), CONFIG)
    assert properties.crop_bbox == ((0, 6), (0, 8), (0, 10))
    assert tuple(output.shape) == (1, 6, 8, 10)


def test_transpose_forward_is_applied_to_data_and_spacing() -> None:
    config = replace(CONFIG, transpose_forward=(2, 0, 1), transpose_backward=(1, 2, 0))
    output, properties = nl.preprocess(volume(), (1.5, 1.5, 3.0), config)
    assert properties.spacing == (3.0, 1.5, 1.5)
    assert properties.shape_before_cropping == (28, 20, 24)
    assert tuple(output.shape) == (1, 56, 20, 24)


def test_2d_configuration_keeps_slice_spacing() -> None:
    config = replace(CONFIG, target_spacing=(1.5, 1.5))
    output, _ = nl.preprocess(volume(), (4.0, 0.75, 0.75), config)
    assert tuple(output.shape) == (1, 20, 12, 14)


def test_tensor_and_array_inputs_match_and_are_not_modified() -> None:
    data = volume().astype(np.int16)
    tensor = torch.from_numpy(data.copy())
    from_array, _ = nl.preprocess(data, (2.0, 1.0, 1.0), CONFIG)
    from_tensor, _ = nl.preprocess(tensor, (2.0, 1.0, 1.0), CONFIG)
    assert torch.equal(from_array, from_tensor)
    assert np.array_equal(tensor.numpy(), data)
    assert tensor.dtype == torch.int16


def test_masked_zscore_leaves_outside_at_zero() -> None:
    channel = replace(CT, normalization="ZScoreNormalization", use_mask_for_norm=True)
    data = volume() + 1000
    data[:, :, :, :5] = 0
    data[:, :3] = 0
    output, _ = nl.preprocess(data, (1.5, 1.5, 1.5), replace(CONFIG, channels=(channel,)))
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
        nl.preprocess(data, spacing, CONFIG)


def test_rejects_unsupported_schemes() -> None:
    torch_resampling = ResamplingConfig("resample_torch_fornnunet", {"is_seg": False})
    with pytest.raises(NotImplementedError, match="resampling function"):
        nl.preprocess(volume(), (1.0, 1.0, 1.0), replace(CONFIG, resampling_data=torch_resampling))
    channel = replace(CT, normalization="MyNormalization")
    with pytest.raises(NotImplementedError, match="normalization scheme"):
        nl.preprocess(volume(), (1.0, 1.0, 1.0), replace(CONFIG, channels=(channel,)))


def test_load_config_reads_bundle_without_weights(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(tmp_path, "3d_fullres")
    bundle = nl.convert(model_dir, output_dir=tmp_path / "bundle")
    (tmp_path / "bundle" / "weights.safetensors").unlink()
    config = nl.load_config(tmp_path / "bundle")
    assert config == bundle.config
    output, _ = nl.preprocess(volume(), (1.5, 1.5, 1.5), config.preprocessing)
    assert tuple(output.shape) == (1, 20, 24, 28)
