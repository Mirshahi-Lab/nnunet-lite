"""Restoring logits to the input grid: the torch path against the numpy (nnU-Net) path."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from nnunet_lite.postprocessing import _argmax_numpy, _argmax_torch
from nnunet_lite.resampling import resample, resample_torch

# One coarse axis (ratio above nnU-Net's anisotropy threshold), so it is resampled
# separately by picking slices.
COARSE = (5.0, 1.0, 1.0)
FINE = (2.5, 1.5, 1.5)
KWARGS = {"order": 1, "order_z": 0, "force_separate_z": None}


def smooth(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(size=shape)
    smoothed: np.ndarray = gaussian_filter(noise, sigma=(0, 1.5, 1.5, 1.5)) * 20
    return smoothed


@pytest.mark.parametrize(
    ("old_shape", "new_shape", "current", "new"),
    [
        ((24, 20, 22), (12, 30, 33), FINE, COARSE),  # separate axis shrinks: slices picked first
        ((12, 30, 33), (24, 20, 22), COARSE, FINE),  # separate axis grows: slices repeated last
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_torch_resampling_matches_numpy_with_a_separate_axis(
    old_shape: tuple[int, ...],
    new_shape: tuple[int, ...],
    current: tuple[float, ...],
    new: tuple[float, ...],
    dtype: torch.dtype,
) -> None:
    data = smooth((2, *old_shape))
    expected = resample(data, new_shape, current, new, **KWARGS)
    actual = resample_torch(torch.from_numpy(data), new_shape, current, new, dtype=dtype, **KWARGS)
    assert actual.dtype == torch.float64  # dtype is the working precision only
    tolerance = 1e-9 if dtype == torch.float64 else 1e-4
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=tolerance)


def test_torch_argmax_matches_numpy_argmax() -> None:
    # float16 logits, as the sliding window returns them; classes are resampled one at
    # a time in float32 and compared after rounding back to float16.
    logits = torch.from_numpy(smooth((5, 24, 20, 22))).to(torch.float16)
    shape = (12, 30, 33)
    arguments = (logits, shape, FINE, COARSE, KWARGS, torch.uint8)
    expected = _argmax_numpy(*arguments)
    actual = _argmax_torch(*arguments, torch.device("cpu"))
    assert actual.dtype == expected.dtype == torch.uint8
    assert len(expected.unique()) == 5
    assert (actual == expected).float().mean() > 0.999


def test_torch_argmax_without_resampling_is_a_plain_argmax() -> None:
    logits = torch.from_numpy(smooth((4, 8, 9, 10))).to(torch.float16)
    labels = _argmax_torch(logits, (8, 9, 10), FINE, FINE, KWARGS, torch.uint8, torch.device("cpu"))
    assert torch.equal(labels, logits.argmax(0).to(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_logits_are_restored_on_the_gpu_one_class_at_a_time() -> None:
    logits = torch.from_numpy(smooth((5, 24, 20, 22))).to(torch.float16)
    shape = (12, 30, 33)
    arguments = (logits, shape, FINE, COARSE, KWARGS, torch.uint8)
    streamed = _argmax_torch(*arguments, torch.device("cuda"))
    on_device = _argmax_torch(logits.cuda(), *arguments[1:], torch.device("cuda"))
    assert streamed.device.type == "cuda"
    assert torch.equal(streamed, on_device)
    assert (streamed.cpu() == _argmax_numpy(*arguments)).float().mean() > 0.999
