from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes

from .config import ChannelConfig, PreprocessingConfig, ResamplingConfig
from .resampling import SUPPORTED_FUNCTION, compute_new_shape, resample

# Port of nnU-Net's DefaultPreprocessor.run_case_npy for inference (no
# segmentation): transpose -> crop to nonzero -> normalize -> resample.
# Output is byte-equal to nnU-Net for the same input array and spacing.

_EPS = 1e-8


@dataclass(frozen=True)
class PreprocessingProperties:
    # Everything needed to map a prediction back onto the input array.
    # All values are in transposed (network) axis order.
    spacing: tuple[float, ...]
    shape_before_cropping: tuple[int, ...]
    crop_bbox: tuple[tuple[int, int], ...]
    shape_after_cropping: tuple[int, ...]


def _crop_to_nonzero(
    data: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.bool_], tuple[tuple[int, int], ...]]:
    mask: NDArray[np.bool_] = binary_fill_holes(np.any(data != 0, axis=0))
    bbox = []
    for axis in range(mask.ndim):
        occupied = np.flatnonzero(np.any(mask, axis=tuple(a for a in range(mask.ndim) if a != axis)))
        # nnU-Net keeps the full extent of an all-zero image.
        bbox.append(
            (int(occupied[0]), int(occupied[-1]) + 1) if occupied.size else (0, mask.shape[axis])
        )
    slicer = tuple(slice(start, end) for start, end in bbox)
    return data[(slice(None), *slicer)], mask[slicer], tuple(bbox)


def _normalize(
    image: NDArray[np.float32], channel: ChannelConfig, mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    scheme = channel.normalization
    if scheme == "CTNormalization":
        np.clip(image, channel.lower, channel.upper, out=image)
        image -= channel.mean
        image /= max(channel.std, _EPS)
    elif scheme == "ZScoreNormalization":
        if channel.use_mask_for_norm:
            inside = image[mask]
            image[mask] = (inside - inside.mean()) / max(inside.std(), _EPS)
        else:
            mean, std = image.mean(), image.std()
            image -= mean
            image /= max(std, _EPS)
    elif scheme == "RescaleTo01Normalization":
        image -= image.min()
        image /= np.clip(image.max(), a_min=_EPS, a_max=None)
    elif scheme == "RGBTo01Normalization":
        if image.min() < 0 or image.max() > 255:
            raise ValueError("RGBTo01Normalization expects values in [0, 255]")
        image /= 255.0
    elif scheme != "NoNormalization":
        raise NotImplementedError(f"Unsupported normalization scheme: {scheme}")
    return image


def resampling_kwargs(resampling: ResamplingConfig) -> dict[str, Any]:
    if resampling.function != SUPPORTED_FUNCTION:
        raise NotImplementedError(f"Unsupported resampling function: {resampling.function}")
    kwargs = dict(resampling.kwargs)
    if kwargs.pop("is_seg", False):
        raise NotImplementedError("Segmentation resampling is not supported")
    unknown = set(kwargs) - {
        "order",
        "order_z",
        "force_separate_z",
        "separate_z_anisotropy_threshold",
    }
    if unknown:
        raise NotImplementedError(f"Unsupported resampling kwargs: {sorted(unknown)}")
    return kwargs


def preprocess(
    data: NDArray[Any] | torch.Tensor,
    spacing: Sequence[float],
    config: PreprocessingConfig,
) -> tuple[torch.Tensor, PreprocessingProperties]:
    array = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    if array.ndim != 4:
        raise ValueError(
            f"Expected data of shape (C, X, Y, Z), got {array.shape}. Add a leading channel "
            "axis to a single-channel volume, and a singleton first spatial axis to a 2D image."
        )
    if array.shape[0] != len(config.channels):
        raise ValueError(
            f"Model expects {len(config.channels)} channel(s) "
            f"({', '.join(c.name for c in config.channels)}), got {array.shape[0]}"
        )
    if len(spacing) != 3:
        raise ValueError(f"Expected one spacing value per spatial axis (3), got {len(spacing)}")
    kwargs = resampling_kwargs(config.resampling_data)

    # astype copies, so the caller's array is never modified.
    volume: NDArray[np.float32] = array.astype(np.float32).transpose(
        [0, *[axis + 1 for axis in config.transpose_forward]]
    )
    original_spacing = tuple(float(spacing[axis]) for axis in config.transpose_forward)

    shape_before_cropping = volume.shape[1:]
    volume, mask, bbox = _crop_to_nonzero(volume)

    target_spacing = config.target_spacing
    if len(target_spacing) < volume.ndim - 1:
        # 2D configurations keep the spacing between slices.
        target_spacing = (original_spacing[0], *target_spacing)
    new_shape = compute_new_shape(volume.shape[1:], original_spacing, target_spacing)

    # Normalization must come before resampling, as in nnU-Net.
    for index, channel in enumerate(config.channels):
        volume[index] = _normalize(volume[index], channel, mask)
    shape_after_cropping = volume.shape[1:]
    volume = resample(volume, new_shape, original_spacing, target_spacing, **kwargs)

    properties = PreprocessingProperties(
        spacing=original_spacing,
        shape_before_cropping=tuple(int(v) for v in shape_before_cropping),
        crop_bbox=bbox,
        shape_after_cropping=tuple(int(v) for v in shape_after_cropping),
    )
    return torch.from_numpy(np.ascontiguousarray(volume)), properties
