from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates, zoom

# Port of nnU-Net's ``resample_data_or_seg_to_shape`` for continuous data
# (images and logits). Results are byte-equal to nnU-Net, which resamples in
# float64 with skimage; ``_resize`` reproduces the skimage call it makes.

ANISOTROPY_THRESHOLD = 3.0
SUPPORTED_FUNCTION = "resample_data_or_seg_to_shape"


def compute_new_shape(
    old_shape: Sequence[int], old_spacing: Sequence[float], new_spacing: Sequence[float]
) -> tuple[int, ...]:
    return tuple(
        round(float(old) / float(new) * int(size))
        for old, new, size in zip(old_spacing, new_spacing, old_shape, strict=True)
    )


def _lowres_axes(spacing: NDArray[np.float64]) -> NDArray[np.intp]:
    return np.where(spacing.max() / spacing == 1)[0]


def _separate_axis(
    force_separate_z: bool | None,
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    threshold: float,
) -> int | None:
    current = np.asarray(current_spacing, dtype=float)
    new = np.asarray(new_spacing, dtype=float)
    if force_separate_z is not None:
        if not force_separate_z:
            return None
        axes = _lowres_axes(current)
    elif current.max() / current.min() > threshold:
        axes = _lowres_axes(current)
    elif new.max() / new.min() > threshold:
        axes = _lowres_axes(new)
    else:
        return None
    # Two or three equally coarse axes: no single out-of-plane axis to treat separately.
    return int(axes[0]) if len(axes) == 1 else None


def _resize(image: NDArray[np.float64], output_shape: Sequence[int], order: int) -> Any:
    # skimage.transform.resize(image, output_shape, order, mode="edge", anti_aliasing=False)
    factors = np.divide(image.shape, tuple(output_shape))
    out = zoom(image, [1 / f for f in factors], order=order, mode="nearest", grid_mode=True)
    low = np.min(image)
    if np.isnan(low):
        low, high = np.nanmin(image), np.nanmax(image)
    else:
        high = np.max(image)
    np.clip(out, low, high, out=out)
    return out


def resample(
    data: NDArray[Any],
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    *,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: bool | None = False,
    separate_z_anisotropy_threshold: float = ANISOTROPY_THRESHOLD,
) -> NDArray[Any]:
    if data.ndim != 4:
        raise ValueError(f"Expected (C, X, Y, Z) data, got shape {data.shape}")
    old_shape = data.shape[1:]
    target = tuple(int(v) for v in new_shape)
    if old_shape == target:
        return data

    axis = _separate_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold
    )
    output = np.zeros((data.shape[0], *target), dtype=data.dtype)
    continuous = data.astype(float, copy=False)
    if axis is None:
        for channel in range(data.shape[0]):
            output[channel] = _resize(continuous[channel], target, order)
        return output

    in_plane = tuple(v for index, v in enumerate(target) if index != axis)
    intermediate_shape = tuple(
        old_shape[index] if index == axis else v for index, v in enumerate(target)
    )
    coordinates = None
    if old_shape[axis] != target[axis]:
        grid = np.mgrid[: target[0], : target[1], : target[2]]
        coordinates = np.array(
            [float(intermediate_shape[d]) / target[d] * (grid[d] + 0.5) - 0.5 for d in range(3)]
        )
    slicer: list[Any] = [slice(None)] * 3
    for channel in range(data.shape[0]):
        intermediate = np.zeros(intermediate_shape)
        for index in range(old_shape[axis]):
            slicer[axis] = index
            intermediate[tuple(slicer)] = _resize(continuous[channel][tuple(slicer)], in_plane, order)
        if coordinates is None:
            output[channel] = intermediate
        else:
            output[channel] = map_coordinates(intermediate, coordinates, order=order_z, mode="nearest")
    return output
