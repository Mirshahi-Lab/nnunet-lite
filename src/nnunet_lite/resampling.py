from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch
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


# --- torch implementation (any device) ---------------------------------------
# Spline resizing is linear and separable, so each axis is one matrix product.
# The matrices come from the same scipy call the numpy path makes, which keeps
# the result within float64 rounding of it instead of re-deriving the splines.


@lru_cache(maxsize=64)
def _resize_matrix(n_in: int, n_out: int, order: int) -> NDArray[np.float64]:
    factor = 1 / np.divide(n_in, n_out)
    columns = [
        zoom(impulse, factor, order=order, mode="nearest", grid_mode=True) for impulse in np.eye(n_in)
    ]
    matrix: NDArray[np.float64] = np.stack(columns, axis=1)
    if matrix.shape != (n_out, n_in):
        raise RuntimeError(f"Resize matrix has shape {matrix.shape}, expected {(n_out, n_in)}")
    return matrix


def _z_coordinates(n_in: int, n_out: int) -> NDArray[np.float64]:
    return np.asarray(float(n_in) / n_out * (np.arange(n_out) + 0.5) - 0.5, dtype=np.float64)


@lru_cache(maxsize=64)
def _z_indices(n_in: int, n_out: int) -> NDArray[np.int64]:
    # Let scipy pick the nearest slice so rounding at .5 matches the numpy path.
    picked = map_coordinates(
        np.arange(n_in, dtype=float), [_z_coordinates(n_in, n_out)], order=0, mode="nearest"
    )
    indices: NDArray[np.int64] = np.rint(picked).astype(np.int64)
    return indices


@lru_cache(maxsize=64)
def _z_matrix(n_in: int, n_out: int, order: int) -> NDArray[np.float64]:
    coordinates = [_z_coordinates(n_in, n_out)]
    columns = [
        map_coordinates(impulse, coordinates, order=order, mode="nearest") for impulse in np.eye(n_in)
    ]
    matrix: NDArray[np.float64] = np.stack(columns, axis=1)
    return matrix


@lru_cache(maxsize=64)
def _device_matrix(
    kind: str, n_in: int, n_out: int, order: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # The same few matrices are applied to every channel and class: upload them once.
    matrix = _resize_matrix(n_in, n_out, order) if kind == "resize" else _z_matrix(n_in, n_out, order)
    return torch.from_numpy(matrix).to(device=device, dtype=dtype)


def _apply_along(x: torch.Tensor, weights: torch.Tensor, dim: int) -> torch.Tensor:
    resized: torch.Tensor = torch.tensordot(x, weights, dims=([dim], [1]))
    return resized.movedim(-1, dim)


def resample_torch(
    data: torch.Tensor,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    *,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: bool | None = False,
    separate_z_anisotropy_threshold: float = ANISOTROPY_THRESHOLD,
    dtype: torch.dtype = torch.float64,
    check_finite: bool = True,
) -> torch.Tensor:
    # dtype is the precision of the matrix products; the result has the dtype of data.
    if data.ndim != 4:
        raise ValueError(f"Expected (C, X, Y, Z) data, got shape {tuple(data.shape)}")
    old_shape = tuple(data.shape[1:])
    target = tuple(int(v) for v in new_shape)
    if old_shape == target:
        return data
    if check_finite and not torch.isfinite(data).all():
        # NaN/inf need scipy's exact propagation; leave them to the numpy path.
        raise RuntimeError("non-finite values in the input")

    axis = _separate_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold
    )
    device = data.device
    in_plane = [d for d in range(3) if d != axis]
    # Shrink first so the later, larger products work on less data.
    in_plane.sort(key=lambda d: target[d] / old_shape[d])
    weights = [
        (d, _device_matrix("resize", old_shape[d], target[d], order, device, dtype)) for d in in_plane
    ]

    # The separate axis, if it changes length: slices are picked (order_z == 0) or mixed.
    z_axis = axis if axis is not None and old_shape[axis] != target[axis] else None
    indices = None
    z_weights = None
    select_first = False
    if z_axis is not None and order_z == 0:
        indices = torch.from_numpy(_z_indices(old_shape[z_axis], target[z_axis])).to(device)
        # Picking slices commutes with resizing inside them, so drop slices before the
        # products when the axis shrinks and repeat them afterwards when it grows.
        select_first = target[z_axis] < old_shape[z_axis]
    elif z_axis is not None:
        z_weights = _device_matrix("z", old_shape[z_axis], target[z_axis], order_z, device, dtype)

    output = torch.empty((data.shape[0], *target), dtype=data.dtype, device=device)
    for channel in range(data.shape[0]):
        x = data[channel].to(dtype)
        if z_axis is not None and indices is not None and select_first:
            x = x.index_select(z_axis, indices)
        # skimage clips each resize to its input range: per slice when one axis is separate.
        low = x.amin(dim=in_plane, keepdim=True)
        high = x.amax(dim=in_plane, keepdim=True)
        for d, matrix in weights:
            x = _apply_along(x, matrix, d)
        x = torch.minimum(torch.maximum(x, low), high)
        if z_axis is not None and indices is not None and not select_first:
            x = x.index_select(z_axis, indices)
        elif z_axis is not None and z_weights is not None:
            x = _apply_along(x, z_weights, z_axis)
        output[channel] = x
    return output
