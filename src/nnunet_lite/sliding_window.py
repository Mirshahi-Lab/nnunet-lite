from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.ndimage import gaussian_filter

from .config import BundleConfig

# Port of nnU-Net's sliding-window prediction (nnUNetPredictor): same tile positions,
# Gaussian weighting, padding and half-precision accumulation. Patches may be batched,
# which nnU-Net does not do; the per-patch results are the same.

_Tile = tuple[slice | int, ...]


def compute_steps(
    image_size: Sequence[int], tile_size: Sequence[int], tile_step_size: float
) -> list[list[int]]:
    if any(image < tile for image, tile in zip(image_size, tile_size, strict=True)):
        raise ValueError("Image must be padded to at least the tile size")
    if not 0 < tile_step_size <= 1:
        raise ValueError("tile_step_size must be in (0, 1]")
    steps: list[list[int]] = []
    for image, tile in zip(image_size, tile_size, strict=True):
        count = int(np.ceil((image - tile) / (tile * tile_step_size))) + 1
        maximum = image - tile
        actual = maximum / (count - 1) if count > 1 else 99999999999
        steps.append([int(np.round(actual * index)) for index in range(count)])
    return steps


@lru_cache(maxsize=8)
def gaussian_importance_map(
    tile_size: tuple[int, ...], sigma_scale: float, value_scaling_factor: float
) -> torch.Tensor:
    impulse = np.zeros(tile_size)
    impulse[tuple(size // 2 for size in tile_size)] = 1
    gaussian = torch.from_numpy(
        gaussian_filter(impulse, [size * sigma_scale for size in tile_size], 0, mode="constant", cval=0)
    )
    gaussian /= torch.max(gaussian) / value_scaling_factor
    gaussian = gaussian.to(torch.float16)
    zero = gaussian == 0
    gaussian[zero] = torch.min(gaussian[~zero])
    return gaussian


def pad_to_patch(
    image: torch.Tensor, patch_size: Sequence[int]
) -> tuple[torch.Tensor, tuple[slice, ...]]:
    # Pads the trailing len(patch_size) axes symmetrically with zeros (the extra voxel
    # goes at the end) and returns the slices that undo it.
    ndim = len(patch_size)
    sizes = image.shape[-ndim:]
    difference = [max(size, patch) - size for size, patch in zip(sizes, patch_size, strict=True)]
    below = [d // 2 for d in difference]
    above = [d // 2 + d % 2 for d in difference]
    padding = [
        v for low, high in zip(reversed(below), reversed(above), strict=True) for v in (low, high)
    ]
    padded = functional.pad(image, padding, "constant", 0) if any(difference) else image
    revert = tuple(slice(None) for _ in range(image.ndim - ndim)) + tuple(
        slice(low, low + size) for low, size in zip(below, sizes, strict=True)
    )
    return padded, revert


def _tiles(shape: Sequence[int], patch_size: Sequence[int], tile_step_size: float) -> list[_Tile]:
    if len(patch_size) == len(shape) - 1:
        # 2D network on a stack of slices: every slice of the first axis is tiled in-plane.
        steps = compute_steps(shape[1:], patch_size, tile_step_size)
        return [
            (index, *(slice(s, s + p) for s, p in zip(start, patch_size, strict=True)))
            for index in range(shape[0])
            for start in itertools.product(*steps)
        ]
    steps = compute_steps(shape, patch_size, tile_step_size)
    return [
        tuple(slice(s, s + p) for s, p in zip(start, patch_size, strict=True))
        for start in itertools.product(*steps)
    ]


def _forward(
    network: torch.nn.Module, batch: torch.Tensor, mirror_axes: tuple[int, ...]
) -> torch.Tensor:
    prediction: torch.Tensor = network(batch)
    if mirror_axes:
        axes = [axis + 2 for axis in mirror_axes]
        combinations = [
            c for length in range(1, len(axes) + 1) for c in itertools.combinations(axes, length)
        ]
        for combination in combinations:
            prediction += torch.flip(network(torch.flip(batch, combination)), combination)
        prediction /= len(combinations) + 1
    return prediction


def _accumulate(
    network: torch.nn.Module,
    data: torch.Tensor,
    tiles: list[_Tile],
    config: BundleConfig,
    network_device: torch.device,
    results_device: torch.device,
    mirror_axes: tuple[int, ...],
    batch_size: int,
) -> torch.Tensor:
    data = data.to(results_device)
    spatial = data.shape[1:]
    logits = torch.zeros(
        (config.network.output_channels, *spatial), dtype=torch.float16, device=results_device
    )
    counts = torch.zeros(spatial, dtype=torch.float16, device=results_device)
    inference = config.inference
    weight: torch.Tensor | float = 1.0
    if inference.use_gaussian:
        weight = gaussian_importance_map(
            config.patch_size, inference.gaussian_sigma_scale, inference.gaussian_value_scaling_factor
        ).to(results_device)

    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start : start + batch_size]
        batch = torch.stack([data[(slice(None), *tile)] for tile in batch_tiles]).to(
            network_device, memory_format=torch.contiguous_format
        )
        with torch.autocast("cuda", enabled=network_device.type == "cuda"):
            predictions = _forward(network, batch, mirror_axes)
        predictions = predictions.to(results_device)
        for tile, prediction in zip(batch_tiles, predictions, strict=True):
            if inference.use_gaussian:
                prediction *= weight
            logits[(slice(None), *tile)] += prediction
            counts[tile] += weight
    torch.div(logits, counts, out=logits)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Sliding-window prediction produced non-finite logits")
    return logits


@torch.inference_mode()
def predict_logits(
    network: torch.nn.Module,
    data: torch.Tensor,
    config: BundleConfig,
    *,
    mirror: bool = False,
    batch_size: int = 1,
) -> torch.Tensor:
    # data: preprocessed (C, X, Y, Z). Returns float16 logits (classes, X, Y, Z) on the
    # network's device, or on the CPU if they did not fit there.
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if data.ndim != 4 or data.shape[0] != config.network.input_channels:
        raise ValueError(
            f"Expected preprocessed data of shape ({config.network.input_channels}, X, Y, Z), "
            f"got {tuple(data.shape)}"
        )
    patch_size = config.patch_size
    padded, revert = pad_to_patch(data, patch_size)
    tiles = _tiles(tuple(padded.shape[1:]), patch_size, config.inference.tile_step_size)
    mirror_axes = config.mirror_axes if mirror else ()
    network_device = next(network.parameters()).device

    logits = None
    if network_device.type == "cuda":
        try:
            logits = _accumulate(
                network, padded, tiles, config, network_device, network_device, mirror_axes, batch_size
            )
        except torch.OutOfMemoryError:
            warnings.warn(
                "CUDA out of memory during sliding-window prediction; accumulating on CPU instead",
                RuntimeWarning,
                stacklevel=2,
            )
            torch.cuda.empty_cache()
    if logits is None:
        logits = _accumulate(
            network, padded, tiles, config, network_device, torch.device("cpu"), mirror_axes, batch_size
        )
    return logits[revert]
