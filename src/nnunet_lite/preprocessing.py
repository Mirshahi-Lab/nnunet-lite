from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes

from .config import ChannelConfig, PreprocessingConfig, ResamplingConfig
from .resampling import SUPPORTED_FUNCTION, compute_new_shape, resample, resample_torch

# Port of nnU-Net's DefaultPreprocessor.run_case_npy for inference (no
# segmentation): transpose -> crop to nonzero -> normalize -> resample.
# On CPU the output is byte-equal to nnU-Net for the same input array and
# spacing. On other devices the same steps run in torch: equal up to
# floating-point rounding, not bit for bit.

_EPS = 1e-8
_NORMALIZATIONS = (
    "CTNormalization",
    "ZScoreNormalization",
    "RescaleTo01Normalization",
    "RGBTo01Normalization",
    "NoNormalization",
)


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


def _preprocess_numpy(
    data: NDArray[Any] | torch.Tensor,
    original_spacing: tuple[float, ...],
    config: PreprocessingConfig,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, PreprocessingProperties]:
    array = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    # astype copies, so the caller's array is never modified.
    volume: NDArray[np.float32] = array.astype(np.float32).transpose(
        [0, *[axis + 1 for axis in config.transpose_forward]]
    )
    shape_before_cropping = volume.shape[1:]
    volume, mask, bbox = _crop_to_nonzero(volume)
    target_spacing = _target_spacing(config, original_spacing)
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


def _normalize_torch(image: torch.Tensor, channel: ChannelConfig, mask: torch.Tensor | None) -> None:
    # In place, on a view of the volume.
    scheme = channel.normalization
    if scheme == "CTNormalization":
        image.clamp_(channel.lower, channel.upper)
        image -= channel.mean
        image /= max(channel.std, _EPS)
    elif scheme == "ZScoreNormalization":
        values = image if mask is None else image[mask]
        mean = values.mean(dtype=torch.float64).to(image.dtype)
        std = (values - mean).square().mean(dtype=torch.float64).sqrt().to(image.dtype)
        values = (values - mean) / std.clamp(min=_EPS)
        if mask is None:
            image.copy_(values)
        else:
            image[mask] = values
    elif scheme == "RescaleTo01Normalization":
        image -= image.min()
        image /= image.max().clamp(min=_EPS)
    elif scheme == "RGBTo01Normalization":
        if image.min() < 0 or image.max() > 255:
            raise ValueError("RGBTo01Normalization expects values in [0, 255]")
        image /= 255.0


def _preprocess_torch(
    data: NDArray[Any] | torch.Tensor,
    original_spacing: tuple[float, ...],
    config: PreprocessingConfig,
    kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, PreprocessingProperties]:
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only arrays: the copy below makes it safe
            tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    # copy=True, so the caller's data is never modified.
    volume = tensor.to(device=device, dtype=torch.float32, copy=True).permute(
        0, *[axis + 1 for axis in config.transpose_forward]
    )
    shape_before_cropping = tuple(volume.shape[1:])

    # Filling holes never changes the bounding box, so the box needs no fill.
    nonzero = (volume != 0).any(dim=0)
    bbox = []
    for axis in range(3):
        occupied = nonzero.any(dim=[a for a in range(3) if a != axis]).nonzero().flatten()
        bbox.append(
            (int(occupied[0]), int(occupied[-1]) + 1) if occupied.numel() else (0, nonzero.shape[axis])
        )
    slicer = tuple(slice(start, end) for start, end in bbox)
    volume = volume[(slice(None), *slicer)]
    mask = None
    if any(c.normalization == "ZScoreNormalization" and c.use_mask_for_norm for c in config.channels):
        # The filled mask itself has no GPU equivalent; this one step runs on CPU.
        filled = binary_fill_holes(nonzero[slicer].cpu().numpy())
        mask = torch.from_numpy(filled).to(device)

    target_spacing = _target_spacing(config, original_spacing)
    shape_after_cropping = tuple(volume.shape[1:])
    new_shape = compute_new_shape(shape_after_cropping, original_spacing, target_spacing)
    for index, channel in enumerate(config.channels):
        _normalize_torch(volume[index], channel, mask if channel.use_mask_for_norm else None)
    volume = resample_torch(volume, new_shape, original_spacing, target_spacing, **kwargs)

    properties = PreprocessingProperties(
        spacing=original_spacing,
        shape_before_cropping=shape_before_cropping,
        crop_bbox=tuple(bbox),
        shape_after_cropping=shape_after_cropping,
    )
    return volume.contiguous(), properties


def _target_spacing(
    config: PreprocessingConfig, original_spacing: tuple[float, ...]
) -> tuple[float, ...]:
    if len(config.target_spacing) < len(original_spacing):
        # 2D configurations keep the spacing between slices.
        return (original_spacing[0], *config.target_spacing)
    return config.target_spacing


def preprocess(
    data: NDArray[Any] | torch.Tensor,
    spacing: Sequence[float],
    config: PreprocessingConfig,
    *,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, PreprocessingProperties]:
    shape = tuple(data.shape)
    if len(shape) != 4:
        raise ValueError(
            f"Expected data of shape (C, X, Y, Z), got {shape}. Add a leading channel "
            "axis to a single-channel volume, and a singleton first spatial axis to a 2D image."
        )
    if shape[0] != len(config.channels):
        raise ValueError(
            f"Model expects {len(config.channels)} channel(s) "
            f"({', '.join(c.name for c in config.channels)}), got {shape[0]}"
        )
    if len(spacing) != 3:
        raise ValueError(f"Expected one spacing value per spatial axis (3), got {len(spacing)}")
    for channel in config.channels:
        if channel.normalization not in _NORMALIZATIONS:
            raise NotImplementedError(f"Unsupported normalization scheme: {channel.normalization}")
    kwargs = resampling_kwargs(config.resampling_data)
    original_spacing = tuple(float(spacing[axis]) for axis in config.transpose_forward)

    if device is None:
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target = torch.device(device)
    if target.type != "cpu":
        try:
            return _preprocess_torch(data, original_spacing, config, kwargs, target)
        except ValueError:
            raise  # invalid input, not a device failure
        except Exception as error:  # CUDA out-of-memory, unsupported device or dtype, ...
            reason = (str(error).strip() or type(error).__name__).splitlines()[0]
            warnings.warn(
                f"Preprocessing on {target} failed ({reason}); falling back to CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            if target.type == "cuda":
                torch.cuda.empty_cache()
    return _preprocess_numpy(data, original_spacing, config, kwargs)
