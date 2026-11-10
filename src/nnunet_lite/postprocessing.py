from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ._device import warn_cpu_fallback
from .config import PreprocessingConfig
from .preprocessing import PreprocessingProperties, network_spacing, resampling_kwargs
from .resampling import resample, resample_torch

# Port of nnU-Net's convert_predicted_logits_to_segmentation_with_correct_shape:
# resample the logits to the cropped input shape, argmax, undo the crop, undo the
# transpose. Classes are resampled a few at a time against a running maximum, so the
# full-resolution logits of every class never exist at once; the result is the same
# as an argmax over all of them (ties go to the lowest label, as in numpy).

_CPU_CHUNK = 8


def _argmax_torch(
    logits: torch.Tensor,
    shape: tuple[int, ...],
    current_spacing: tuple[float, ...],
    new_spacing: tuple[float, ...],
    kwargs: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # Works on `device`; logits that live elsewhere are moved there one class at a time.
    if tuple(logits.shape[1:]) == shape:
        return logits.argmax(0).to(dtype)

    def resampled(index: int) -> torch.Tensor:
        # float32 products are ample: the result is rounded to float16 before it is
        # compared. The sliding window has already checked that the logits are finite.
        return resample_torch(
            logits[index : index + 1].to(device),
            shape,
            current_spacing,
            new_spacing,
            dtype=torch.float32,
            check_finite=False,
            **kwargs,
        )[0]

    best = resampled(0)
    labels = torch.zeros(shape, dtype=dtype, device=device)
    for index in range(1, logits.shape[0]):
        candidate = resampled(index)
        labels.masked_fill_(candidate > best, index)
        torch.maximum(best, candidate, out=best)
    return labels


def _argmax_numpy(
    logits: torch.Tensor,
    shape: tuple[int, ...],
    current_spacing: tuple[float, ...],
    new_spacing: tuple[float, ...],
    kwargs: dict[str, Any],
    dtype: torch.dtype,
) -> torch.Tensor:
    array = logits.cpu().numpy()
    best = None
    labels = np.zeros(shape, dtype=np.int64)
    for start in range(0, array.shape[0], _CPU_CHUNK):
        # float16 in, float16 out: the rounding nnU-Net applies before its argmax.
        chunk = resample(
            array[start : start + _CPU_CHUNK], shape, current_spacing, new_spacing, **kwargs
        )
        candidate = chunk.max(axis=0)
        candidate_labels = chunk.argmax(axis=0) + start
        if best is None:
            best, labels = candidate, candidate_labels
        else:
            better = candidate > best
            labels[better] = candidate_labels[better]
            best = np.where(better, candidate, best)
    return torch.from_numpy(labels).to(dtype)


def restore_segmentation(
    logits: torch.Tensor,
    properties: PreprocessingProperties,
    config: PreprocessingConfig,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    # logits: (classes, X, Y, Z) at network resolution. Returns integer labels on the
    # CPU with the shape and axis order of the array that was preprocessed.
    # device: where to resample, by default where the logits are. Logits that did not
    # fit on the GPU can still be resampled there, one class at a time.
    kwargs = resampling_kwargs(config.resampling_probabilities)
    current_spacing = network_spacing(config, properties.spacing)
    shape = properties.shape_after_cropping
    dtype = torch.uint8 if logits.shape[0] - 1 <= 255 else torch.int16
    compute = logits.device if device is None else torch.device(device)

    cropped = None
    if compute.type != "cpu":
        try:
            cropped = _argmax_torch(
                logits, shape, current_spacing, properties.spacing, kwargs, dtype, compute
            ).cpu()
        except Exception as error:  # CUDA out-of-memory, unsupported device or dtype, ...
            warn_cpu_fallback("Restoring the segmentation", compute, error)
    if cropped is None:
        cropped = _argmax_numpy(logits, shape, current_spacing, properties.spacing, kwargs, dtype)

    segmentation = torch.zeros(properties.shape_before_cropping, dtype=dtype)
    segmentation[tuple(slice(start, end) for start, end in properties.crop_bbox)] = cropped
    return segmentation.permute(*config.transpose_backward).contiguous()
