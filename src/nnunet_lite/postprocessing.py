from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

from .config import PreprocessingConfig
from .preprocessing import PreprocessingProperties, _target_spacing, resampling_kwargs
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
) -> torch.Tensor:
    if tuple(logits.shape[1:]) == shape:
        return logits.argmax(0).to(dtype)
    best = resample_torch(logits[:1], shape, current_spacing, new_spacing, **kwargs)[0]
    labels = torch.zeros(shape, dtype=dtype, device=logits.device)
    for index in range(1, logits.shape[0]):
        candidate = resample_torch(
            logits[index : index + 1], shape, current_spacing, new_spacing, **kwargs
        )[0]
        better = candidate > best
        labels[better] = index
        best = torch.where(better, candidate, best)
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
    logits: torch.Tensor, properties: PreprocessingProperties, config: PreprocessingConfig
) -> torch.Tensor:
    # logits: (classes, X, Y, Z) at network resolution. Returns integer labels on the
    # CPU with the shape and axis order of the array that was preprocessed.
    kwargs = resampling_kwargs(config.resampling_probabilities)
    current_spacing = _target_spacing(config, properties.spacing)
    shape = properties.shape_after_cropping
    dtype = torch.uint8 if logits.shape[0] - 1 <= 255 else torch.int16

    cropped = None
    if logits.device.type != "cpu":
        try:
            cropped = _argmax_torch(logits, shape, current_spacing, properties.spacing, kwargs, dtype)
            cropped = cropped.cpu()
        except Exception as error:  # CUDA out-of-memory, unsupported device or dtype, ...
            reason = (str(error).strip() or type(error).__name__).splitlines()[0]
            warnings.warn(
                f"Restoring the segmentation on {logits.device} failed ({reason}); falling back to CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            if logits.device.type == "cuda":
                torch.cuda.empty_cache()
    if cropped is None:
        cropped = _argmax_numpy(logits, shape, current_spacing, properties.spacing, kwargs, dtype)

    segmentation = torch.zeros(properties.shape_before_cropping, dtype=dtype)
    segmentation[tuple(slice(start, end) for start, end in properties.crop_bbox)] = cropped
    return segmentation.permute(*config.transpose_backward).contiguous()
