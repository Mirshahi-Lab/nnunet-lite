from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cc3d
import fastremap
import numpy as np
import torch
from numpy.typing import NDArray

# Label-free quality control for a predicted segmentation, one record per label:
# connected-component statistics of the final label map, and confidence statistics
# of the softmax probabilities at network resolution. Nothing here changes the
# segmentation.

LOW_MARGIN = 0.5
SMALL_COMPONENT_VOXELS = 100
_CHUNK_SLICES = 16


@dataclass(frozen=True)
class LabelQC:
    # Connected components (26-connectivity) of the label in the output segmentation.
    voxel_count: int
    component_count: int
    largest_component_fraction: float
    small_component_voxel_counts: tuple[int, ...]
    touches_boundary: bool
    # Confidence over the voxels predicted as this label, at network resolution.
    # estimated_dice is the Dice the prediction would score if the model's own
    # probabilities were the truth; it is not calibrated against real Dice.
    estimated_dice: float
    mean_margin: float
    low_margin_fraction: float
    mean_entropy: float
    network_voxel_count: int


def resolve_labels(selection: bool | Sequence[int | str], labels: Mapping[int, str]) -> tuple[int, ...]:
    # False -> nothing, True -> every foreground label, otherwise label names or ids.
    if selection is False:
        return ()
    if selection is True:
        return tuple(sorted(index for index in labels if index != 0))
    if isinstance(selection, str):
        selection = [selection]
    by_name = {name: index for index, name in labels.items()}
    resolved = set()
    for item in selection:
        index = by_name.get(item) if isinstance(item, str) else int(item)
        if index is None or index not in labels:
            raise ValueError(f"Unknown label for QC: {item!r}")
        if index == 0:
            raise ValueError("QC is not available for the background label")
        resolved.add(index)
    return tuple(sorted(resolved))


def component_stats(
    segmentation: NDArray[Any],
    label_ids: Sequence[int],
    *,
    small_component_voxels: int = SMALL_COMPONENT_VOXELS,
) -> dict[int, dict[str, Any]]:
    # Components never span two labels, so every label is analysed on its own; labels
    # that were not asked for are removed first and cost nothing.
    selected = [int(label) for label in label_ids]
    if set(selected) >= set(range(1, int(segmentation.max(initial=0)) + 1)):
        volume = np.ascontiguousarray(segmentation)  # nothing to remove
    else:
        volume = np.ascontiguousarray(fastremap.mask_except(segmentation, selected))
    components, count = cc3d.connected_components(volume, connectivity=26, return_N=True)
    sizes = np.zeros(count + 1, dtype=np.int64)
    found, voxels = fastremap.unique(components, return_counts=True)
    sizes[found] = voxels
    component_labels = np.zeros(count + 1, dtype=np.int64)
    for component, label in fastremap.component_map(components, volume).items():
        component_labels[int(component)] = int(label)

    faces = []
    for axis in range(components.ndim):
        if components.shape[axis] > 1:  # a singleton axis (2D image) has no boundary
            faces += [
                np.take(components, 0, axis=axis).ravel(),
                np.take(components, -1, axis=axis).ravel(),
            ]
    boundary = np.unique(np.concatenate(faces)) if faces else np.zeros(0, dtype=np.int64)
    boundary_labels = {int(v) for v in component_labels[boundary]}

    stats: dict[int, dict[str, Any]] = {}
    for label in label_ids:
        ids = np.flatnonzero(component_labels == label)
        label_sizes = sizes[ids]
        total = int(label_sizes.sum())
        stats[label] = {
            "voxel_count": total,
            "component_count": len(ids),
            "largest_component_fraction": float(label_sizes.max() / total) if total else 0.0,
            "small_component_voxel_counts": tuple(
                sorted(int(v) for v in label_sizes if v < small_component_voxels)
            ),
            "touches_boundary": bool(total) and label in boundary_labels,
        }
    return stats


@torch.inference_mode()
def confidence_stats(
    logits: torch.Tensor, label_ids: Sequence[int], *, temperature: float = 1.0
) -> dict[int, dict[str, Any]]:
    # logits: (classes, X, Y, Z). Sums are accumulated over chunks of the first spatial
    # axis so the probabilities of a many-label model never exist all at once.
    classes = logits.shape[0]
    if classes < 2:
        raise ValueError("Confidence QC needs at least two classes")
    device = logits.device

    def zeros() -> torch.Tensor:
        return torch.zeros(classes, dtype=torch.float64, device=device)

    count, margin_sum, low_margin, entropy_sum = zeros(), zeros(), zeros(), zeros()
    probability_sum, probability_on_prediction = zeros(), zeros()
    log_classes = float(np.log(classes))
    for start in range(0, logits.shape[1], _CHUNK_SLICES):
        chunk = logits[:, start : start + _CHUNK_SLICES].float().flatten(1)
        probability = (chunk / temperature).softmax(0)
        top, prediction = probability.max(0)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(0) / log_classes
        probability_sum += probability.sum(1).double()
        # The runner-up is the maximum once the winner is taken out; the probabilities
        # are not needed after this, so they are overwritten.
        probability.scatter_(0, prediction[None], -1.0)
        margin = top - probability.amax(0)
        count += torch.bincount(prediction, minlength=classes)
        margin_sum += torch.bincount(prediction, margin.double(), minlength=classes)
        low_margin += torch.bincount(prediction, (margin < LOW_MARGIN).double(), minlength=classes)
        entropy_sum += torch.bincount(prediction, entropy.double(), minlength=classes)
        probability_on_prediction += torch.bincount(prediction, top.double(), minlength=classes)

    voxels = count.clamp_min(1)
    estimated_dice = (2 * probability_on_prediction / (probability_sum + count).clamp_min(1e-6)).cpu()
    mean_margin = (margin_sum / voxels).cpu()
    low_margin_fraction = (low_margin / voxels).cpu()
    mean_entropy = (entropy_sum / voxels).cpu()
    count = count.cpu()
    return {
        label: {
            "estimated_dice": float(estimated_dice[label]),
            "mean_margin": float(mean_margin[label]),
            "low_margin_fraction": float(low_margin_fraction[label]),
            "mean_entropy": float(mean_entropy[label]),
            "network_voxel_count": int(count[label]),
        }
        for label in label_ids
    }
