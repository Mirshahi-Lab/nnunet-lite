"""QC statistics on hand-built segmentations and logits."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nnunet_lite.qc import component_stats, confidence_stats, resolve_labels

LABELS = {0: "background", 1: "organ", 2: "lesion"}


def test_resolve_labels() -> None:
    assert resolve_labels(False, LABELS) == ()
    assert resolve_labels(True, LABELS) == (1, 2)
    assert resolve_labels(["lesion"], LABELS) == (2,)
    assert resolve_labels("lesion", LABELS) == (2,)
    assert resolve_labels([2, "organ", 1], LABELS) == (1, 2)
    assert resolve_labels([], LABELS) == ()
    for bad in (["spleen"], [7], [0], ["background"]):
        with pytest.raises(ValueError, match="QC"):
            resolve_labels(bad, LABELS)


def segmentation() -> np.ndarray:
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[0:2, 0:2, 0:2] = 1  # 8 voxels on the boundary
    volume[5:7, 5:7, 5:7] = 1  # 8 voxels inside
    volume[3:5, 3:5, 3] = 2  # 4 voxels inside, diagonal neighbour of nothing else
    volume[1, 5, 5] = 2  # a second, single-voxel component
    return volume


def test_component_stats_are_per_label() -> None:
    volume = segmentation()
    before = volume.copy()
    stats = component_stats(volume, (1, 2))
    assert np.array_equal(volume, before)
    assert stats[1] == {
        "voxel_count": 16,
        "component_count": 2,
        "largest_component_fraction": 0.5,
        "small_component_voxel_counts": (8, 8),
        "touches_boundary": True,
    }
    assert stats[2] == {
        "voxel_count": 5,
        "component_count": 2,
        "largest_component_fraction": 0.8,
        "small_component_voxel_counts": (1, 4),
        "touches_boundary": False,
    }


def test_component_stats_for_selected_and_absent_labels() -> None:
    volume = segmentation()
    assert component_stats(volume, (2,)) == {2: component_stats(volume, (1, 2))[2]}
    assert component_stats(volume, ()) == {}
    absent = component_stats(volume, (5,))[5]
    assert absent == {
        "voxel_count": 0,
        "component_count": 0,
        "largest_component_fraction": 0.0,
        "small_component_voxel_counts": (),
        "touches_boundary": False,
    }
    large = component_stats(volume, (1,), small_component_voxels=8)[1]
    assert large["small_component_voxel_counts"] == ()


def test_component_stats_on_a_2d_image() -> None:
    volume = np.zeros((1, 8, 8), dtype=np.uint8)
    volume[0, 3:5, 3:5] = 1
    assert component_stats(volume, (1,))[1]["touches_boundary"] is False
    volume[0, 0, 0] = 1
    assert component_stats(volume, (1,))[1]["touches_boundary"] is True


def test_confidence_of_certain_and_uncertain_predictions() -> None:
    logits = torch.zeros(3, 4, 4, 4)
    logits[0] = 30.0
    logits[0, :2] = 0.0
    logits[1, :2] = 30.0  # half the volume is confidently label 1
    stats = confidence_stats(logits, (1, 2))
    assert stats[1]["network_voxel_count"] == 32
    assert stats[1]["estimated_dice"] == pytest.approx(1.0, abs=1e-6)
    assert stats[1]["mean_margin"] == pytest.approx(1.0, abs=1e-6)
    assert stats[1]["mean_entropy"] == pytest.approx(0.0, abs=1e-6)
    assert stats[1]["low_margin_fraction"] == 0.0
    assert stats[2] == {
        "estimated_dice": 0.0,
        "mean_margin": 0.0,
        "low_margin_fraction": 0.0,
        "mean_entropy": 0.0,
        "network_voxel_count": 0,
    }

    uniform = confidence_stats(torch.zeros(3, 4, 4, 4), (1,), temperature=1.0)
    assert uniform[1]["network_voxel_count"] == 0  # ties go to the background
    tied = torch.zeros(3, 4, 4, 4)
    tied[1] = 1e-3
    almost = confidence_stats(tied, (1,))[1]
    assert almost["mean_entropy"] == pytest.approx(1.0, abs=1e-4)
    assert almost["mean_margin"] == pytest.approx(0.0, abs=1e-3)
    assert almost["low_margin_fraction"] == 1.0
    assert almost["estimated_dice"] == pytest.approx(0.5, abs=1e-3)  # 2 * N/3 / (N/3 + N)


@pytest.mark.parametrize("temperature", [1.0, 2.5])
def test_confidence_matches_an_unchunked_computation(temperature: float) -> None:
    torch.manual_seed(0)
    logits = (torch.randn(4, 40, 5, 6) * 3).half()  # 40 slices: three chunks
    stats = confidence_stats(logits, (1, 2, 3), temperature=temperature)
    probability = (logits.double() / temperature).softmax(0)
    prediction = probability.argmax(0)
    top = probability.topk(2, dim=0).values
    margin = top[0] - top[1]
    entropy = -(probability * probability.log()).sum(0) / np.log(4)
    for label in (1, 2, 3):
        mask = prediction == label
        expected_dice = 2 * probability[label][mask].sum() / (probability[label].sum() + mask.sum())
        assert stats[label]["network_voxel_count"] == int(mask.sum())
        assert stats[label]["estimated_dice"] == pytest.approx(float(expected_dice), abs=1e-5)
        assert stats[label]["mean_margin"] == pytest.approx(float(margin[mask].mean()), abs=1e-5)
        assert stats[label]["mean_entropy"] == pytest.approx(float(entropy[mask].mean()), abs=1e-5)
        assert stats[label]["low_margin_fraction"] == pytest.approx(
            float((margin[mask] < 0.5).double().mean()), abs=1e-3
        )
