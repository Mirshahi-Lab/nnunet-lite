"""Whole-image segmentations against nnU-Net's own predictor.

Needs nnunetv2, CUDA, real model folders (see test_nnunet_equivalence.py) and a CT:
NNUNET_LITE_TEST_IMAGE, defaulting to a small case from the leadup data. Skipped when
any of them is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from test_nnunet_equivalence import MODEL_DIRS

import nnunet_lite as nl

pytest.importorskip("nnunetv2")

IMAGE = Path(
    os.environ.get(
        "NNUNET_LITE_TEST_IMAGE",
        Path(__file__).parents[1]
        / "packages/nnunet_model_conversion/data/nnunet_raw/Dataset997_ChestAbd/imagesTr"
        / "s0003_0000.nii.gz",
    )
)
# The gate used for the leadup validation cohort.
MIN_AGREEMENT = 0.999
MIN_DICE = 0.999
MIN_VOXELS_FOR_DICE = 1000

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not IMAGE.is_file(), reason=f"test image not found: {IMAGE}"),
]


def nnunet_segmentation(folder: Path) -> np.ndarray:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(folder), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    image, properties = predictor.plans_manager.image_reader_writer_class().read_images([str(IMAGE)])
    segmentation: np.ndarray = predictor.predict_single_npy_array(image, properties, None, None, False)
    del predictor
    torch.cuda.empty_cache()
    return segmentation


@pytest.mark.parametrize("folder", MODEL_DIRS, ids=[p.parent.name for p in MODEL_DIRS])
def test_segmentation_matches_nnunet(folder: Path) -> None:
    expected = nnunet_segmentation(folder)
    result = nl.Predictor(nl.convert(folder), device="cuda").predict(IMAGE, qc=True)
    actual = result.segmentation.numpy()
    torch.cuda.empty_cache()

    assert actual.shape == expected.shape
    assert (actual == expected).mean() > MIN_AGREEMENT
    for label in np.unique(expected):
        reference, candidate = expected == label, actual == label
        if label == 0 or reference.sum() < MIN_VOXELS_FOR_DICE:
            continue
        dice = 2 * (reference & candidate).sum() / (reference.sum() + candidate.sum())
        assert dice > MIN_DICE, f"{result.labels[int(label)]}: Dice {dice:.5f}"

    assert result.qc is not None
    for index, name in result.labels.items():
        if index:
            assert result.qc[name].voxel_count == int((actual == index).sum())
