"""nl.preprocess against nnU-Net's DefaultPreprocessor.run_case_npy: byte-equal output.

Needs nnunetv2, so it is skipped in the default environment.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from test_convert import LEGACY_3D, RESAMPLING
from test_nnunet_equivalence import MODEL_DIRS

import nnunet_lite as nl
from nnunet_lite.convert import _preprocessing_config, _resolve_configuration

pytest.importorskip("nnunetv2")

INTENSITY = {"percentile_00_5": -150.0, "percentile_99_5": 250.0, "mean": 20.5, "std": 77.25}


def configuration(**overrides: Any) -> dict[str, Any]:
    # Keys nnU-Net's plans loader needs but the converter does not read.
    nnunet_only = {
        "num_pool_per_axis": [2, 2, 2],
        "resampling_fn_seg": "resample_data_or_seg_to_shape",
        "resampling_fn_seg_kwargs": {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        },
    }
    return {**copy.deepcopy(LEGACY_3D), **nnunet_only, **overrides}


def resampling(**kwargs: Any) -> dict[str, Any]:
    result = copy.deepcopy(RESAMPLING)
    result["resampling_fn_data_kwargs"].update(kwargs)
    return result


# name -> (plans overrides, configuration, channel names)
PLANS: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, str]]] = {
    "ct_isotropic": ({}, configuration(), {"0": "CT"}),
    "ct_anisotropic_target": ({}, configuration(spacing=[5.0, 0.9, 0.9]), {"0": "CT"}),
    "ct_order1": ({}, configuration(**resampling(order=1, order_z=1)), {"0": "CT"}),
    "ct_force_separate_z": ({}, configuration(**resampling(force_separate_z=True)), {"0": "CT"}),
    "ct_never_separate_z": ({}, configuration(**resampling(force_separate_z=False)), {"0": "CT"}),
    "ct_transposed": (
        {"transpose_forward": [2, 0, 1], "transpose_backward": [1, 2, 0]},
        configuration(spacing=[3.0, 1.0, 1.0]),
        {"0": "CT"},
    ),
    "ct_2d": ({}, configuration(spacing=[1.25, 1.25], patch_size=[16, 16]), {"0": "CT"}),
    "mri_masked_zscore": (
        {},
        configuration(
            normalization_schemes=["ZScoreNormalization", "ZScoreNormalization"],
            use_mask_for_norm=[True, False],
        ),
        {"0": "T1", "1": "T2"},
    ),
    "mixed_schemes": (
        {},
        configuration(
            normalization_schemes=[
                "RescaleTo01Normalization",
                "NoNormalization",
                "CTNormalization",
            ],
            use_mask_for_norm=[False, False, False],
        ),
        {"0": "a", "1": "b", "2": "CT"},
    ),
}

# name -> (spatial shape, spacing)
GEOMETRIES: dict[str, tuple[tuple[int, int, int], tuple[float, float, float]]] = {
    "isotropic_fine": ((30, 44, 40), (1.0, 1.0, 1.0)),
    "matches_target": ((30, 44, 40), (1.5, 1.5, 1.5)),
    "thick_slices": ((12, 50, 46), (5.0, 0.8, 0.8)),
    "thick_middle_axis": ((40, 10, 44), (0.9, 4.5, 0.9)),
    "thick_last_axis": ((44, 40, 11), (0.7, 0.7, 4.0)),
    "two_coarse_axes": ((14, 16, 60), (4.0, 4.0, 1.0)),
    "odd_spacing": ((33, 37, 41), (1.37, 0.91, 2.6)),
}


def make_volume(shape: tuple[int, ...], seed: int, *, border: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = gaussian_filter(rng.normal(0, 1, shape), sigma=(0, 2, 2, 2)) * 4000
    data = (data + rng.normal(0, 30, shape)).astype(np.float32)
    if border:
        # Zero background with a zero hole inside the body, as in masked MRI or padded CT.
        data[:, :3] = 0
        data[:, :, -5:] = 0
        data[:, :, :, :2] = 0
        centre = tuple(s // 2 for s in shape[1:])
        data[:, centre[0] - 1 : centre[0] + 2, centre[1] - 2 : centre[1] + 2, centre[2]] = 0
    return data


def run_nnunet(
    data: np.ndarray, spacing: tuple[float, ...], plans: dict[str, Any], name: str, dataset: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    manager = PlansManager(copy.deepcopy(plans))
    reference, _, properties = DefaultPreprocessor(verbose=False).run_case_npy(
        data.copy(), None, {"spacing": list(spacing)}, manager, manager.get_configuration(name), dataset
    )
    return reference, properties


def assert_equal_to_nnunet(
    data: np.ndarray, spacing: tuple[float, ...], plans: dict[str, Any], name: str, dataset: Any
) -> None:
    reference, reference_properties = run_nnunet(data, spacing, plans, name, dataset)
    config = _preprocessing_config(plans, _resolve_configuration(plans, name), dataset)
    output, properties = nl.preprocess(data, spacing, config)

    assert output.numpy().dtype == reference.dtype
    assert output.numpy().shape == reference.shape
    assert np.array_equal(output.numpy(), reference, equal_nan=True)
    assert properties.shape_before_cropping == tuple(reference_properties["shape_before_cropping"])
    assert [list(b) for b in properties.crop_bbox] == [
        list(b) for b in reference_properties["bbox_used_for_cropping"]
    ]
    assert properties.shape_after_cropping == tuple(
        reference_properties["shape_after_cropping_and_before_resampling"]
    )


@pytest.mark.parametrize("border", [False, True], ids=["full", "zero_border"])
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("plan", sorted(PLANS))
def test_synthetic_plans_byte_equal_to_nnunet(plan: str, geometry: str, border: bool) -> None:
    overrides, config, channel_names = PLANS[plan]
    plans = {
        "dataset_name": "Dataset001_Synthetic",
        "plans_name": "nnUNetPlans",
        "image_reader_writer": "NibabelIOWithReorient",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "foreground_intensity_properties_per_channel": dict.fromkeys(channel_names, INTENSITY),
        "configurations": {"test": config},
        **overrides,
    }
    dataset = {"channel_names": channel_names, "labels": {"background": 0, "organ": 1}}
    shape, spacing = GEOMETRIES[geometry]
    data = make_volume((len(channel_names), *shape), seed=len(plan) + len(geometry), border=border)
    assert_equal_to_nnunet(data, spacing, plans, "test", dataset)


@pytest.mark.parametrize("folder", MODEL_DIRS, ids=[p.parent.name for p in MODEL_DIRS])
@pytest.mark.parametrize("geometry", ["isotropic_fine", "thick_slices", "odd_spacing"])
def test_real_plans_byte_equal_to_nnunet(folder: Path, geometry: str) -> None:
    import torch

    plans = json.loads((folder / "plans.json").read_text())
    dataset = json.loads((folder / "dataset.json").read_text())
    checkpoint = torch.load(folder / "fold_0/checkpoint_final.pth", weights_only=False)
    shape, spacing = GEOMETRIES[geometry]
    data = make_volume((len(dataset["channel_names"]), *shape), seed=3, border=True)
    assert_equal_to_nnunet(data, spacing, plans, checkpoint["init_args"]["configuration"], dataset)
