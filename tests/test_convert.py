"""Conversion and bundle round trips on synthetic nnU-Net model folders (no nnunetv2 needed)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import nnunet_lite as nl
from nnunet_lite.bundle import state_dict_mismatches
from nnunet_lite.network import build_network

RESAMPLING = {
    "resampling_fn_data": "resample_data_or_seg_to_shape",
    "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
    "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
    "resampling_fn_probabilities_kwargs": {
        "is_seg": False,
        "order": 1,
        "order_z": 0,
        "force_separate_z": None,
    },
}

LEGACY_3D = {
    "patch_size": [16, 16, 16],
    "spacing": [1.5, 1.5, 1.5],
    "normalization_schemes": ["CTNormalization"],
    "use_mask_for_norm": [False],
    "UNet_class_name": "PlainConvUNet",
    "UNet_base_num_features": 4,
    "unet_max_num_features": 8,
    "n_conv_per_stage_encoder": [2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2],
    "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
    "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
    **RESAMPLING,
}

RESENC_2D = {
    "patch_size": [16, 16],
    "spacing": [1.0, 1.0],
    "normalization_schemes": ["CTNormalization"],
    "use_mask_for_norm": [False],
    "architecture": {
        "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        "arch_kwargs": {
            "n_stages": 3,
            "features_per_stage": [4, 8, 8],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2]],
            "n_blocks_per_stage": [1, 2, 2],
            "n_conv_per_stage_decoder": [1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    },
    **RESAMPLING,
}

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "3d_fullres": LEGACY_3D,
    "2d": RESENC_2D,
    "3d_fullres_bs8": {"inherits_from": "3d_fullres", "batch_size": 8},
}


def with_dna_aliases(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add the duplicate names dynamic_network_architectures stores in checkpoints."""
    aliased = dict(state)
    for name, tensor in state.items():
        alias = name.replace(".conv.", ".all_modules.0.").replace(".norm.", ".all_modules.1.")
        aliased[alias] = tensor.clone()
    for name, tensor in list(aliased.items()):
        if name.startswith("encoder."):
            aliased[f"decoder.{name}"] = tensor.clone()
    return aliased


def make_model_dir(
    root: Path, configuration: str, *, labels: dict[str, Any] | None = None
) -> tuple[Path, dict[str, torch.Tensor]]:
    """Write plans.json, dataset.json and fold_0/checkpoint_final.pth with random weights."""
    from nnunet_lite.convert import _network_config, _resolve_configuration

    plans = {
        "dataset_name": "Dataset001_Synthetic",
        "plans_name": "nnUNetPlans",
        "image_reader_writer": "NibabelIOWithReorient",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "foreground_intensity_properties_per_channel": {
            "0": {"percentile_00_5": -100.0, "percentile_99_5": 200.0, "mean": 10.0, "std": 50.0}
        },
        "configurations": CONFIGURATIONS,
    }
    dataset = {
        "channel_names": {"0": "CT"},
        "labels": labels or {"background": 0, "organ": 1, "lesion": 2},
    }
    torch.manual_seed(0)
    network = build_network(
        _network_config(_resolve_configuration(plans, configuration), 1, len(dataset["labels"]))
    )
    weights = {name: tensor.clone() for name, tensor in network.state_dict().items()}
    model_dir = root / "nnUNetTrainer__nnUNetPlans__synthetic"
    (model_dir / "fold_0").mkdir(parents=True)
    (model_dir / "plans.json").write_text(json.dumps(plans))
    (model_dir / "dataset.json").write_text(json.dumps(dataset))
    torch.save(
        {
            "network_weights": with_dna_aliases(weights),
            "optimizer_state": {"lr": 0.01},
            "_best_ema": np.float64(0.9),
            "init_args": {"configuration": configuration, "fold": 0},
            "trainer_name": "nnUNetTrainer",
            "inference_allowed_mirroring_axes": (0, 1, 2),
        },
        model_dir / "fold_0" / "checkpoint_final.pth",
    )
    return model_dir, weights


@pytest.fixture(params=["3d_fullres", "2d", "3d_fullres_bs8"])
def model(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    return make_model_dir(tmp_path, request.param)


def test_convert_on_the_fly(model: tuple[Path, dict[str, torch.Tensor]]) -> None:
    model_dir, weights = model
    bundle = nl.convert(model_dir)
    assert not state_dict_mismatches(weights, bundle.network.state_dict())
    assert not bundle.network.training
    assert bundle.config.labels == {0: "background", 1: "organ", 2: "lesion"}
    assert bundle.config.mirror_axes == (0, 1, 2)
    assert bundle.config.trainer == "nnUNetTrainer"
    assert bundle.config.preprocessing.image_reader == "NibabelIOWithReorient"
    channel = bundle.config.preprocessing.channels[0]
    assert (channel.lower, channel.upper, channel.mean, channel.std) == (-100.0, 200.0, 10.0, 50.0)
    patch = torch.randn(1, 1, *bundle.config.patch_size)
    with torch.inference_mode():
        assert bundle.network(patch).shape == (1, 3, *bundle.config.patch_size)


def test_save_and_load_round_trip(model: tuple[Path, dict[str, torch.Tensor]], tmp_path: Path) -> None:
    model_dir, weights = model
    output = tmp_path / "bundles" / "synthetic"
    converted = nl.convert(model_dir, output_dir=output)
    assert sorted(p.name for p in output.iterdir()) == ["bundle.json", "weights.safetensors"]

    loaded = nl.load_bundle(output)
    assert loaded.config == converted.config
    assert not state_dict_mismatches(weights, loaded.network.state_dict())
    patch = torch.randn(1, 1, *loaded.config.patch_size)
    with torch.inference_mode():
        assert torch.equal(converted.network(patch), loaded.network(patch))


def test_save_refuses_to_overwrite(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(tmp_path, "3d_fullres")
    bundle = nl.convert(model_dir, output_dir=tmp_path / "bundle")
    with pytest.raises(FileExistsError):
        bundle.save(tmp_path / "bundle")
    bundle.save(tmp_path / "bundle", overwrite=True)
    nl.load_bundle(tmp_path / "bundle")


def test_load_detects_modified_weights(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(tmp_path, "3d_fullres")
    nl.convert(model_dir, output_dir=tmp_path / "bundle")
    weights = tmp_path / "bundle" / "weights.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256"):
        nl.load_bundle(tmp_path / "bundle")


def test_rejects_region_labels(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(
        tmp_path, "3d_fullres", labels={"background": 0, "whole": [1, 2], "core": 2}
    )
    with pytest.raises(NotImplementedError, match="Region-based"):
        nl.convert(model_dir)


def test_rejects_alias_that_differs_from_primary(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(tmp_path, "3d_fullres")
    path = model_dir / "fold_0" / "checkpoint_final.pth"
    checkpoint = torch.load(path, weights_only=False)
    alias = next(name for name in checkpoint["network_weights"] if ".all_modules.0." in name)
    checkpoint["network_weights"][alias] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="differs from its primary"):
        nl.convert(model_dir)


class _Malicious:
    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return os.system, ("true",)


def test_checkpoint_with_arbitrary_pickle_is_not_executed(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(tmp_path, "3d_fullres")
    path = model_dir / "fold_0" / "checkpoint_final.pth"
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["payload"] = _Malicious()
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="not a plain nnU-Net checkpoint"):
        nl.convert(model_dir)


def test_missing_checkpoint(tmp_path: Path) -> None:
    model_dir, _ = make_model_dir(tmp_path, "3d_fullres")
    with pytest.raises(FileNotFoundError):
        nl.convert(model_dir, fold=3)
