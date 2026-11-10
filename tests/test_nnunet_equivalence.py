"""Converted networks against nnU-Net's own network construction.

Needs nnunetv2 and real model folders, so it is skipped in the default
environment. Point NNUNET_LITE_TEST_MODELS at a directory laid out as
``<dataset>/<trainer__plans__configuration>/`` (defaults to the leadup models in
``packages/nnunet_model_conversion/models``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

import nnunet_lite as nl

pytest.importorskip("nnunetv2")

MODELS_ROOT = Path(
    os.environ.get(
        "NNUNET_LITE_TEST_MODELS",
        Path(__file__).parents[1] / "packages/nnunet_model_conversion/models",
    )
)
MODEL_DIRS = sorted(p.parent for p in MODELS_ROOT.glob("*/*/plans.json"))


def reference_network(folder: Path) -> torch.nn.Module:
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    dataset = json.loads((folder / "dataset.json").read_text())
    plans = PlansManager(str(folder / "plans.json"))
    checkpoint = torch.load(
        folder / "fold_0/checkpoint_final.pth", map_location="cpu", weights_only=False
    )
    configuration = plans.get_configuration(checkpoint["init_args"]["configuration"])
    network: torch.nn.Module = get_network_from_plans(
        configuration.network_arch_class_name,
        configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import,
        determine_num_input_channels(plans, configuration, dataset),
        plans.get_label_manager(dataset).num_segmentation_heads,
        allow_init=False,
        deep_supervision=False,
    )
    network.load_state_dict(checkpoint["network_weights"], strict=True)
    return network.eval()


@pytest.fixture(scope="module", params=MODEL_DIRS, ids=[p.parent.name for p in MODEL_DIRS])
def networks(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    folder: Path = request.param
    output = tmp_path_factory.mktemp("bundle")
    nl.convert(folder, output_dir=output)
    return reference_network(folder), nl.load_bundle(output)


@torch.inference_mode()
def test_cpu_fp32_bitwise_equal_to_nnunet(networks: Any) -> None:
    reference, bundle = networks
    channels = bundle.config.network.input_channels
    patch = torch.randn((1, channels, 64, 64, 64), generator=torch.Generator().manual_seed(7))
    assert torch.equal(reference(patch), bundle.network(patch))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_cuda_autocast_bitwise_equal_to_nnunet(networks: Any) -> None:
    reference, bundle = networks
    channels = bundle.config.network.input_channels
    patch = torch.randn(
        (1, channels, *bundle.config.patch_size), generator=torch.Generator().manual_seed(7)
    ).cuda()
    try:
        reference.cuda()
        bundle.network.cuda()
        with torch.autocast("cuda"):
            assert torch.equal(reference(patch), bundle.network(patch))
    finally:
        reference.cpu()
        bundle.network.cpu()
        torch.cuda.empty_cache()


def test_config_matches_plans(networks: Any) -> None:
    _, bundle = networks
    assert bundle.config.labels[0] == "background"
    assert bundle.config.network.output_channels == len(bundle.config.labels)
    assert len(bundle.config.patch_size) == bundle.config.network.ndim
