from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bundle import Bundle, state_dict_mismatches
from .config import (
    BundleConfig,
    ChannelConfig,
    InferenceConfig,
    NetworkConfig,
    PreprocessingConfig,
    ResamplingConfig,
)
from .network import build_network

_KNOWN_ARCH_KWARGS = {
    "n_stages",
    "features_per_stage",
    "conv_op",
    "kernel_sizes",
    "strides",
    "n_conv_per_stage",
    "n_blocks_per_stage",
    "n_conv_per_stage_decoder",
    "conv_bias",
    "norm_op",
    "norm_op_kwargs",
    "dropout_op",
    "dropout_op_kwargs",
    "nonlin",
    "nonlin_kwargs",
    "nonlin_first",
}


def _class_name(qualified: Any) -> str:
    return str(qualified).rsplit(".", 1)[-1]


def _per_stage(value: Any, stages: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * stages
    return tuple(int(v) for v in value)


def _per_axis(values: Any, ndim: int) -> tuple[tuple[int, ...], ...]:
    return tuple((int(v),) * ndim if isinstance(v, int) else tuple(int(a) for a in v) for v in values)


def _resolve_configuration(plans: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in plans["configurations"]:
        raise KeyError(f"Configuration {name!r} not in plans: {sorted(plans['configurations'])}")
    configuration = dict(plans["configurations"][name])
    parent_name = configuration.pop("inherits_from", None)
    if parent_name is None:
        return configuration
    parent = _resolve_configuration(plans, parent_name)
    parent.update(configuration)
    return parent


def _network_config(
    configuration: dict[str, Any], input_channels: int, output_channels: int
) -> NetworkConfig:
    spec = configuration.get("architecture")
    if spec is None:
        # Legacy plans: nnU-Net hard-codes InstanceNorm(eps=1e-5, affine) and
        # LeakyReLU(0.01), conv bias on, norm before nonlin.
        kernel_sizes = configuration["conv_kernel_sizes"]
        return NetworkConfig(
            architecture=str(configuration["UNet_class_name"]),
            input_channels=input_channels,
            output_channels=output_channels,
            features_per_stage=tuple(
                min(
                    int(configuration["UNet_base_num_features"]) * 2**stage,
                    int(configuration["unet_max_num_features"]),
                )
                for stage in range(len(kernel_sizes))
            ),
            kernel_sizes=tuple(tuple(int(k) for k in kernel) for kernel in kernel_sizes),
            strides=tuple(tuple(int(s) for s in st) for st in configuration["pool_op_kernel_sizes"]),
            encoder_blocks=tuple(int(n) for n in configuration["n_conv_per_stage_encoder"]),
            decoder_convolutions=tuple(int(n) for n in configuration["n_conv_per_stage_decoder"]),
        )

    architecture = _class_name(spec["network_class_name"])
    kwargs = spec["arch_kwargs"]
    unknown = set(kwargs) - _KNOWN_ARCH_KWARGS
    if unknown:
        raise NotImplementedError(f"Unsupported arch_kwargs: {sorted(unknown)}")
    conv = _class_name(kwargs["conv_op"])
    if conv not in ("Conv2d", "Conv3d"):
        raise NotImplementedError(f"Unsupported conv_op: {kwargs['conv_op']}")
    ndim = int(conv[4])
    if _class_name(kwargs.get("norm_op")) != f"InstanceNorm{ndim}d":
        raise NotImplementedError(f"Unsupported norm_op: {kwargs.get('norm_op')}")
    if kwargs.get("dropout_op") is not None:
        raise NotImplementedError(f"Dropout is not supported: {kwargs['dropout_op']}")
    norm_kwargs = kwargs.get("norm_op_kwargs") or {}
    nonlin_kwargs = kwargs.get("nonlin_kwargs") or {}
    stages = len(kwargs["features_per_stage"])
    encoder_key = "n_conv_per_stage" if architecture == "PlainConvUNet" else "n_blocks_per_stage"
    return NetworkConfig(
        architecture=architecture,
        input_channels=input_channels,
        output_channels=output_channels,
        features_per_stage=tuple(int(f) for f in kwargs["features_per_stage"]),
        kernel_sizes=_per_axis(kwargs["kernel_sizes"], ndim),
        strides=_per_axis(kwargs["strides"], ndim),
        encoder_blocks=_per_stage(kwargs[encoder_key], stages),
        decoder_convolutions=_per_stage(kwargs["n_conv_per_stage_decoder"], stages - 1),
        conv_bias=bool(kwargs.get("conv_bias", False)),
        norm_eps=float(norm_kwargs.get("eps", 1e-5)),
        norm_affine=bool(norm_kwargs.get("affine", False)),
        nonlin=_class_name(kwargs.get("nonlin")),
        nonlin_negative_slope=float(nonlin_kwargs.get("negative_slope", 0.01)),
        nonlin_first=bool(kwargs.get("nonlin_first", False)),
    )


def _labels(dataset: dict[str, Any]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for name, value in dataset["labels"].items():
        if name == "ignore" or not isinstance(value, int):
            raise NotImplementedError(
                f"Region-based and ignore labels are not supported: {name!r}: {value!r}"
            )
        labels[value] = str(name)
    if sorted(labels) != list(range(len(labels))):
        raise ValueError(f"Labels must be consecutive integers starting at 0: {sorted(labels)}")
    return dict(sorted(labels.items()))


def _preprocessing_config(
    plans: dict[str, Any], configuration: dict[str, Any], dataset: dict[str, Any]
) -> PreprocessingConfig:
    channel_names = dataset.get("channel_names") or dataset["modality"]
    intensities = plans["foreground_intensity_properties_per_channel"]
    channels = []
    for index in range(len(channel_names)):
        intensity = intensities[str(index)]
        channels.append(
            ChannelConfig(
                name=str(channel_names[str(index)]),
                normalization=str(configuration["normalization_schemes"][index]),
                use_mask_for_norm=bool(configuration["use_mask_for_norm"][index]),
                lower=float(intensity["percentile_00_5"]),
                upper=float(intensity["percentile_99_5"]),
                mean=float(intensity["mean"]),
                std=float(intensity["std"]),
            )
        )
    return PreprocessingConfig(
        image_reader=str(plans["image_reader_writer"]),
        target_spacing=tuple(float(s) for s in configuration["spacing"]),
        transpose_forward=tuple(int(a) for a in plans["transpose_forward"]),
        transpose_backward=tuple(int(a) for a in plans["transpose_backward"]),
        channels=tuple(channels),
        resampling_data=ResamplingConfig(
            function=str(configuration["resampling_fn_data"]),
            kwargs=dict(configuration["resampling_fn_data_kwargs"]),
        ),
        resampling_probabilities=ResamplingConfig(
            function=str(configuration["resampling_fn_probabilities"]),
            kwargs=dict(configuration["resampling_fn_probabilities_kwargs"]),
        ),
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    # nnU-Net checkpoints pickle a few numpy scalars next to the tensors. Allow
    # exactly those so the file is never unpickled with arbitrary code execution.
    scalar = np._core.multiarray.scalar  # type: ignore[attr-defined]
    dtypes = [type(np.dtype(t)) for t in (np.float16, np.float32, np.float64, np.int32, np.int64)]
    allowed: list[Any] = [
        scalar,
        (scalar, "numpy.core.multiarray.scalar"),
        (scalar, "numpy._core.multiarray.scalar"),
        np.dtype,
        *dtypes,
    ]
    try:
        with torch.serialization.safe_globals(allowed):
            checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise RuntimeError(
            f"{path} is not a plain nnU-Net checkpoint and was not loaded, because doing so "
            f"would execute code from the file: {error}"
        ) from error
    return checkpoint


def _alias_target(name: str) -> str | None:
    # DNA exposes the encoder again as ``decoder.encoder.*`` and registers each
    # conv block's layers a second time in ``all_modules`` as [conv, norm, nonlin].
    target = name.removeprefix("decoder.") if name.startswith("decoder.encoder.") else name
    target = target.replace(".all_modules.0.", ".conv.").replace(".all_modules.1.", ".norm.")
    if ".all_modules." in target:
        raise ValueError(f"Unrecognised checkpoint weight layout: {name}")
    return None if target == name else target


def _canonical_weights(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Every dropped alias is verified bitwise against the tensor it duplicates,
    # so nothing in the checkpoint is silently discarded.
    primary = {name: tensor for name, tensor in weights.items() if _alias_target(name) is None}
    for name, tensor in weights.items():
        target = _alias_target(name)
        if target is None:
            continue
        if target not in primary:
            raise ValueError(f"Alias {name} has no primary tensor {target}")
        if state_dict_mismatches({target: primary[target]}, {target: tensor}):
            raise ValueError(f"Alias {name} differs from its primary tensor {target}")
    return primary


def convert(
    model_dir: str | Path,
    *,
    fold: int | str = 0,
    checkpoint: str = "checkpoint_final.pth",
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    device: str | torch.device = "cpu",
    inference: InferenceConfig | None = None,
) -> Bundle:
    source = Path(model_dir)
    checkpoint_path = source / f"fold_{fold}" / checkpoint
    for required in (source / "plans.json", source / "dataset.json", checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    plans: dict[str, Any] = json.loads((source / "plans.json").read_text())
    dataset: dict[str, Any] = json.loads((source / "dataset.json").read_text())
    state = _load_checkpoint(checkpoint_path)

    configuration_name = str(state["init_args"]["configuration"])
    configuration = _resolve_configuration(plans, configuration_name)
    if configuration.get("previous_stage") is not None:
        raise NotImplementedError("Cascade configurations are not supported")
    labels = _labels(dataset)
    preprocessing = _preprocessing_config(plans, configuration, dataset)
    config = BundleConfig(
        dataset=str(plans["dataset_name"]),
        trainer=str(state["trainer_name"]),
        plans=str(plans["plans_name"]),
        configuration=configuration_name,
        fold=str(fold),
        network=_network_config(configuration, len(preprocessing.channels), len(labels)),
        preprocessing=preprocessing,
        patch_size=tuple(int(p) for p in configuration["patch_size"]),
        mirror_axes=tuple(int(a) for a in state.get("inference_allowed_mirroring_axes") or ()),
        labels=labels,
        inference=inference if inference is not None else InferenceConfig(),
    )

    network = build_network(config.network, initialize=False)
    network.load_state_dict(_canonical_weights(state["network_weights"]), strict=True, assign=True)
    network.eval().to(torch.device(device))
    bundle = Bundle(config=config, network=network)
    if output_dir is not None:
        bundle.save(output_dir, overwrite=overwrite)
    return bundle
