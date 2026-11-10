from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BUNDLE_FILE = "bundle.json"
WEIGHTS_FILE = "weights.safetensors"
SCHEMA_VERSION = 1

SUPPORTED_ARCHITECTURES = ("PlainConvUNet", "ResidualEncoderUNet")
SUPPORTED_NONLINS = ("LeakyReLU", "ReLU")


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class NetworkConfig:
    architecture: str
    input_channels: int
    output_channels: int
    features_per_stage: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]
    # n_conv_per_stage (PlainConvUNet) or n_blocks_per_stage (ResidualEncoderUNet)
    encoder_blocks: tuple[int, ...]
    decoder_convolutions: tuple[int, ...]
    conv_bias: bool = True
    norm_eps: float = 1e-5
    norm_affine: bool = True
    nonlin: str = "LeakyReLU"
    nonlin_negative_slope: float = 0.01
    nonlin_first: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise NotImplementedError(f"Unsupported architecture: {self.architecture}")
        if self.nonlin not in SUPPORTED_NONLINS:
            raise NotImplementedError(f"Unsupported nonlinearity: {self.nonlin}")
        if self.ndim not in (2, 3):
            raise NotImplementedError(f"Only 2D and 3D networks are supported, got {self.ndim}D")
        stages = len(self.features_per_stage)
        lengths = {
            "kernel_sizes": len(self.kernel_sizes),
            "strides": len(self.strides),
            "encoder_blocks": len(self.encoder_blocks),
            "decoder_convolutions": len(self.decoder_convolutions) + 1,
        }
        bad = {name: n for name, n in lengths.items() if n != stages}
        if bad:
            raise ValueError(f"Network config does not describe {stages} stages: {bad}")
        if any(len(v) != self.ndim for v in (*self.kernel_sizes, *self.strides)):
            raise ValueError("kernel_sizes and strides must all have the same dimensionality")

    @property
    def ndim(self) -> int:
        return len(self.kernel_sizes[0])

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NetworkConfig:
        return cls(
            architecture=str(raw["architecture"]),
            input_channels=int(raw["input_channels"]),
            output_channels=int(raw["output_channels"]),
            features_per_stage=_int_tuple(raw["features_per_stage"]),
            kernel_sizes=tuple(_int_tuple(v) for v in raw["kernel_sizes"]),
            strides=tuple(_int_tuple(v) for v in raw["strides"]),
            encoder_blocks=_int_tuple(raw["encoder_blocks"]),
            decoder_convolutions=_int_tuple(raw["decoder_convolutions"]),
            conv_bias=bool(raw["conv_bias"]),
            norm_eps=float(raw["norm_eps"]),
            norm_affine=bool(raw["norm_affine"]),
            nonlin=str(raw["nonlin"]),
            nonlin_negative_slope=float(raw["nonlin_negative_slope"]),
            nonlin_first=bool(raw["nonlin_first"]),
        )


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    normalization: str
    use_mask_for_norm: bool
    # Foreground intensity statistics of the training set (used by CTNormalization).
    lower: float
    upper: float
    mean: float
    std: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChannelConfig:
        return cls(
            name=str(raw["name"]),
            normalization=str(raw["normalization"]),
            use_mask_for_norm=bool(raw["use_mask_for_norm"]),
            lower=float(raw["lower"]),
            upper=float(raw["upper"]),
            mean=float(raw["mean"]),
            std=float(raw["std"]),
        )


@dataclass(frozen=True)
class ResamplingConfig:
    # nnU-Net resampling function name and its kwargs, verbatim from plans.json.
    function: str
    kwargs: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResamplingConfig:
        return cls(function=str(raw["function"]), kwargs=dict(raw["kwargs"]))


@dataclass(frozen=True)
class PreprocessingConfig:
    target_spacing: tuple[float, ...]
    transpose_forward: tuple[int, ...]
    transpose_backward: tuple[int, ...]
    channels: tuple[ChannelConfig, ...]
    resampling_data: ResamplingConfig
    resampling_probabilities: ResamplingConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PreprocessingConfig:
        return cls(
            target_spacing=tuple(float(v) for v in raw["target_spacing"]),
            transpose_forward=_int_tuple(raw["transpose_forward"]),
            transpose_backward=_int_tuple(raw["transpose_backward"]),
            channels=tuple(ChannelConfig.from_dict(c) for c in raw["channels"]),
            resampling_data=ResamplingConfig.from_dict(raw["resampling_data"]),
            resampling_probabilities=ResamplingConfig.from_dict(raw["resampling_probabilities"]),
        )


@dataclass(frozen=True)
class BundleConfig:
    dataset: str
    trainer: str
    plans: str
    configuration: str
    fold: str
    network: NetworkConfig
    preprocessing: PreprocessingConfig
    patch_size: tuple[int, ...]
    # Axes nnU-Net allows for test-time mirroring; empty if the trainer disabled it.
    mirror_axes: tuple[int, ...]
    labels: dict[int, str]

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["labels"] = {str(index): name for index, name in sorted(self.labels.items())}
        return {"schema_version": SCHEMA_VERSION, **raw}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BundleConfig:
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported bundle schema_version {version!r}; expected {SCHEMA_VERSION}")
        return cls(
            dataset=str(raw["dataset"]),
            trainer=str(raw["trainer"]),
            plans=str(raw["plans"]),
            configuration=str(raw["configuration"]),
            fold=str(raw["fold"]),
            network=NetworkConfig.from_dict(raw["network"]),
            preprocessing=PreprocessingConfig.from_dict(raw["preprocessing"]),
            patch_size=_int_tuple(raw["patch_size"]),
            mirror_axes=_int_tuple(raw["mirror_axes"]),
            labels={int(index): str(name) for index, name in raw["labels"].items()},
        )
