from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .config import NetworkConfig

# Plain-PyTorch PlainConvUNet / ResidualEncoderUNet, wired exactly like
# dynamic_network_architectures (DNA) builds them for nnU-Net inference.
# Parameter names match DNA's primary names; DNA's duplicate registrations
# (``*.all_modules.*`` and ``decoder.encoder.*``) are not created.

_CONV: dict[int, type[nn.Module]] = {2: nn.Conv2d, 3: nn.Conv3d}
_CONV_TRANSPOSE: dict[int, type[nn.Module]] = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
_INSTANCE_NORM: dict[int, type[nn.Module]] = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
_AVG_POOL: dict[int, type[nn.Module]] = {2: nn.AvgPool2d, 3: nn.AvgPool3d}


def _nonlin(config: NetworkConfig) -> nn.Module:
    if config.nonlin == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=config.nonlin_negative_slope, inplace=True)
    return nn.ReLU(inplace=True)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        config: NetworkConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int] | int,
        *,
        bias: bool | None = None,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = _CONV[config.ndim](
            in_channels,
            out_channels,
            tuple(kernel_size),
            stride if isinstance(stride, int) else tuple(stride),
            padding=tuple((k - 1) // 2 for k in kernel_size),
            bias=config.conv_bias if bias is None else bias,
        )
        self.norm = _INSTANCE_NORM[config.ndim](
            out_channels, eps=config.norm_eps, affine=config.norm_affine
        )
        self.nonlin = _nonlin(config) if activation else None
        self.nonlin_first = config.nonlin_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.nonlin is None:
            return self.norm(x)  # type: ignore[no-any-return]
        if self.nonlin_first:
            return self.norm(self.nonlin(x))  # type: ignore[no-any-return]
        return self.nonlin(self.norm(x))  # type: ignore[no-any-return]


class ConvStack(nn.Module):
    def __init__(
        self,
        config: NetworkConfig,
        count: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int] | int,
    ) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            ConvNormAct(config, in_channels, out_channels, kernel_size, stride),
            *[ConvNormAct(config, out_channels, out_channels, kernel_size, 1) for _ in range(count - 1)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)  # type: ignore[no-any-return]


class BasicBlockD(nn.Module):
    def __init__(
        self,
        config: NetworkConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int],
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(config, in_channels, out_channels, kernel_size, stride)
        self.conv2 = ConvNormAct(config, out_channels, out_channels, kernel_size, 1, activation=False)
        self.nonlin2 = _nonlin(config)
        skip: list[nn.Module] = []
        if any(step != 1 for step in stride):
            skip.append(_AVG_POOL[config.ndim](tuple(stride), tuple(stride)))
        if in_channels != out_channels:
            skip.append(
                ConvNormAct(
                    config,
                    in_channels,
                    out_channels,
                    (1,) * config.ndim,
                    1,
                    bias=False,
                    activation=False,
                )
            )
        self.skip = nn.Sequential(*skip) if skip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv2(self.conv1(x))
        out += self.skip(x)
        return self.nonlin2(out)  # type: ignore[no-any-return]


class ResidualStage(nn.Module):
    def __init__(
        self,
        config: NetworkConfig,
        count: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int],
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            BasicBlockD(config, in_channels, out_channels, kernel_size, stride),
            *[
                BasicBlockD(config, out_channels, out_channels, kernel_size, (1,) * config.ndim)
                for _ in range(count - 1)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)  # type: ignore[no-any-return]


class PlainEncoder(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_channels = config.input_channels
        for count, out_channels, kernel, stride in zip(
            config.encoder_blocks,
            config.features_per_stage,
            config.kernel_sizes,
            config.strides,
            strict=True,
        ):
            # The extra Sequential reproduces DNA's ``encoder.stages.<s>.0.convs`` names.
            stages.append(
                nn.Sequential(ConvStack(config, count, in_channels, out_channels, kernel, stride))
            )
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class ResidualEncoder(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        stem_channels = config.features_per_stage[0]
        self.stem = ConvStack(config, 1, config.input_channels, stem_channels, config.kernel_sizes[0], 1)
        stages: list[nn.Module] = []
        in_channels = stem_channels
        for count, out_channels, kernel, stride in zip(
            config.encoder_blocks,
            config.features_per_stage,
            config.kernel_sizes,
            config.strides,
            strict=True,
        ):
            stages.append(ResidualStage(config, count, in_channels, out_channels, kernel, stride))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        features = config.features_per_stage
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for step in range(1, len(features)):
            below, skip = features[-step], features[-(step + 1)]
            stride = tuple(config.strides[-step])
            self.transpconvs.append(
                _CONV_TRANSPOSE[config.ndim](below, skip, stride, stride, bias=config.conv_bias)
            )
            self.stages.append(
                ConvStack(
                    config,
                    config.decoder_convolutions[step - 1],
                    2 * skip,
                    skip,
                    config.kernel_sizes[-(step + 1)],
                    1,
                )
            )
            # nnU-Net trains one head per stage (deep supervision); only the last
            # is used at inference, but all are kept so checkpoints load strictly.
            self.seg_layers.append(_CONV[config.ndim](skip, config.output_channels, 1, 1, 0, bias=True))

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        x = skips[-1]
        for step, (upsample, stage) in enumerate(zip(self.transpconvs, self.stages, strict=True)):
            x = stage(torch.cat((upsample(x), skips[-(step + 2)]), 1))
        return self.seg_layers[-1](x)  # type: ignore[no-any-return]


class UNet(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.encoder: nn.Module = (
            PlainEncoder(config) if config.architecture == "PlainConvUNet" else ResidualEncoder(config)
        )
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))  # type: ignore[no-any-return]


def build_network(config: NetworkConfig) -> UNet:
    return UNet(config)
