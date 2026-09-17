from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_CHANNELS = (32, 48, 48, 64, 64, 96, 96, 128, 160)
DEFAULT_STRIDES = (
    (1, 1),
    (1, 1),
    (2, 2),
    (1, 1),
    (2, 2),
    (1, 1),
    (2, 2),
    (1, 1),
)


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.depthwise = ConvNormActivation(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvNormActivation(in_channels, out_channels, kernel_size=1)
        self.use_residual = stride == (1, 1) and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pointwise(self.depthwise(inputs))
        return outputs + inputs if self.use_residual else outputs


class AudioMultiTaskCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_azimuths: int,
        in_channels: int = 5,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in channels)
        if len(channels) != len(DEFAULT_STRIDES) + 1:
            raise ValueError("channels length must be len(DEFAULT_STRIDES) + 1")

        self.stem = ConvNormActivation(in_channels, channels[0], kernel_size=3, stride=(2, 2))
        self.blocks = nn.Sequential(
            *[
                DepthwiseSeparableBlock(in_ch, out_ch, stride)
                for in_ch, out_ch, stride in zip(channels[:-1], channels[1:], DEFAULT_STRIDES)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(channels[-1], num_classes)
        self.azimuth_head = nn.Linear(channels[-1], num_azimuths)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.pool(self.blocks(self.stem(inputs))).flatten(1)
        embedding = self.dropout(embedding)
        return self.class_head(embedding), self.azimuth_head(embedding)


class AudioMultiTaskCNNAngle(AudioMultiTaskCNN):
    """Baseline CNN with an auxiliary circular angle head."""

    def __init__(
        self,
        num_classes: int,
        num_azimuths: int,
        in_channels: int = 5,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(num_classes, num_azimuths, in_channels, channels, dropout)
        self.angle_head = nn.Linear(tuple(int(channel) for channel in channels)[-1], 2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.pool(self.blocks(self.stem(inputs))).flatten(1)
        embedding = self.dropout(embedding)
        angle = nn.functional.normalize(self.angle_head(embedding), dim=1)
        return self.class_head(embedding), self.azimuth_head(embedding), angle


class AudioMultiTaskCNNSpatial(AudioMultiTaskCNN):
    """Baseline trunk with task-specific pooling for sound and localization.

    Global average pooling is deliberately avoided here.  Transient sound classes
    benefit from max statistics, while localization needs the frequency-dependent
    HRTF pattern that global frequency pooling destroys.
    """

    def __init__(
        self,
        num_classes: int,
        num_azimuths: int,
        in_channels: int = 5,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(num_classes, num_azimuths, in_channels, channels, dropout)
        final_channels = tuple(int(channel) for channel in channels)[-1]
        self.class_head = nn.Sequential(
            nn.LayerNorm(final_channels * 2),
            nn.Dropout(dropout),
            nn.Linear(final_channels * 2, num_classes),
        )
        # With 64 mel bins and four 2x frequency reductions the trunk emits 4 bins.
        spatial_dim = final_channels * 4 * 2
        self.azimuth_head = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_azimuths),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.blocks(self.stem(inputs))
        class_embedding = torch.cat(
            (features.mean(dim=(2, 3)), features.amax(dim=(2, 3))), dim=1
        )
        # Pool time, not frequency. Mean and standard deviation make the head
        # robust to event timing while retaining the four HRTF frequency regions.
        spatial_mean = features.mean(dim=3)
        spatial_std = features.var(dim=3, unbiased=False).add(1e-5).sqrt()
        spatial_embedding = torch.cat((spatial_mean, spatial_std), dim=1).flatten(1)
        return self.class_head(class_embedding), self.azimuth_head(spatial_embedding)


class AudioMultiTaskCNNDualPath(AudioMultiTaskCNN):
    """CNN plus an absolute-frequency localization path over raw ILD/IPD.

    The convolutional trunk remains useful for sound recognition and local
    spectro-temporal patterns.  The second path does not convolve or pool over
    frequency, so it can learn the absolute-frequency phase slope and HRTF cues
    that encode interaural delay.
    """

    def __init__(
        self,
        num_classes: int,
        num_azimuths: int,
        in_channels: int = 5,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(num_classes, num_azimuths, in_channels, channels, dropout)
        # Three spatial channels x 64 absolute mel bins x four robust moments.
        spatial_dim = 3 * 64 * 4
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_azimuths),
        )
        # A learnable scale starts small so warm-starting exactly preserves the
        # proven baseline while the new physical-cue path becomes calibrated.
        self.spatial_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.pool(self.blocks(self.stem(inputs))).flatten(1)
        embedding = self.dropout(embedding)
        class_logits = self.class_head(embedding)
        azimuth_logits = self.azimuth_head(embedding)

        spatial = inputs[:, 2:5]
        moments = torch.cat(
            (
                spatial.mean(dim=3),
                spatial.std(dim=3, unbiased=False),
                spatial.amax(dim=3),
                spatial.amin(dim=3),
            ),
            dim=1,
        ).flatten(1)
        azimuth_logits = azimuth_logits + self.spatial_scale * self.spatial_head(moments)
        return class_logits, azimuth_logits


class AudioMultiTaskCNNGCC(AudioMultiTaskCNN):
    """Baseline CNN fused with an explicit GCC-PHAT delay-vector head."""

    def __init__(self, num_classes: int, num_azimuths: int) -> None:
        super().__init__(num_classes, num_azimuths, in_channels=5)
        self.gcc_head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_azimuths),
        )
        self.gcc_scale = nn.Parameter(torch.tensor(0.1))
        self.freeze_base = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_base:
            # Freezing parameters alone does not freeze BatchNorm buffers or
            # dropout behavior. Keep the warm-started baseline bit-for-bit fixed.
            self.stem.eval()
            self.blocks.eval()
            self.dropout.eval()
            self.class_head.eval()
            self.azimuth_head.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectral = inputs[:, :5]
        embedding = self.pool(self.blocks(self.stem(spectral))).flatten(1)
        embedding = self.dropout(embedding)
        class_logits = self.class_head(embedding)
        azimuth_logits = self.azimuth_head(embedding)
        gcc_vector = inputs[:, 5, :, 0]
        return class_logits, azimuth_logits + self.gcc_scale * self.gcc_head(gcc_vector)


class AudioMultiTaskCNNGCCTemporal(AudioMultiTaskCNNGCC):
    """GCC localizer plus a time-aware residual sound-class head."""

    def __init__(self, num_classes: int, num_azimuths: int) -> None:
        super().__init__(num_classes, num_azimuths)
        channels = DEFAULT_CHANNELS[-1]
        self.temporal_attention = nn.Conv1d(channels, 1, kernel_size=1)
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )
        self.temporal_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectral = inputs[:, :5]
        features = self.blocks(self.stem(spectral))
        embedding = self.pool(features).flatten(1)
        embedding = self.dropout(embedding)
        class_logits = self.class_head(embedding)
        azimuth_logits = self.azimuth_head(embedding)

        # Average frequency only. The remaining 13-step sequence preserves when
        # a transient occurs; learned attention complements mean/max statistics.
        temporal = features.mean(dim=2)
        attention = torch.softmax(self.temporal_attention(temporal), dim=2)
        attended = (temporal * attention).sum(dim=2)
        temporal_embedding = torch.cat(
            (temporal.mean(dim=2), temporal.amax(dim=2), attended), dim=1
        )
        class_logits = class_logits + self.temporal_scale * self.temporal_head(temporal_embedding)

        gcc_vector = inputs[:, 5, :, 0]
        azimuth_logits = azimuth_logits + self.gcc_scale * self.gcc_head(gcc_vector)
        return class_logits, azimuth_logits


class AudioMultiTaskCNNGCCClassBranch(AudioMultiTaskCNNGCC):
    """GCC model with an independent mono-spectrogram sound classifier."""

    def __init__(self, num_classes: int, num_azimuths: int) -> None:
        super().__init__(num_classes, num_azimuths)
        self.class_branch = nn.Sequential(
            ConvNormActivation(2, 24, kernel_size=5, stride=(2, 2)),
            DepthwiseSeparableBlock(24, 48, (2, 2)),
            DepthwiseSeparableBlock(48, 64, (2, 2)),
            DepthwiseSeparableBlock(64, 96, (2, 2)),
            DepthwiseSeparableBlock(96, 128, (1, 2)),
        )
        self.class_branch_norm = nn.LayerNorm(256)
        self.class_branch_head = nn.Linear(256, num_classes)
        self.class_branch_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectral = inputs[:, :5]
        embedding = self.pool(self.blocks(self.stem(spectral))).flatten(1)
        embedding = self.dropout(embedding)
        class_logits = self.class_head(embedding)
        azimuth_logits = self.azimuth_head(embedding)

        branch_features = self.class_branch(inputs[:, :2])
        branch_embedding = torch.cat(
            (
                branch_features.mean(dim=(2, 3)),
                branch_features.amax(dim=(2, 3)),
            ),
            dim=1,
        )
        branch_logits = self.class_branch_head(self.class_branch_norm(branch_embedding))
        class_logits = class_logits + self.class_branch_scale * branch_logits

        gcc_vector = inputs[:, 5, :, 0]
        azimuth_logits = azimuth_logits + self.gcc_scale * self.gcc_head(gcc_vector)
        return class_logits, azimuth_logits


class AudioMultiTaskCNNGCCStats(AudioMultiTaskCNNGCC):
    """GCC localizer with richer temporal class and frequency-aware angle heads."""

    def __init__(self, num_classes: int, num_azimuths: int) -> None:
        super().__init__(num_classes, num_azimuths)
        channels = DEFAULT_CHANNELS[-1]
        self.stats_attention = nn.Conv1d(channels, 1, kernel_size=1)
        self.stats_class_head = nn.Sequential(
            nn.LayerNorm(channels * 4),
            nn.Linear(channels * 4, 320),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(320, num_classes),
        )
        self.stats_azimuth_head = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, num_azimuths),
        )
        self.angle_head = nn.Linear(channels * 3, 2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectral = inputs[:, :5]
        features = self.blocks(self.stem(spectral))

        temporal = features.mean(dim=2)
        attention = torch.softmax(self.stats_attention(temporal), dim=2)
        attended = (temporal * attention).sum(dim=2)
        mean = temporal.mean(dim=2)
        std = temporal.var(dim=2, unbiased=False).add(1e-5).sqrt()
        maximum = temporal.amax(dim=2)
        class_embedding = torch.cat((mean, std, maximum, attended), dim=1)
        class_logits = self.stats_class_head(class_embedding)

        spatial = features.mean(dim=3)
        spatial_mean = spatial.mean(dim=2)
        spatial_std = spatial.var(dim=2, unbiased=False).add(1e-5).sqrt()
        spatial_max = spatial.amax(dim=2)
        azimuth_embedding = torch.cat((spatial_mean, spatial_std, spatial_max), dim=1)
        azimuth_logits = self.stats_azimuth_head(azimuth_embedding)
        gcc_vector = inputs[:, 5, :, 0]
        azimuth_logits = azimuth_logits + self.gcc_scale * self.gcc_head(gcc_vector)
        angle = nn.functional.normalize(self.angle_head(azimuth_embedding), dim=1)
        return class_logits, azimuth_logits, angle


class AudioMultiTaskCNNDisentangled(nn.Module):
    """Task-specific acoustic and spatial trunks with an explicit GCC cue.

    Sound identity should be invariant to binaural rendering, while azimuth
    estimation should not spend capacity modelling the event class.  Keeping
    the trunks separate prevents the two losses from fighting over the same
    compact representation.
    """

    def __init__(self, num_classes: int, num_azimuths: int) -> None:
        super().__init__()
        class_channels = (32, 48, 48, 64, 64, 96, 96, 128, 160)
        spatial_channels = (24, 32, 32, 48, 48, 64, 64, 80, 96)

        self.class_stem = ConvNormActivation(2, class_channels[0], kernel_size=5, stride=(2, 2))
        self.class_blocks = nn.Sequential(
            *[
                DepthwiseSeparableBlock(in_ch, out_ch, stride)
                for in_ch, out_ch, stride in zip(
                    class_channels[:-1], class_channels[1:], DEFAULT_STRIDES
                )
            ]
        )
        class_dim = class_channels[-1]
        self.class_attention = nn.Conv1d(class_dim, 1, kernel_size=1)
        self.class_head = nn.Sequential(
            nn.LayerNorm(class_dim * 4),
            nn.Linear(class_dim * 4, 320),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(320, num_classes),
        )

        self.spatial_stem = ConvNormActivation(3, spatial_channels[0], kernel_size=3, stride=(2, 2))
        self.spatial_blocks = nn.Sequential(
            *[
                DepthwiseSeparableBlock(in_ch, out_ch, stride)
                for in_ch, out_ch, stride in zip(
                    spatial_channels[:-1], spatial_channels[1:], DEFAULT_STRIDES
                )
            ]
        )
        spatial_dim = spatial_channels[-1]
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(spatial_dim * 3),
            nn.Linear(spatial_dim * 3, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(192, num_azimuths),
        )
        self.gcc_head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_azimuths),
        )
        self.gcc_scale = nn.Parameter(torch.tensor(0.5))
        self.angle_head = nn.Linear(spatial_dim * 3, 2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acoustic = self.class_blocks(self.class_stem(inputs[:, :2]))
        temporal = acoustic.mean(dim=2)
        attention = torch.softmax(self.class_attention(temporal), dim=2)
        attended = (temporal * attention).sum(dim=2)
        class_embedding = torch.cat(
            (
                temporal.mean(dim=2),
                temporal.var(dim=2, unbiased=False).add(1e-5).sqrt(),
                temporal.amax(dim=2),
                attended,
            ),
            dim=1,
        )
        class_logits = self.class_head(class_embedding)

        spatial = self.spatial_blocks(self.spatial_stem(inputs[:, 2:5])).mean(dim=3)
        spatial_embedding = torch.cat(
            (
                spatial.mean(dim=2),
                spatial.var(dim=2, unbiased=False).add(1e-5).sqrt(),
                spatial.amax(dim=2),
            ),
            dim=1,
        )
        gcc_vector = inputs[:, 5, :, 0]
        azimuth_logits = self.spatial_head(spatial_embedding)
        azimuth_logits = azimuth_logits + self.gcc_scale * self.gcc_head(gcc_vector)
        angle = nn.functional.normalize(self.angle_head(spatial_embedding), dim=1)
        return class_logits, azimuth_logits, angle


V2_CHANNELS = (48, 64, 64, 96, 96, 128, 128, 160, 192)
V3_CHANNELS = (32, 48, 48, 64, 64, 96, 96, 128, 160)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.gate(self.pool(inputs))


class AudioMultiTaskCNNV2(nn.Module):
    """Wider spatial CNN with channel attention and dual global pooling."""

    def __init__(
        self,
        num_classes: int,
        num_azimuths: int,
        in_channels: int = 5,
        channels: Sequence[int] = V2_CHANNELS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in channels)
        if len(channels) != len(DEFAULT_STRIDES) + 1:
            raise ValueError("channels length must be len(DEFAULT_STRIDES) + 1")

        self.stem = ConvNormActivation(in_channels, channels[0], kernel_size=5, stride=(2, 2))
        blocks = []
        for in_ch, out_ch, stride in zip(channels[:-1], channels[1:], DEFAULT_STRIDES):
            blocks.append(DepthwiseSeparableBlock(in_ch, out_ch, stride))
            blocks.append(SqueezeExcitation(out_ch))
        self.blocks = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        embedding_dim = channels[-1] * 2
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, channels[-1]),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.azimuth_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, channels[-1]),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_azimuths),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.blocks(self.stem(inputs))
        embedding = torch.cat((self.avg_pool(features), self.max_pool(features)), dim=1).flatten(1)
        embedding = self.dropout(embedding)
        return self.class_head(embedding), self.azimuth_head(embedding)


class AudioMultiTaskCNNV3(AudioMultiTaskCNNV2):
    """Compact attention CNN sized for two-view mirror TTA."""

    def __init__(self, num_classes: int, num_azimuths: int, in_channels: int = 5) -> None:
        super().__init__(num_classes, num_azimuths, in_channels=in_channels, channels=V3_CHANNELS, dropout=0.2)


def build_model(arch: str, num_classes: int, num_azimuths: int) -> nn.Module:
    if arch == "baseline":
        return AudioMultiTaskCNN(num_classes, num_azimuths)
    if arch == "baseline_angle":
        return AudioMultiTaskCNNAngle(num_classes, num_azimuths)
    if arch == "spatial":
        return AudioMultiTaskCNNSpatial(num_classes, num_azimuths)
    if arch == "dualpath":
        return AudioMultiTaskCNNDualPath(num_classes, num_azimuths)
    if arch == "gcc":
        return AudioMultiTaskCNNGCC(num_classes, num_azimuths)
    if arch == "gcc_temporal":
        return AudioMultiTaskCNNGCCTemporal(num_classes, num_azimuths)
    if arch == "gcc_class":
        return AudioMultiTaskCNNGCCClassBranch(num_classes, num_azimuths)
    if arch == "gcc_stats":
        return AudioMultiTaskCNNGCCStats(num_classes, num_azimuths)
    if arch == "disentangled":
        return AudioMultiTaskCNNDisentangled(num_classes, num_azimuths)
    if arch == "v2":
        return AudioMultiTaskCNNV2(num_classes, num_azimuths)
    if arch == "v3":
        return AudioMultiTaskCNNV3(num_classes, num_azimuths)
    raise ValueError(f"Unknown architecture: {arch}")
