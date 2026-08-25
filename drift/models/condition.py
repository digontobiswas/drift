"""ConditionalForecaster: Condition Generator + Conditional Forecaster (hypernetwork).

REUSED from OccProphet (Sec. 3.3). Origin: OccProphet (ICLR 2025).

A hypernetwork: a global per-sample scene condition (pooled from the input volume) is fed
through a `Linear` layer that predicts the weights of a `Conv3d` kernel, which is then applied
-- grouped over the batch, one distinct kernel per sample -- to map `T_p` input frames to `T_o`
output frames, all in a single `F.conv3d` call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_MAX_HYPERNET_PARAMS = 5e7


class ConditionalForecaster(nn.Module):
    """Condition Generator + Conditional Forecaster. REUSED from OccProphet (Sec. 3.3).

    Args:
        in_timesteps: `T_p`, number of input frames.
        out_timesteps: `T_o`, number of output frames.
        in_channels: Channel width (shared by input and output).
        kernel_size: Spatial kernel edge length of the predicted `Conv3d` kernel. Must be odd
            (so `padding=kernel_size//2` preserves spatial size).
        norm_and_act: If True, apply LayerNorm + GELU to the pooled condition vector before the
            hypernetwork `Linear`.
    """

    def __init__(
        self,
        in_timesteps: int,
        out_timesteps: int,
        in_channels: int,
        kernel_size: int = 1,
        norm_and_act: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"ConditionalForecaster: kernel_size must be odd, got {kernel_size}."
            )
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        n_hyper_params = (
            in_timesteps * in_channels * out_timesteps * in_channels * kernel_size ** 3
        )
        if n_hyper_params >= _MAX_HYPERNET_PARAMS:
            raise ValueError(
                f"ConditionalForecaster: the hypernetwork Linear would need "
                f"{n_hyper_params:.3e} output params (in_timesteps={in_timesteps} * "
                f"in_channels={in_channels} * out_timesteps={out_timesteps} * in_channels="
                f"{in_channels} * kernel_size^3={kernel_size ** 3}), which exceeds the "
                f"{_MAX_HYPERNET_PARAMS:.0e} guard. Reduce in_channels, kernel_size, or the "
                f"timestep counts."
            )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.per_frame_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.norm_and_act = norm_and_act
        if norm_and_act:
            self.norm = nn.LayerNorm(in_timesteps * in_channels)
            self.act = nn.GELU()
        self.cond_linear = nn.Linear(
            in_timesteps * in_channels,
            out_timesteps * in_channels * in_timesteps * in_channels * kernel_size ** 3,
        )

    def forward(self, vox_feats: Tensor) -> Tensor:
        """
        Args:
            vox_feats: (B, T_p, C, X, Y, Z), T_p == in_timesteps, C == in_channels.

        Returns:
            (B, T_o, C, X, Y, Z)
        """
        if vox_feats.dim() != 6:
            raise ValueError(
                f"ConditionalForecaster expects a 6D (B,T_p,C,X,Y,Z) tensor, got shape "
                f"{tuple(vox_feats.shape)}."
            )
        b, t_p, c, x_dim, y_dim, z_dim = vox_feats.shape
        if t_p != self.in_timesteps or c != self.in_channels:
            raise ValueError(
                f"ConditionalForecaster: expected (T_p={self.in_timesteps}, C="
                f"{self.in_channels}), got (T_p={t_p}, C={c})."
            )

        x = vox_feats.reshape(b * t_p, c, x_dim, y_dim, z_dim)
        pooled = self.pool(x)  # (B*T_p, C, 1, 1, 1)
        pooled = self.per_frame_conv(pooled)
        pooled = pooled.reshape(b, t_p * c)
        if self.norm_and_act:
            pooled = self.act(self.norm(pooled))

        weights = self.cond_linear(pooled)  # (B, T_o*C*T_p*C*k^3)
        k = self.kernel_size
        kernel = weights.reshape(b * self.out_timesteps * c, t_p * c, k, k, k)

        x_in = vox_feats.reshape(1, b * t_p * c, x_dim, y_dim, z_dim)
        out = F.conv3d(x_in, kernel, groups=b, padding=k // 2)
        out = out.reshape(b, self.out_timesteps, c, x_dim, y_dim, z_dim)
        return out
