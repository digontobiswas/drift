"""Prediction heads: occupancy, flow, and (novel) per-voxel uncertainty.

`OccupancyHead` / `FlowHead`: REUSED (standard dense prediction heads over the OccProphet-style
latent grid). `UncertaintyHead`: ★ NOVEL.

All heads operate at **latent** resolution (e.g. 128x128x10); upsampling to full resolution
(512x512x40) happens only in the metric code, by trilinear interpolation of logits followed by
argmax -- never argmax first (see spec §2.14).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class OccupancyHead(nn.Module):
    """Per-voxel semantic logits at full (latent) resolution.

    Args:
        in_channels: Channels of the input feature volume.
        num_classes: Number of semantic classes (index 0 == EMPTY_IDX / free).
        hidden_channels: Optional hidden width for a small two-layer conv head; if None, a
            single 1x1x1 conv is used.
    """

    def __init__(
        self, in_channels: int, num_classes: int, hidden_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        if hidden_channels is None:
            self.net = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        else:
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                nn.GELU(),
                nn.Conv3d(hidden_channels, num_classes, kernel_size=1),
            )

    def forward(self, feats: Tensor) -> Tensor:
        """
        Args:
            feats: (B, T_o, C, X, Y, Z), C == in_channels.

        Returns:
            (B, T_o, num_classes, X, Y, Z)
        """
        if feats.dim() != 6:
            raise ValueError(
                f"OccupancyHead expects (B,T_o,C,X,Y,Z), got {tuple(feats.shape)}."
            )
        b, t, c, x_dim, y_dim, z_dim = feats.shape
        if c != self.in_channels:
            raise ValueError(
                f"OccupancyHead: input has {c} channels but in_channels={self.in_channels}."
            )
        flat = feats.reshape(b * t, c, x_dim, y_dim, z_dim)
        out = self.net(flat)
        return out.reshape(b, t, self.num_classes, x_dim, y_dim, z_dim)


class FlowHead(nn.Module):
    """Per-voxel 3D flow, in units of LATENT voxels (0.8 m), backward centroid-offset convention.

    Args:
        in_channels: Channels of the input feature volume.
        hidden_channels: Optional hidden width for a small two-layer conv head; if None, a
            single 1x1x1 conv is used.
    """

    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        if hidden_channels is None:
            self.net = nn.Conv3d(in_channels, 3, kernel_size=1)
        else:
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                nn.GELU(),
                nn.Conv3d(hidden_channels, 3, kernel_size=1),
            )

    def forward(self, feats: Tensor) -> Tensor:
        """
        Args:
            feats: (B, T_o, C, X, Y, Z), C == in_channels.

        Returns:
            (B, T_o, 3, X, Y, Z)
        """
        if feats.dim() != 6:
            raise ValueError(f"FlowHead expects (B,T_o,C,X,Y,Z), got {tuple(feats.shape)}.")
        b, t, c, x_dim, y_dim, z_dim = feats.shape
        if c != self.in_channels:
            raise ValueError(
                f"FlowHead: input has {c} channels but in_channels={self.in_channels}."
            )
        flat = feats.reshape(b * t, c, x_dim, y_dim, z_dim)
        out = self.net(flat)
        return out.reshape(b, t, 3, x_dim, y_dim, z_dim)


class UncertaintyHead(nn.Module):
    """★ NOVEL. Per-voxel predictive uncertainty that grows with horizon.

    Consumes the latent feature volume plus (optionally) the CMLI cross-modal disagreement
    signal, and predicts a raw (unbounded) per-voxel log-variance; callers apply
    softplus/exp as needed. A learned per-horizon bias lets the network express uncertainty
    that systematically grows with forecast horizon without hand-coding a schedule.

    Args:
        in_channels: Channels of the input feature volume.
        num_future: `T_o`, used to size the per-horizon bias.
        mode: `"variance"` (default) directly regresses log-variance via a conv head.
            `"evidential"` instead predicts Normal-Inverse-Gamma parameters (gamma, nu, alpha,
            beta) and derives an aleatoric+epistemic log-variance from them
            (`log(beta * (1+nu) / (nu * alpha))`), a documented simplification of full
            evidential deep learning (no NLL-specific loss term is implemented here; that lives
            in `drift/losses/uncertainty.py`).
        use_disagreement: If True, `forward` requires a `disagreement` tensor and concatenates
            it as an extra input channel.
    """

    def __init__(
        self,
        in_channels: int,
        num_future: int,
        mode: Literal["variance", "evidential"] = "variance",
        use_disagreement: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("variance", "evidential"):
            raise ValueError(
                f"UncertaintyHead: mode must be 'variance' or 'evidential', got {mode!r}."
            )
        self.in_channels = in_channels
        self.num_future = num_future
        self.mode = mode
        self.use_disagreement = use_disagreement

        conv_in = in_channels + (1 if use_disagreement else 0)
        out_ch = 1 if mode == "variance" else 4  # evidential: gamma, nu, alpha, beta
        self.net = nn.Sequential(
            nn.Conv3d(conv_in, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, out_ch, kernel_size=1),
        )
        self.horizon_bias = nn.Parameter(torch.zeros(num_future))

    def forward(self, feats: Tensor, disagreement: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            feats: (B, T_o, C, X, Y, Z), C == in_channels.
            disagreement: (B, T_o, 1, X, Y, Z), required iff `use_disagreement=True`. Must
                already be temporally aligned to `feats`' T_o horizon axis -- the caller (top
                -level DRIFT assembly) is responsible for broadcasting/aligning CMLI's
                per-observation-frame disagreement signal to the T_o future horizon before
                calling this head.

        Returns:
            (B, T_o, 1, X, Y, Z) raw log-variance (unbounded).
        """
        if feats.dim() != 6:
            raise ValueError(f"UncertaintyHead expects (B,T_o,C,X,Y,Z), got {tuple(feats.shape)}.")
        b, t, c, x_dim, y_dim, z_dim = feats.shape
        if c != self.in_channels:
            raise ValueError(
                f"UncertaintyHead: input has {c} channels but in_channels={self.in_channels}."
            )
        if t != self.num_future:
            raise ValueError(
                f"UncertaintyHead: feats has T_o={t} but num_future={self.num_future}."
            )

        if self.use_disagreement:
            if disagreement is None:
                raise ValueError(
                    "UncertaintyHead was built with use_disagreement=True but forward() was "
                    "called without a disagreement tensor."
                )
            if tuple(disagreement.shape) != (b, t, 1, x_dim, y_dim, z_dim):
                raise ValueError(
                    f"UncertaintyHead: expected disagreement shape {(b, t, 1, x_dim, y_dim, z_dim)}, "
                    f"got {tuple(disagreement.shape)}."
                )
            x = torch.cat([feats, disagreement], dim=2)
        else:
            if disagreement is not None:
                raise ValueError(
                    "UncertaintyHead: disagreement was provided but this module was built with "
                    "use_disagreement=False."
                )
            x = feats

        flat = x.reshape(b * t, x.shape[2], x_dim, y_dim, z_dim)
        out = self.net(flat)
        out = out.reshape(b, t, out.shape[1], x_dim, y_dim, z_dim)

        if self.mode == "variance":
            log_var = out[:, :, 0]  # (B,T_o,X,Y,Z)
        else:
            gamma, nu, alpha, beta = out.unbind(dim=2)
            nu = F.softplus(nu) + 1e-6
            alpha = F.softplus(alpha) + 1.0 + 1e-6
            beta = F.softplus(beta) + 1e-6
            var = beta * (1.0 + nu) / (nu * alpha)
            log_var = torch.log(var + 1e-8)

        bias = self.horizon_bias.reshape(1, t, 1, 1, 1)
        log_var = log_var + bias
        return log_var.unsqueeze(2)
