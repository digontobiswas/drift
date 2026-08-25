"""StaticForecastPath: forecast the static scene by ego-pose warping.

★ NOVEL assembly component (part of the decoupled forecaster), built from a REUSED primitive
(`grid_sample`-based warping is standard practice; DFIT-OccWorld / OccProphet-style baselines use
a nearest-neighbour forward scatter instead -- see the SPEC-DEVIATION note below).

Near parameter-free: warps the present-frame observation latent into each future frame's ego
pose via inverse-transform backward sampling, using bilinear `grid_sample` with
`align_corners=False`. An optional small learned residual conv can compensate for warping seams
and (mild) scene evolution.

SPEC-DEVIATION note (intentional, per spec §2.9): the reference OccProphet code performs a
nearest-neighbour *forward* scatter of voxels (and does so with an off-by-one `range(split_num
- 1)` that silently drops ~3% of voxels). This implementation instead performs a *backward*
warp: for every voxel of the future frame, find where it was in the present frame and bilinearly
resample. This is differentiable, avoids the forward-scatter collision/gap problem entirely, and
is what the spec requires.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_DEFAULT_LATENT_SIZE = (128, 128, 10)
_DEFAULT_PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


class StaticForecastPath(nn.Module):
    """Forecast the static scene by ego-pose warping. Near parameter-free.

    Args:
        latent_size: (X, Y, Z) of the latent grid this module is configured for. The actual
            grid size is read from the input at forward time; this value is only used as a
            documented default / sanity expectation.
        point_cloud_range: [x_min,y_min,z_min,x_max,y_max,z_max] metric extent of the grid.
        learned_residual: If True, add a small learned `Conv3d` residual after warping.
        channels: Channel count of `obs_latent`, required to build the residual conv eagerly.
            If `learned_residual=True` and `channels` is None, a `LazyConv3d` is used instead
            (materialized on first call).
    """

    def __init__(
        self,
        latent_size: Tuple[int, int, int] = _DEFAULT_LATENT_SIZE,
        point_cloud_range: List[float] = _DEFAULT_PC_RANGE,
        learned_residual: bool = True,
        channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"StaticForecastPath: point_cloud_range must have 6 entries, got "
                f"{point_cloud_range}."
            )
        self.latent_size = tuple(latent_size)
        self.point_cloud_range = list(point_cloud_range)
        self.learned_residual = learned_residual

        self.residual_conv: Optional[nn.Conv3d] = None
        if learned_residual and channels is not None:
            self.residual_conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
            nn.init.zeros_(self.residual_conv.weight)
            nn.init.zeros_(self.residual_conv.bias)
        # If learned_residual=True and channels is None, the conv is built lazily on the first
        # forward() call (see _build_lazy_residual), once the actual channel count is known.

    def _build_lazy_residual(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        self.residual_conv = conv.to(device=device, dtype=dtype)

    def forward(self, obs_latent: Tensor, future_ego: Tensor) -> Tensor:
        """
        Args:
            obs_latent: (B, C, X, Y, Z) present-frame observation latent.
            future_ego: (B, T_o, 4, 4) cumulative transforms present -> each future frame, i.e.
                `future_ego[:, t]` maps a point expressed in present-frame coords to
                future-frame-`t` coords.

        Returns:
            (B, T_o, C, X, Y, Z)
        """
        if obs_latent.dim() != 5:
            raise ValueError(
                f"StaticForecastPath expects obs_latent (B,C,X,Y,Z), got "
                f"{tuple(obs_latent.shape)}."
            )
        if future_ego.dim() != 4 or future_ego.shape[-2:] != (4, 4):
            raise ValueError(
                f"StaticForecastPath expects future_ego (B,T_o,4,4), got "
                f"{tuple(future_ego.shape)}."
            )
        b, c, x_dim, y_dim, z_dim = obs_latent.shape
        if future_ego.shape[0] != b:
            raise ValueError(
                f"StaticForecastPath: batch mismatch, obs_latent B={b} vs future_ego "
                f"B={future_ego.shape[0]}."
            )
        t_o = future_ego.shape[1]
        device, dtype = obs_latent.device, obs_latent.dtype

        pcr = self.point_cloud_range
        x_min, y_min, z_min, x_max, y_max, z_max = pcr
        vx = (x_max - x_min) / x_dim
        vy = (y_max - y_min) / y_dim
        vz = (z_max - z_min) / z_dim

        xs = x_min + (torch.arange(x_dim, device=device, dtype=dtype) + 0.5) * vx
        ys = y_min + (torch.arange(y_dim, device=device, dtype=dtype) + 0.5) * vy
        zs = z_min + (torch.arange(z_dim, device=device, dtype=dtype) + 0.5) * vz
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")  # each (X,Y,Z)
        ones = torch.ones_like(gx)
        pts_future = torch.stack([gx, gy, gz, ones], dim=-1)  # (X,Y,Z,4), homogeneous

        # inverse of future_ego: present <- future
        inv = torch.linalg.inv(future_ego)  # (B,T_o,4,4)

        # pts_future: (X,Y,Z,4) -> (1,1,X,Y,Z,4,1) for batched matmul against (B,T_o,1,1,1,4,4)
        pts = pts_future.reshape(1, 1, x_dim, y_dim, z_dim, 4, 1).expand(
            b, t_o, x_dim, y_dim, z_dim, 4, 1
        )
        inv_b = inv.reshape(b, t_o, 1, 1, 1, 4, 4)
        pts_present = torch.matmul(inv_b, pts).squeeze(-1)  # (B,T_o,X,Y,Z,4)
        px = pts_present[..., 0]
        py = pts_present[..., 1]
        pz = pts_present[..., 2]

        # normalize to [-1, 1] for grid_sample (align_corners=False convention)
        nx = 2.0 * (px - x_min) / (x_max - x_min) - 1.0
        ny = 2.0 * (py - y_min) / (y_max - y_min) - 1.0
        nz = 2.0 * (pz - z_min) / (z_max - z_min) - 1.0

        # grid_sample 5D convention: input (N,C,D,H,W), grid[...,0]=W-index, [...,1]=H-index,
        # [...,2]=D-index. Our input is (B,C,X,Y,Z) i.e. D=X,H=Y,W=Z, so grid channels are
        # (z,y,x) in that order.
        grid = torch.stack([nz, ny, nx], dim=-1)  # (B,T_o,X,Y,Z,3)

        out_frames: List[Tensor] = []
        for t in range(t_o):
            sampled = F.grid_sample(
                obs_latent, grid[:, t], mode="bilinear", padding_mode="zeros",
                align_corners=False,
            )  # (B,C,X,Y,Z)
            out_frames.append(sampled)
        out = torch.stack(out_frames, dim=1)  # (B,T_o,C,X,Y,Z)

        if self.learned_residual:
            if self.residual_conv is None:
                self._build_lazy_residual(c, device, dtype)
            flat = out.reshape(b * t_o, c, x_dim, y_dim, z_dim)
            flat = flat + self.residual_conv(flat)
            out = flat.reshape(b, t_o, c, x_dim, y_dim, z_dim)

        return out
