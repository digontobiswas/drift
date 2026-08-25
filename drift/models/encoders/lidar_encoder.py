"""LiDAR point-cloud encoder producing a genuine 3D latent volume.

MODIFIED from Doracamom (Zhang et al., TCSVT 2026)'s radar branch. Doracamom broadcasts
its BEV radar feature to every height (``unsqueeze(-1).repeat(..., Z)``) because 4D radar
has almost no elevation resolution. LiDAR carries real height information, so DRIFT
replaces that broadcast with genuine 3D voxelization: point features are scattered into a
dense ``(X, Y, Z)`` grid with :func:`torch.scatter_reduce`, then refined by a small 3D (or
pillar+learned z-unfold) conv stack. See ``docs/DESIGN_SPEC.md`` §2.2 and §0.

No custom CUDA / spconv: voxelization uses only ``torch.scatter_reduce`` on flattened
voxel indices, batched with a Python loop over ``(B, T)`` samples (ragged point counts
prevent a single vectorized scatter across samples; the loop body is fully vectorized
over points).
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch
from torch import Tensor, nn

__all__ = ["LidarEncoder"]


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= max_groups dividing channels."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class LidarEncoder(nn.Module):
    """LiDAR points -> latent 3D volume.

    MODIFIED from Doracamom (real 3D volume, no height broadcast); see module docstring.

    Args:
        in_channels: Per-point feature dimension (e.g. 4 for x, y, z, intensity).
        out_channels: Output voxel feature channels.
        latent_size: Target latent voxel grid ``(X, Y, Z)``.
        point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in metres.
        backbone: ``"voxelnet"`` bins points directly into a 3D grid (mean-pooled) and
            refines it with a small ``Conv3d`` stack — the preferred, most literally
            height-resolved path. ``"pillar3d"`` bins points into 2D BEV pillars (mean-
            pooled over height) then applies a learned ``Conv2d(C -> C * Z)`` "z-unfold"
            so every height slice still gets independently-learned features (never a
            plain repeat along Z, which the spec explicitly forbids).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 64,
        latent_size: Tuple[int, int, int] = (128, 128, 10),
        point_cloud_range: List[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        backbone: Literal["pillar3d", "voxelnet"] = "pillar3d",
    ) -> None:
        super().__init__()
        if len(latent_size) != 3:
            raise ValueError(f"latent_size must be (X, Y, Z), got {latent_size!r}.")
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"point_cloud_range must have 6 elements, got {point_cloud_range!r}."
            )
        if backbone not in ("pillar3d", "voxelnet"):
            raise ValueError(
                f"backbone must be 'pillar3d' or 'voxelnet', got {backbone!r}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = tuple(int(v) for v in latent_size)
        self.point_cloud_range = [float(v) for v in point_cloud_range]
        self.backbone = backbone

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        X, Y, Z = self.latent_size
        self.voxel_size = (
            (x_max - x_min) / X,
            (y_max - y_min) / Y,
            (z_max - z_min) / Z,
        )

        # +1 channel for a per-voxel/pillar point-presence (occupancy count) feature.
        stem_in = in_channels + 1
        hidden = max(32, out_channels)
        if backbone == "voxelnet":
            self.trunk = nn.Sequential(
                nn.Conv3d(stem_in, hidden, 3, padding=1),
                _group_norm(hidden),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden, hidden, 3, padding=1),
                _group_norm(hidden),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden, out_channels, 3, padding=1),
            )
        else:  # pillar3d
            self.pillar_trunk = nn.Sequential(
                nn.Conv2d(stem_in, hidden, 3, padding=1),
                _group_norm(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                _group_norm(hidden),
                nn.ReLU(inplace=True),
            )
            self.z_unfold = nn.Conv2d(hidden, out_channels * Z, 1)

    def _voxelize_3d(self, pts: Tensor) -> Tensor:
        """Scatter one point cloud into a dense ``(in_channels+1, X, Y, Z)`` grid."""
        X, Y, Z = self.latent_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        grid = torch.zeros(self.in_channels + 1, X, Y, Z, device=device, dtype=dtype)
        if pts.numel() == 0:
            return grid
        pts = pts.to(device=device, dtype=dtype)
        if pts.shape[-1] != self.in_channels:
            raise ValueError(
                f"LiDAR points last dim must be in_channels={self.in_channels}, "
                f"got {pts.shape[-1]}."
            )
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        vx, vy, vz = self.voxel_size
        ix = torch.floor((pts[:, 0] - x_min) / vx).long()
        iy = torch.floor((pts[:, 1] - y_min) / vy).long()
        iz = torch.floor((pts[:, 2] - z_min) / vz).long()
        valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        if not torch.any(valid):
            return grid
        ix, iy, iz = ix[valid], iy[valid], iz[valid]
        feat = pts[valid]
        flat_idx = (ix * Y + iy) * Z + iz  # spec §1 flatten order: z fastest

        num_bins = X * Y * Z
        sums = torch.zeros(num_bins, self.in_channels, device=device, dtype=dtype)
        sums.scatter_reduce_(
            0, flat_idx.unsqueeze(-1).expand(-1, self.in_channels), feat,
            reduce="mean", include_self=False,
        )
        counts = torch.zeros(num_bins, device=device, dtype=dtype)
        counts.scatter_reduce_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=dtype),
            reduce="sum", include_self=False,
        )
        presence = (counts > 0).to(dtype)

        grid_feat = sums.view(X, Y, Z, self.in_channels).permute(3, 0, 1, 2)
        grid_presence = presence.view(1, X, Y, Z)
        return torch.cat([grid_feat, grid_presence], dim=0)

    def _pillarize(self, pts: Tensor) -> Tensor:
        """Scatter one point cloud into a dense ``(in_channels+1, X, Y)`` BEV grid."""
        X, Y, _ = self.latent_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        grid = torch.zeros(self.in_channels + 1, X, Y, device=device, dtype=dtype)
        if pts.numel() == 0:
            return grid
        pts = pts.to(device=device, dtype=dtype)
        if pts.shape[-1] != self.in_channels:
            raise ValueError(
                f"LiDAR points last dim must be in_channels={self.in_channels}, "
                f"got {pts.shape[-1]}."
            )
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        vx, vy, _ = self.voxel_size
        ix = torch.floor((pts[:, 0] - x_min) / vx).long()
        iy = torch.floor((pts[:, 1] - y_min) / vy).long()
        valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
        if not torch.any(valid):
            return grid
        ix, iy = ix[valid], iy[valid]
        feat = pts[valid]
        flat_idx = ix * Y + iy

        num_bins = X * Y
        sums = torch.zeros(num_bins, self.in_channels, device=device, dtype=dtype)
        sums.scatter_reduce_(
            0, flat_idx.unsqueeze(-1).expand(-1, self.in_channels), feat,
            reduce="mean", include_self=False,
        )
        counts = torch.zeros(num_bins, device=device, dtype=dtype)
        counts.scatter_reduce_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=dtype),
            reduce="sum", include_self=False,
        )
        presence = (counts > 0).to(dtype)

        grid_feat = sums.view(X, Y, self.in_channels).permute(2, 0, 1)
        grid_presence = presence.view(1, X, Y)
        return torch.cat([grid_feat, grid_presence], dim=0)

    def forward(self, points: List[List[Tensor]]) -> Tensor:
        """Encode ragged per-frame point clouds into the shared latent voxel grid.

        Args:
            points: Nested list ``[B][T]`` of ``(N_pts_i, in_channels)`` float tensors,
                each in that frame's LiDAR coordinates. Ragged by design (no padding
                required); an empty tensor for a frame is legal and yields an all-zero
                grid for that frame.

        Returns:
            ``(B, T, out_channels, X, Y, Z)`` latent voxel volume.

        Raises:
            ValueError: If ``points`` is empty, ragged across ``T``, or a per-frame
                tensor has the wrong feature dimension.
        """
        if not isinstance(points, list) or len(points) == 0:
            raise ValueError("points must be a non-empty list of length B.")
        B = len(points)
        if not isinstance(points[0], list):
            raise ValueError("points must be a nested list [B][T] of tensors.")
        T = len(points[0])
        for b in range(B):
            if not isinstance(points[b], list) or len(points[b]) != T:
                raise ValueError(
                    f"points is ragged in T: sample 0 has {T} frames, sample {b} has "
                    f"{len(points[b]) if isinstance(points[b], list) else 'N/A'}."
                )

        X, Y, Z = self.latent_size
        grids = []
        for b in range(B):
            for t in range(T):
                pts = points[b][t]
                if not torch.is_tensor(pts):
                    raise ValueError(
                        f"points[{b}][{t}] must be a torch.Tensor, got {type(pts)!r}."
                    )
                if self.backbone == "voxelnet":
                    grids.append(self._voxelize_3d(pts))
                else:
                    grids.append(self._pillarize(pts))
        stacked = torch.stack(grids, dim=0)  # (B*T, C+1, X, Y[, Z])

        if self.backbone == "voxelnet":
            out = self.trunk(stacked)  # (B*T, out_channels, X, Y, Z)
        else:
            feat = self.pillar_trunk(stacked)  # (B*T, hidden, X, Y)
            feat = self.z_unfold(feat)  # (B*T, out_channels*Z, X, Y)
            out = feat.view(B * T, self.out_channels, Z, X, Y).permute(0, 1, 3, 4, 2)
            out = out.contiguous()

        out = out.view(B, T, self.out_channels, X, Y, Z)
        return out
