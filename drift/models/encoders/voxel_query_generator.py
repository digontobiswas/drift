"""Coarse voxel query initialization from LiDAR geometry + camera semantics.

MODIFIED from Doracamom (Zhang et al., TCSVT 2026)'s Coarse Voxel Query Generator (CVQG).
Doracamom sums the radar and image query volumes unconditionally (``Q = Q_R + Q_I``)
because 4D radar is a weak, low-norm prior that image semantics can safely dominate. Real
LiDAR geometry is a *strong* signal — summing unconditionally would let raw LiDAR geometry
dominate the query with no way for the network to down-weight noisy or sparse regions. DRIFT
therefore defaults to a learned gate on the camera contribution instead of a plain sum. See
``docs/DESIGN_SPEC.md`` §2.3 and §0.
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch
from torch import Tensor, nn

__all__ = ["CoarseVoxelQueryGenerator"]


class CoarseVoxelQueryGenerator(nn.Module):
    """Initialize voxel queries from LiDAR geometry + camera semantics.

    MODIFIED from Doracamom CVQG; see module docstring.

    Args:
        embed_dims: Output query embedding dimension.
        latent_size: Latent voxel grid ``(X, Y, Z)`` (kept for interface symmetry with
            other modules; the per-voxel 1x1x1 convolutions used here do not themselves
            depend on the spatial extent).
        point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in metres (kept
            for interface symmetry; unused by the current per-voxel fusion).
        fusion: ``"gate"`` (default): ``Q = proj_l(L) + sigmoid(g(L)) * proj_c(C)`` — a
            learned gate on the LiDAR features controls how much camera semantics is let
            through, keeping LiDAR geometry as the controlled-dominant prior. ``"sum"``:
            plain ``Q = proj_l(L) + proj_c(C)``, kept for ablation. ``"concat"``:
            ``Q = proj(concat(L, C))``, kept for ablation.
    """

    # SPEC-DEVIATION: §2.3 lists only (embed_dims, latent_size, point_cloud_range,
    # fusion) in the constructor. A real (non-lazy) nn.Conv3d needs its input channel
    # count at construction time, so `lidar_channels`/`cam_channels` are added as extra
    # keyword args (defaulted to 64 to match the encoders' default `out_channels`).
    # Passing wrong values raises a clear ValueError in forward() rather than an opaque
    # Conv3d shape-mismatch.
    def __init__(
        self,
        embed_dims: int = 128,
        latent_size: Tuple[int, int, int] = (128, 128, 10),
        point_cloud_range: List[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        fusion: Literal["sum", "concat", "gate"] = "gate",
        lidar_channels: int = 64,
        cam_channels: int = 64,
    ) -> None:
        super().__init__()
        if fusion not in ("sum", "concat", "gate"):
            raise ValueError(f"fusion must be 'sum', 'concat', or 'gate', got {fusion!r}.")
        if len(latent_size) != 3:
            raise ValueError(f"latent_size must be (X, Y, Z), got {latent_size!r}.")

        self.embed_dims = embed_dims
        self.latent_size = tuple(int(v) for v in latent_size)
        self.point_cloud_range = [float(v) for v in point_cloud_range]
        self.fusion = fusion
        self.lidar_channels = lidar_channels
        self.cam_channels = cam_channels

        self.proj_l = nn.Conv3d(lidar_channels, embed_dims, 1)
        if fusion == "concat":
            self.proj_concat = nn.Conv3d(lidar_channels + cam_channels, embed_dims, 1)
        else:
            self.proj_c = nn.Conv3d(cam_channels, embed_dims, 1)
            if fusion == "gate":
                self.gate = nn.Conv3d(lidar_channels, embed_dims, 1)

    def forward(self, lidar_vol: Tensor, cam_vol: Tensor) -> Tensor:
        """Fuse LiDAR and camera latent volumes into initial voxel queries.

        Args:
            lidar_vol: ``(B, T, C_l, X, Y, Z)``.
            cam_vol: ``(B, T, C_c, X, Y, Z)``.

        Returns:
            ``(B, T, embed_dims, X, Y, Z)`` query volume.

        Raises:
            ValueError: On mismatched batch/time/spatial shapes or unexpected channel
                counts.
        """
        if lidar_vol.dim() != 6 or cam_vol.dim() != 6:
            raise ValueError(
                "lidar_vol and cam_vol must be (B, T, C, X, Y, Z); got shapes "
                f"{tuple(lidar_vol.shape)} and {tuple(cam_vol.shape)}."
            )
        B, T = lidar_vol.shape[:2]
        if cam_vol.shape[:2] != (B, T):
            raise ValueError(
                f"lidar_vol batch/time {(B, T)} does not match cam_vol "
                f"{tuple(cam_vol.shape[:2])}."
            )
        if lidar_vol.shape[3:] != cam_vol.shape[3:]:
            raise ValueError(
                f"lidar_vol spatial shape {tuple(lidar_vol.shape[3:])} does not match "
                f"cam_vol spatial shape {tuple(cam_vol.shape[3:])}."
            )
        if lidar_vol.shape[2] != self.lidar_channels:
            raise ValueError(
                f"lidar_vol has {lidar_vol.shape[2]} channels, expected "
                f"lidar_channels={self.lidar_channels}."
            )
        if cam_vol.shape[2] != self.cam_channels:
            raise ValueError(
                f"cam_vol has {cam_vol.shape[2]} channels, expected "
                f"cam_channels={self.cam_channels}."
            )

        X, Y, Z = lidar_vol.shape[3:]
        L = lidar_vol.reshape(B * T, self.lidar_channels, X, Y, Z)
        C = cam_vol.reshape(B * T, self.cam_channels, X, Y, Z)

        if self.fusion == "sum":
            q = self.proj_l(L) + self.proj_c(C)
        elif self.fusion == "concat":
            q = self.proj_concat(torch.cat([L, C], dim=1))
        else:  # gate
            gate = torch.sigmoid(self.gate(L))
            q = self.proj_l(L) + gate * self.proj_c(C)

        return q.view(B, T, self.embed_dims, X, Y, Z)
