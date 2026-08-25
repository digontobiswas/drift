"""Two-sided gated fusion of LiDAR and camera latent volumes.

MODIFIED from Doracamom (Zhang et al., TCSVT 2026)'s Cross-Modal BEV-Voxel Fusion (CMF).
Doracamom's residual carries *only* the camera feature (``add_radar=False``): the fused
output is ``f * sigmoid(g(f)) + camera``, encoding "camera is primary, radar merely
modulates it" — a reasonable choice for a weak, sparse 4D-radar prior. LiDAR is not a weak
prior, so DRIFT defaults to a symmetric two-sided residual that lets either modality
dominate where it is locally more informative. See ``docs/DESIGN_SPEC.md`` §2.5 and §0.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["CrossModalFusion"]


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= max_groups dividing channels."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class CrossModalFusion(nn.Module):
    """Two-sided gated fusion of LiDAR and camera latents.

    MODIFIED from Doracamom CMF; see module docstring.

    Both inputs must already share ``channels`` channels (e.g. both encoders configured
    with the same ``out_channels``, or both pre-projected upstream) — the fusion formula
    ``f = ConvBNReLU3D(concat(L, C))`` followed by ``... + L + C`` requires matching
    channel counts for the residual add.

    Args:
        channels: Shared input channel count of ``lidar_vol`` and ``cam_vol``.
        out_channels: Output channel count; defaults to ``channels`` if ``None``.
        two_sided: If ``True`` (default), use the symmetric two-sided residual
            ``f*sigmoid(g_l(f)) + f*sigmoid(g_c(f)) + L + C``. If ``False``, reproduce
            Doracamom's camera-only residual ``f*sigmoid(g_c(f)) + C``.
        aux_heads: If ``True``, also predict a binary occupied/free auxiliary logit per
            voxel from the fused output.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        two_sided: bool = True,
        aux_heads: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}.")
        self.channels = channels
        self.out_channels = out_channels if out_channels is not None else channels
        self.two_sided = two_sided
        self.aux_heads = aux_heads

        self.fuse_conv = nn.Sequential(
            nn.Conv3d(2 * channels, channels, 3, padding=1),
            _group_norm(channels),
            nn.ReLU(inplace=True),
        )
        if two_sided:
            self.gate_l = nn.Conv3d(channels, channels, 1)
        self.gate_c = nn.Conv3d(channels, channels, 1)

        self.out_proj = (
            nn.Identity()
            if self.out_channels == channels
            else nn.Conv3d(channels, self.out_channels, 1)
        )
        if aux_heads:
            self.occ_head = nn.Conv3d(self.out_channels, 1, 1)

    def forward(self, lidar_vol: Tensor, cam_vol: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Fuse LiDAR and camera latent volumes.

        Args:
            lidar_vol: ``(B, T, channels, X, Y, Z)``.
            cam_vol: ``(B, T, channels, X, Y, Z)``.

        Returns:
            Tuple of:
                fused: ``(B, T, out_channels, X, Y, Z)``.
                aux: dict with ``'occ_mask_logits': (B, T, 1, X, Y, Z)`` when
                    ``aux_heads=True``, else an empty dict.

        Raises:
            ValueError: On mismatched shapes or a channel count other than
                ``self.channels``.
        """
        if lidar_vol.dim() != 6 or cam_vol.dim() != 6:
            raise ValueError(
                "lidar_vol and cam_vol must be (B, T, C, X, Y, Z); got shapes "
                f"{tuple(lidar_vol.shape)} and {tuple(cam_vol.shape)}."
            )
        if lidar_vol.shape != cam_vol.shape:
            raise ValueError(
                f"lidar_vol shape {tuple(lidar_vol.shape)} must match cam_vol shape "
                f"{tuple(cam_vol.shape)}."
            )
        B, T, C = lidar_vol.shape[:3]
        if C != self.channels:
            raise ValueError(
                f"lidar_vol/cam_vol have {C} channels, expected channels={self.channels}."
            )
        X, Y, Z = lidar_vol.shape[3:]

        L = lidar_vol.reshape(B * T, C, X, Y, Z)
        Cm = cam_vol.reshape(B * T, C, X, Y, Z)

        f = self.fuse_conv(torch.cat([L, Cm], dim=1))
        if self.two_sided:
            out = f * torch.sigmoid(self.gate_l(f)) + f * torch.sigmoid(self.gate_c(f)) + L + Cm
        else:
            out = f * torch.sigmoid(self.gate_c(f)) + Cm

        out = self.out_proj(out)
        out = out.view(B, T, self.out_channels, X, Y, Z)

        aux: Dict[str, Tensor] = {}
        if self.aux_heads:
            flat = out.reshape(B * T, self.out_channels, X, Y, Z)
            occ_logits = self.occ_head(flat).view(B, T, 1, X, Y, Z)
            aux["occ_mask_logits"] = occ_logits

        return out, aux
