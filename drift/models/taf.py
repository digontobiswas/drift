"""Tripling-Attention Fusion (TAF).

REUSED from OccProphet (Sec. 3.2.2). Origin: OccProphet (ICLR 2025).

Three branches (scene / bev / height) summarize the latent voxel volume at different
granularities, each is refined by a **causal** temporal self-attention (queries may only
attend to the present and past frames), and the three refined summaries are broadcast-added
back into a full `(B,T,C,X,Y,Z)` volume.

Simplifications relative to the reference implementation (documented per spec §2.6):
  * The "windowed" spatial self-attention over the BEV branch is a plain
    `nn.MultiheadAttention` applied independently inside each non-overlapping window (with
    zero-padding to a multiple of `window_size` when `X`/`Y` are not evenly divisible), not a
    full Swin-style `ShiftWindowMSA` with shifted windows and relative position bias.
  * Zero-padded tokens participate in the in-window attention and are cropped away afterwards;
    for the sizes used in this codebase this is a negligible approximation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _causal_mask(timesteps: int, device: torch.device) -> Tensor:
    """Build an additive causal mask: position i may attend to j <= i only."""
    return torch.triu(
        torch.full((timesteps, timesteps), float("-inf"), device=device), diagonal=1
    )


class TriplingAttentionFusion(nn.Module):
    """Tripling-Attention Fusion. REUSED from OccProphet (Sec. 3.2.2).

    Args:
        embed_dims: Channel width of the input/output volume. Input `vox_feats` must have
            exactly this many channels.
        num_heads: Number of attention heads, shared by all four internal attention modules
            (one windowed-spatial, three causal-temporal). Must evenly divide `embed_dims`.
        window_size: Spatial window edge length (in voxels) for the BEV branch's windowed
            self-attention. Clamped/padded internally to handle grids not evenly divisible by
            this value (including tiny smoke-test grids).
        timesteps: Documented expected number of input frames; not enforced at forward time
            (the actual `T` is read from the input each call), kept only for interface parity
            with the spec.
        height_kernel_size: Kernel size of the 1D convolution applied along `z` in the height
            branch.
        residual: If True, add the original `vox_feats` to the output (requires the input to
            already be at `embed_dims` channels, which is required regardless).
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int = 7,
        timesteps: int = 3,
        height_kernel_size: int = 3,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"TriplingAttentionFusion: embed_dims ({embed_dims}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.timesteps = timesteps
        self.residual = residual

        self.scene_proj = nn.Conv3d(embed_dims, embed_dims, kernel_size=1)
        self.height_conv = nn.Conv1d(
            embed_dims, embed_dims, kernel_size=height_kernel_size,
            padding=height_kernel_size // 2,
        )

        self.spatial_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.scene_temporal_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.bev_temporal_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.height_temporal_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)

    def _temporal(self, x: Tensor, mha: nn.MultiheadAttention) -> Tensor:
        """Causal self-attention over the T axis. x: (N, T, C) -> (N, T, C)."""
        n, t, c = x.shape
        mask = _causal_mask(t, x.device)
        out, _ = mha(x, x, x, attn_mask=mask, need_weights=False)
        return out

    def _window_partition(self, x: Tensor) -> Tuple[Tensor, int, int]:
        """x: (N, X, Y, C) -> (N * nWin, ws*ws, C), padded X, padded Y."""
        n, x_dim, y_dim, c = x.shape
        ws = self.window_size
        pad_x = (ws - x_dim % ws) % ws
        pad_y = (ws - y_dim % ws) % ws
        if pad_x or pad_y:
            x = F.pad(x, (0, 0, 0, pad_y, 0, pad_x))
        xp, yp = x_dim + pad_x, y_dim + pad_y
        x = x.view(n, xp // ws, ws, yp // ws, ws, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        return x, xp, yp

    def _window_reverse(
        self, x: Tensor, n: int, xp: int, yp: int, x_dim: int, y_dim: int
    ) -> Tensor:
        ws = self.window_size
        c = x.shape[-1]
        n_x, n_y = xp // ws, yp // ws
        x = x.view(n, n_x, n_y, ws, ws, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(n, xp, yp, c)
        return x[:, :x_dim, :y_dim, :]

    def forward(self, vox_feats: Tensor) -> Tensor:
        """
        Args:
            vox_feats: (B, T, C, X, Y, Z) with C == embed_dims.

        Returns:
            (B, T, C, X, Y, Z), same shape as the input.
        """
        if vox_feats.dim() != 6:
            raise ValueError(
                f"TriplingAttentionFusion expects a 6D (B,T,C,X,Y,Z) tensor, got shape "
                f"{tuple(vox_feats.shape)}."
            )
        b, t, c, x_dim, y_dim, z_dim = vox_feats.shape
        if c != self.embed_dims:
            raise ValueError(
                f"TriplingAttentionFusion: input has {c} channels but embed_dims="
                f"{self.embed_dims}."
            )
        n = b * t
        flat = vox_feats.reshape(n, c, x_dim, y_dim, z_dim)

        # --- scene branch: (B,T,C) ---
        scene = F.adaptive_avg_pool3d(flat, 1)  # (N,C,1,1,1)
        scene = self.scene_proj(scene).reshape(b, t, c)
        scene = self._temporal(scene, self.scene_temporal_attn)  # (B,T,C)

        # --- bev branch: (B,T,X*Y,C) ---
        bev = flat.mean(dim=4)  # (N,C,X,Y)
        bev = bev.permute(0, 2, 3, 1)  # (N,X,Y,C)
        bev_win, xp, yp = self._window_partition(bev)
        bev_win, _ = self.spatial_attn(bev_win, bev_win, bev_win, need_weights=False)
        bev = self._window_reverse(bev_win, n, xp, yp, x_dim, y_dim)  # (N,X,Y,C)
        bev = bev.reshape(b, t, x_dim * y_dim, c)
        bev = bev.permute(0, 2, 1, 3).reshape(b * x_dim * y_dim, t, c)
        bev = self._temporal(bev, self.bev_temporal_attn)
        bev = bev.reshape(b, x_dim * y_dim, t, c).permute(0, 2, 1, 3)  # (B,T,X*Y,C)

        # --- height branch: (B,T,Z,C) ---
        # flat is (N,C,X,Y,Z); mean over X,Y (dims 2,3) leaves (N,C,Z).
        height = flat.mean(dim=(2, 3))
        height = self.height_conv(height)  # (N,C,Z)
        height = height.reshape(b, t, c, z_dim).permute(0, 1, 3, 2)  # (B,T,Z,C)
        height = height.permute(0, 2, 1, 3).reshape(b * z_dim, t, c)
        height = self._temporal(height, self.height_temporal_attn)
        height = height.reshape(b, z_dim, t, c).permute(0, 2, 1, 3)  # (B,T,Z,C)

        # --- combine ---
        bev_c = bev.permute(0, 1, 3, 2).reshape(b, t, c, x_dim, y_dim).unsqueeze(-1)
        scene_c = scene.reshape(b, t, c, 1, 1, 1)
        height_c = height.permute(0, 1, 3, 2).reshape(b, t, c, 1, 1, z_dim)

        out = bev_c + scene_c + height_c  # broadcast -> (B,T,C,X,Y,Z)
        if self.residual:
            out = out + vox_feats
        return out
