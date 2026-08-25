"""Instance-query dynamic path. ★ NOVEL (core contribution).

Contrast with DFIT-OccWorld (2412.13772), which decouples static/dynamic forecasting using a
DENSE per-voxel flow field. Cam4DOcc's flow ground truth is *already* instance-centric (a
per-instance backward centroid-offset rendered densely, see spec §0), so predicting it with a
dense field spends model capacity re-deriving a structure known a priori. This module instead
extracts a small set of object queries from the observation latent (`InstanceQueryExtractor`),
rolls each one forward through time as an explicit rigid-body-ish state
(`MotionForecaster`), and rasterizes the forecasted states back into the voxel grid
(`InstanceSplatter`) with a differentiable soft-splat, so gradient from the dense occupancy/flow
losses reaches the instance queries directly.

No prior published component is reused here; general query-based-detection (DETR-style) and
splatting ideas are well established in the literature but this specific
extract -> roll-forward -> soft-splat pipeline, and its role as one branch of a decoupled
static/dynamic forecaster, is the paper's novel contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_DEFAULT_LATENT_SIZE = (128, 128, 10)
_DEFAULT_PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def _auto_num_heads(channels: int, max_heads: int = 8) -> int:
    for h in (max_heads, 4, 2, 1):
        if h <= channels and channels % h == 0:
            return h
    return 1


@dataclass
class InstanceState:
    """Per-query agent state. All tensors (B, Q, ...).

    Attributes:
        center: (B, Q, 3) metric xyz in present-frame ego coordinates.
        size: (B, Q, 3) length, width, height in metres (positive).
        yaw: (B, Q, 1) heading, radians.
        velocity: (B, Q, 3) m/s.
        logits: (B, Q, num_classes) unnormalized class scores.
        embed: (B, Q, C) query feature.
        score: (B, Q, 1) objectness in [0, 1].
    """

    center: Tensor
    size: Tensor
    yaw: Tensor
    velocity: Tensor
    logits: Tensor
    embed: Tensor
    score: Tensor


def _sincos_pos_encoding_3d(
    x_dim: int, y_dim: int, z_dim: int, channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Fixed sinusoidal 3D positional encoding, flattened (x,y,z) with z fastest.

    Returns:
        (X*Y*Z, channels)
    """
    if channels % 6 != 0:
        pad = 6 - (channels % 6)
    else:
        pad = 0
    c_per_axis = (channels + pad) // 3
    if c_per_axis % 2 != 0:
        c_per_axis += 1

    def axis_pe(n: int) -> Tensor:
        pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, c_per_axis, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / c_per_axis)
        )
        pe = torch.zeros(n, c_per_axis, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    pe_x = axis_pe(x_dim)  # (X, c_per_axis)
    pe_y = axis_pe(y_dim)
    pe_z = axis_pe(z_dim)

    grid = torch.cat(
        [
            pe_x.reshape(x_dim, 1, 1, c_per_axis).expand(x_dim, y_dim, z_dim, c_per_axis),
            pe_y.reshape(1, y_dim, 1, c_per_axis).expand(x_dim, y_dim, z_dim, c_per_axis),
            pe_z.reshape(1, 1, z_dim, c_per_axis).expand(x_dim, y_dim, z_dim, c_per_axis),
        ],
        dim=-1,
    )  # (X,Y,Z, 3*c_per_axis)
    grid = grid.reshape(x_dim * y_dim * z_dim, 3 * c_per_axis)
    return grid[:, :channels]


class InstanceQueryExtractor(nn.Module):
    """★ NOVEL. Extract dynamic-agent queries from the observation latent.

    A DETR-style query decoder: `num_queries` learned embeddings cross-attend to the flattened
    observation latent (with a fixed sinusoidal 3D positional encoding, since the grid size is
    not known until forward time) through a stack of `nn.TransformerDecoderLayer`s, then small
    per-attribute linear heads read off geometry, class, and objectness.

    Args:
        in_channels: Channels of `obs_latent`.
        embed_dims: Internal query/token width.
        num_queries: Number of instance queries, `Q`.
        num_classes: Number of dynamic-object classes.
        num_decoder_layers: Depth of the transformer decoder.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dims: int = 256,
        num_queries: int = 300,
        num_classes: int = 2,
        num_decoder_layers: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.input_proj = nn.Conv3d(in_channels, embed_dims, kernel_size=1)
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dims) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=_auto_num_heads(embed_dims),
            dim_feedforward=embed_dims * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)

        self.center_head = nn.Linear(embed_dims, 3)
        self.size_head = nn.Linear(embed_dims, 3)
        self.yaw_head = nn.Linear(embed_dims, 1)
        self.velocity_head = nn.Linear(embed_dims, 3)
        self.logits_head = nn.Linear(embed_dims, num_classes)
        self.score_head = nn.Linear(embed_dims, 1)

    def forward(self, obs_latent: Tensor) -> InstanceState:
        """
        Args:
            obs_latent: (B, C, X, Y, Z), C == in_channels.

        Returns:
            InstanceState with Q == num_queries.
        """
        if obs_latent.dim() != 5:
            raise ValueError(
                f"InstanceQueryExtractor expects obs_latent (B,C,X,Y,Z), got "
                f"{tuple(obs_latent.shape)}."
            )
        b, c, x_dim, y_dim, z_dim = obs_latent.shape
        if c != self.in_channels:
            raise ValueError(
                f"InstanceQueryExtractor: input has {c} channels but in_channels="
                f"{self.in_channels}."
            )

        tokens = self.input_proj(obs_latent)  # (B, embed_dims, X, Y, Z)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(b, x_dim * y_dim * z_dim, self.embed_dims)
        pos = _sincos_pos_encoding_3d(
            x_dim, y_dim, z_dim, self.embed_dims, obs_latent.device, tokens.dtype
        )
        memory = tokens + pos.unsqueeze(0)

        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)  # (B, Q, embed_dims)

        center = self.center_head(decoded)
        size = F.softplus(self.size_head(decoded)) + 1e-2
        yaw = self.yaw_head(decoded)
        velocity = self.velocity_head(decoded)
        logits = self.logits_head(decoded)
        score = torch.sigmoid(self.score_head(decoded))

        return InstanceState(
            center=center, size=size, yaw=yaw, velocity=velocity, logits=logits,
            embed=decoded, score=score,
        )


class MotionForecaster(nn.Module):
    """★ NOVEL. Roll instance queries forward to each future horizon.

    Args:
        embed_dims: Query feature width, must match `InstanceState.embed`'s last dim.
        num_future: Number of future horizons to produce, `T_o`.
        mode: `"gru"` recurrently updates a `GRUCell` hidden state once per horizon (default).
            `"transformer"` instead runs a single causal `TransformerEncoderLayer` over
            `[initial_state, step_1, ..., step_num_future]` in parallel -- a documented
            simplification of a full decoder stack, sufficient at this scale.
        scene_condition_dims: If given, `forward` requires a `scene_cond: (B, scene_condition_dims)`
            global context vector (e.g. pooled from `ConditionalForecaster`), projected and added
            to every per-step input.

    Design notes: `size` and `logits` (object class) are carried forward unchanged at every
    horizon -- a rigid-body / fixed-class assumption reasonable at a 0.5s-per-step, few-second
    horizon. Motion is integrated physically: the incoming `state.velocity` seeds `v_0`, each step
    predicts a velocity increment (acceleration), and position advances as
    `c_t = c_{t-1} + v_t * dt + residual`. This makes both the extractor's and the forecaster's
    velocity heads causally load-bearing, so the trajectory loss on `center` supervises them
    without needing a separate velocity target.

    Args (cont.):
        dt: Seconds per forecast step (nuScenes keyframes are 2 Hz -> 0.5 s).
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_future: int = 6,
        mode: Literal["gru", "transformer"] = "gru",
        scene_condition_dims: Optional[int] = None,
        dt: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in ("gru", "transformer"):
            raise ValueError(f"MotionForecaster: mode must be 'gru' or 'transformer', got {mode!r}.")
        self.embed_dims = embed_dims
        self.num_future = num_future
        self.mode = mode
        self.scene_condition_dims = scene_condition_dims

        self.step_embed = nn.Parameter(torch.randn(num_future, embed_dims) * 0.02)

        if scene_condition_dims is not None:
            self.scene_proj: Optional[nn.Linear] = nn.Linear(scene_condition_dims, embed_dims)
        else:
            self.scene_proj = None

        if mode == "gru":
            self.gru_cell: Optional[nn.GRUCell] = nn.GRUCell(embed_dims, embed_dims)
            self.temporal_layer: Optional[nn.TransformerEncoderLayer] = None
        else:
            self.gru_cell = None
            self.temporal_layer = nn.TransformerEncoderLayer(
                d_model=embed_dims, nhead=_auto_num_heads(embed_dims),
                dim_feedforward=embed_dims * 4, dropout=0.0, batch_first=True,
            )

        self.delta_center = nn.Linear(embed_dims, 3)
        self.delta_yaw = nn.Linear(embed_dims, 1)
        # Predicts a velocity INCREMENT (acceleration * dt), not an absolute velocity. Position is
        # integrated from velocity so that `velocity` is causally load-bearing rather than a
        # decorative regression output -- the trajectory loss on `center` then supervises it.
        self.velocity_head = nn.Linear(embed_dims, 3)
        self.score_head = nn.Linear(embed_dims, 1)
        self.dt = float(dt)

    def forward(self, state: InstanceState, scene_cond: Optional[Tensor] = None) -> List[InstanceState]:
        """
        Args:
            state: Present-frame InstanceState (Q queries).
            scene_cond: (B, scene_condition_dims), required iff the module was built with
                `scene_condition_dims` set.

        Returns:
            List of length `num_future`, each an InstanceState at that future horizon.
        """
        b, q, c = state.embed.shape
        if c != self.embed_dims:
            raise ValueError(
                f"MotionForecaster: state.embed has {c} channels but embed_dims={self.embed_dims}."
            )

        if self.scene_condition_dims is not None:
            if scene_cond is None:
                raise ValueError(
                    "MotionForecaster was built with scene_condition_dims set, but forward() "
                    "was called without scene_cond."
                )
            if scene_cond.shape != (b, self.scene_condition_dims):
                raise ValueError(
                    f"MotionForecaster: expected scene_cond shape ({b},{self.scene_condition_dims}), "
                    f"got {tuple(scene_cond.shape)}."
                )
            assert self.scene_proj is not None
            cond = self.scene_proj(scene_cond).unsqueeze(1).expand(b, q, self.embed_dims)
            cond = cond.reshape(b * q, self.embed_dims)
        else:
            if scene_cond is not None:
                raise ValueError(
                    "MotionForecaster: scene_cond was provided but this module was not built "
                    "with scene_condition_dims set."
                )
            cond = None

        center = state.center
        yaw = state.yaw
        velocity = state.velocity  # seeds v_0 -> makes the extractor's velocity head load-bearing
        hidden0 = state.embed.reshape(b * q, c)

        out_states: List[InstanceState] = []

        if self.mode == "gru":
            assert self.gru_cell is not None
            hidden = hidden0
            for t in range(self.num_future):
                step_inp = self.step_embed[t].unsqueeze(0).expand(b * q, -1)
                if cond is not None:
                    step_inp = step_inp + cond
                hidden = self.gru_cell(step_inp, hidden)
                out_states.append(self._heads(hidden, center, yaw, velocity, state, b, q))
                center = out_states[-1].center
                yaw = out_states[-1].yaw
                velocity = out_states[-1].velocity
        else:
            assert self.temporal_layer is not None
            step_embeds = self.step_embed.unsqueeze(0).expand(b * q, -1, -1)
            if cond is not None:
                step_embeds = step_embeds + cond.unsqueeze(1)
            seq = torch.cat([hidden0.unsqueeze(1), step_embeds], dim=1)  # (B*Q, 1+T_o, C)
            s = seq.shape[1]
            mask = torch.triu(
                torch.full((s, s), float("-inf"), device=seq.device, dtype=seq.dtype), diagonal=1
            )
            out_seq = self.temporal_layer(seq, src_mask=mask)
            for t in range(self.num_future):
                hidden = out_seq[:, t + 1, :]
                out_states.append(self._heads(hidden, center, yaw, velocity, state, b, q))
                center = out_states[-1].center
                yaw = out_states[-1].yaw
                velocity = out_states[-1].velocity

        return out_states

    def _heads(
        self, hidden: Tensor, center: Tensor, yaw: Tensor, velocity: Tensor,
        state: InstanceState, b: int, q: int,
    ) -> InstanceState:
        """One integration step: v <- v + a*dt ; c <- c + v*dt + residual."""
        new_velocity = velocity + self.velocity_head(hidden).reshape(b, q, 3)
        new_center = (
            center
            + new_velocity * self.dt
            + self.delta_center(hidden).reshape(b, q, 3)  # residual for non-constant-velocity
        )
        new_yaw = yaw + self.delta_yaw(hidden).reshape(b, q, 1)
        score = torch.sigmoid(self.score_head(hidden)).reshape(b, q, 1)
        embed_out = hidden.reshape(b, q, self.embed_dims)
        return InstanceState(
            center=new_center, size=state.size, yaw=new_yaw, velocity=new_velocity,
            logits=state.logits, embed=embed_out, score=score,
        )


class InstanceSplatter(nn.Module):
    """★ NOVEL. Rasterize forecasted instance states back into the latent voxel grid.

    For each query, a soft (Gaussian) box-indicator is evaluated over that query's local
    bounding sub-grid only (a fixed-size window of voxels around its rounded center), in the
    query's own rotated (yaw) frame, weighted by `score`, and `scatter_add`-ed into the grid --
    vectorized over queries and batch; the only Python-level loop is over the (small) list of
    future timesteps. This keeps the operation cheap (`O(Q * window_size)` instead of
    `O(Q * X*Y*Z)`) while remaining fully differentiable end-to-end: gradient flows both through
    the scattered *feature* (via a linear projection of `embed`) and through the scattered
    *weight* (via `center`, `size`, `yaw`, `score`), so `InstanceQueryExtractor`'s parameters
    receive a training signal from the dense occupancy/flow losses even though only a spatially
    local window is ever touched per query.

    Args:
        embed_dims: Channel width of `InstanceState.embed`.
        out_channels: Output feature channels.
        latent_size: (X, Y, Z) of the latent grid.
        point_cloud_range: [x_min,y_min,z_min,x_max,y_max,z_max] metric extent of the grid.
        soft: If False, disables the Gaussian falloff and only score-weights a hard box (kept
            for ablations); default True.
        sigma: Softness multiplier on each query's half-extent used as the Gaussian std.
        window_voxels: (Wx, Wy, Wz) fixed local sub-grid size (in voxels) searched around each
            query's center. Not part of the literal spec signature; an additive, backward
            -compatible keyword controlling the splat/compute-cost tradeoff.
    """

    def __init__(
        self,
        embed_dims: int,
        out_channels: int,
        latent_size: Tuple[int, int, int] = _DEFAULT_LATENT_SIZE,
        point_cloud_range: List[float] = _DEFAULT_PC_RANGE,
        soft: bool = True,
        sigma: float = 1.0,
        window_voxels: Tuple[int, int, int] = (7, 7, 5),
    ) -> None:
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"InstanceSplatter: point_cloud_range must have 6 entries, got "
                f"{point_cloud_range}."
            )
        self.embed_dims = embed_dims
        self.out_channels = out_channels
        self.latent_size = tuple(latent_size)
        self.point_cloud_range = list(point_cloud_range)
        self.soft = soft
        self.sigma = sigma
        self.window_voxels = tuple(
            min(w, latent_size[i]) if latent_size[i] > 0 else w
            for i, w in enumerate(window_voxels)
        )

        self.feat_proj = nn.Linear(embed_dims, out_channels)

        wx, wy, wz = self.window_voxels
        rx = torch.arange(wx) - wx // 2
        ry = torch.arange(wy) - wy // 2
        rz = torch.arange(wz) - wz // 2
        ox, oy, oz = torch.meshgrid(rx, ry, rz, indexing="ij")
        offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)  # (W,3) long
        self.register_buffer("_offsets", offsets, persistent=False)

    def forward(self, states: List[InstanceState]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            states: List of length T_o of InstanceState (Q queries each), e.g. from
                `MotionForecaster`.

        Returns:
            dyn_feats: (B, T_o, out_channels, X, Y, Z)
            dyn_occ:   (B, T_o, 1, X, Y, Z) soft occupancy mass, used as the merge gate.
        """
        if len(states) == 0:
            raise ValueError("InstanceSplatter.forward received an empty states list.")

        x_dim, y_dim, z_dim = self.latent_size
        pcr = self.point_cloud_range
        x_min, y_min, z_min, x_max, y_max, z_max = pcr
        vx = (x_max - x_min) / x_dim
        vy = (y_max - y_min) / y_dim
        vz = (z_max - z_min) / z_dim
        voxel_size = torch.tensor([vx, vy, vz], device=states[0].embed.device, dtype=states[0].embed.dtype)
        pc_min = torch.tensor([x_min, y_min, z_min], device=voxel_size.device, dtype=voxel_size.dtype)

        offsets = self._offsets.to(device=voxel_size.device)  # (W,3) long
        w = offsets.shape[0]

        feats_out: List[Tensor] = []
        occ_out: List[Tensor] = []

        for state in states:
            b, q, c = state.embed.shape
            if c != self.embed_dims:
                raise ValueError(
                    f"InstanceSplatter: state.embed has {c} channels but embed_dims="
                    f"{self.embed_dims}."
                )

            feat = self.feat_proj(state.embed)  # (B,Q,out_channels)

            center = state.center  # (B,Q,3)
            size = state.size  # (B,Q,3)
            yaw = state.yaw[..., 0]  # (B,Q)
            score = state.score[..., 0]  # (B,Q)

            vox_coord = (center - pc_min) / voxel_size - 0.5  # (B,Q,3) continuous voxel index
            anchor = torch.round(vox_coord.detach()).long()  # (B,Q,3)

            abs_idx = anchor.unsqueeze(2) + offsets.reshape(1, 1, w, 3)  # (B,Q,W,3)
            valid = (
                (abs_idx[..., 0] >= 0) & (abs_idx[..., 0] < x_dim)
                & (abs_idx[..., 1] >= 0) & (abs_idx[..., 1] < y_dim)
                & (abs_idx[..., 2] >= 0) & (abs_idx[..., 2] < z_dim)
            )  # (B,Q,W)
            clipped = torch.stack(
                [
                    abs_idx[..., 0].clamp(0, x_dim - 1),
                    abs_idx[..., 1].clamp(0, y_dim - 1),
                    abs_idx[..., 2].clamp(0, z_dim - 1),
                ],
                dim=-1,
            )  # (B,Q,W,3)

            anchor_metric = pc_min + (anchor.to(voxel_size.dtype) + 0.5) * voxel_size  # (B,Q,3)
            voxel_metric = anchor_metric.unsqueeze(2) + offsets.reshape(1, 1, w, 3).to(voxel_size.dtype) * voxel_size
            # (B,Q,W,3)
            delta = voxel_metric - center.unsqueeze(2)  # (B,Q,W,3), differentiable wrt center

            cos_y = torch.cos(yaw).unsqueeze(-1)  # (B,Q,1)
            sin_y = torch.sin(yaw).unsqueeze(-1)
            dx, dy, dz = delta[..., 0], delta[..., 1], delta[..., 2]
            local_x = dx * cos_y + dy * sin_y
            local_y = -dx * sin_y + dy * cos_y
            local_z = dz

            half = size / 2.0  # (B,Q,3)
            std_x = (half[..., 0:1] * self.sigma).clamp_min(1e-3)
            std_y = (half[..., 1:2] * self.sigma).clamp_min(1e-3)
            std_z = (half[..., 2:3] * self.sigma).clamp_min(1e-3)

            if self.soft:
                gauss = torch.exp(
                    -0.5
                    * (
                        (local_x / std_x) ** 2
                        + (local_y / std_y) ** 2
                        + (local_z / std_z) ** 2
                    )
                )
            else:
                inside = (
                    (local_x.abs() <= half[..., 0:1]) & (local_y.abs() <= half[..., 1:2])
                    & (local_z.abs() <= half[..., 2:3])
                )
                gauss = inside.to(voxel_size.dtype)

            weight = gauss * score.unsqueeze(-1)  # (B,Q,W)
            weight = weight * valid.to(weight.dtype)

            flat_idx = (clipped[..., 0] * y_dim + clipped[..., 1]) * z_dim + clipped[..., 2]  # (B,Q,W)
            flat_idx = flat_idx.reshape(b, q * w)

            contrib_feat = weight.unsqueeze(-1) * feat.unsqueeze(2)  # (B,Q,W,out_channels)
            contrib_feat = contrib_feat.reshape(b, q * w, self.out_channels)
            contrib_occ = weight.reshape(b, q * w, 1)

            n = x_dim * y_dim * z_dim
            grid_feat = torch.zeros(b, n, self.out_channels, device=voxel_size.device, dtype=feat.dtype)
            grid_occ = torch.zeros(b, n, 1, device=voxel_size.device, dtype=feat.dtype)

            idx_feat = flat_idx.unsqueeze(-1).expand(-1, -1, self.out_channels)
            idx_occ = flat_idx.unsqueeze(-1)

            grid_feat = grid_feat.scatter_add(1, idx_feat, contrib_feat)
            grid_occ = grid_occ.scatter_add(1, idx_occ, contrib_occ)

            grid_feat = grid_feat.reshape(b, x_dim, y_dim, z_dim, self.out_channels).permute(0, 4, 1, 2, 3)
            grid_occ = grid_occ.reshape(b, x_dim, y_dim, z_dim, 1).permute(0, 4, 1, 2, 3)

            feats_out.append(grid_feat)
            occ_out.append(grid_occ)

        dyn_feats = torch.stack(feats_out, dim=1)  # (B,T_o,out_channels,X,Y,Z)
        dyn_occ = torch.stack(occ_out, dim=1)  # (B,T_o,1,X,Y,Z)
        return dyn_feats, dyn_occ
