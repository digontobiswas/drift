"""Ego-motion utilities: pose vectorization and voxel-grid warping.

Conventions follow docs/DESIGN_SPEC.md SS1: `ego_motion[:, t]` is the 4x4
transform `T_{t+1<-t}` acting on points expressed in frame `t`'s LiDAR
coordinates. The present frame (index `T_p - 1` in the input sequence) is
never warped; all past frames are warped *into* the present frame, and future
frames are reached from the present frame by cumulative composition.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["mat2pose_vec", "cumulative_warp_to_present", "compose_future_transforms"]


def mat2pose_vec(mat: Tensor) -> Tensor:
    """Convert a 4x4 homogeneous transform to a 6-DoF pose vector.

    Args:
        mat: `(..., 4, 4)` homogeneous transforms.

    Returns:
        `(..., 6)` = `[tx, ty, tz, roll, pitch, yaw]`, translation in metres
        and ZYX-convention Euler angles in radians (roll about x, pitch about
        y, yaw about z; extrinsic ZYX, i.e. `R = Rz(yaw) @ Ry(pitch) @
        Rx(roll)`). This vector is used purely as an auxiliary conditioning
        signal (concatenated as extra channels in `Observer`) and is never
        inverted back to a matrix, so the exact Euler convention is not
        load-bearing -- it only needs to vary smoothly with the input pose,
        which it does away from the gimbal-lock singularity (handled below).
    """
    if mat.shape[-2:] != (4, 4):
        raise ValueError(f"mat2pose_vec expects (...,4,4), got shape {tuple(mat.shape)}")
    t = mat[..., :3, 3]
    R = mat[..., :3, :3]

    sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = sy < 1e-6

    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    pitch = torch.atan2(-R[..., 2, 0], sy.clamp_min(1e-12))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])

    roll_sing = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
    yaw_sing = torch.zeros_like(yaw)
    roll = torch.where(singular, roll_sing, roll)
    yaw = torch.where(singular, yaw_sing, yaw)

    rot_vec = torch.stack([roll, pitch, yaw], dim=-1)
    return torch.cat([t, rot_vec], dim=-1)


def _voxel_centers(pc_range: Sequence[float], size: int, axis: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """1D voxel-center coordinates along one axis of `pc_range`."""
    lo, hi = pc_range[axis], pc_range[axis + 3]
    voxel = (hi - lo) / size
    idx = torch.arange(size, device=device, dtype=dtype)
    return lo + (idx + 0.5) * voxel


def _normalize(coord: Tensor, lo: float, hi: float) -> Tensor:
    """Map a metric coordinate to grid_sample's `[-1, 1]` range (align_corners=False)."""
    return 2.0 * (coord - lo) / (hi - lo) - 1.0


def _warp_feature_volume(
    vol: Tensor,
    transform_out_from_in: Tensor,
    point_cloud_range: Sequence[float],
    align_corners: bool = False,
) -> Tensor:
    """Resample a `(B, C, X, Y, Z)` volume from its own frame into an output frame.

    For each voxel of the *output* grid (indexed the same way as `vol`, i.e.
    over the same fixed `point_cloud_range`/shape but interpreted in the
    output frame's coordinate system), find the corresponding physical
    location in the *input* frame -- `transform_out_from_in`'s inverse -- and
    bilinearly sample `vol` there.

    Args:
        vol: `(B, C, X, Y, Z)` features expressed in the input frame.
        transform_out_from_in: `(B, 4, 4)`, `T_{out <- in}`: maps a point
            expressed in the input frame's coordinates to the output frame's
            coordinates.
        point_cloud_range: `[x_min,y_min,z_min,x_max,y_max,z_max]`, shared by
            both frames (ego-centric grids re-centre on the ego at every
            timestep, so the same metric box applies).
        align_corners: Forwarded to `grid_sample`.

    Returns:
        `(B, C, X, Y, Z)` features resampled into the output frame; voxels
        that fall outside the input frame's grid are zero-filled.
    """
    B, C, X, Y, Z = vol.shape
    device, dtype = vol.device, vol.dtype

    xs = _voxel_centers(point_cloud_range, X, 0, device, dtype)
    ys = _voxel_centers(point_cloud_range, Y, 1, device, dtype)
    zs = _voxel_centers(point_cloud_range, Z, 2, device, dtype)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")  # each (X,Y,Z)
    ones = torch.ones_like(gx)
    pts_out = torch.stack([gx, gy, gz, ones], dim=-1)  # (X,Y,Z,4) homogeneous, in OUTPUT frame

    transform_in_from_out = torch.linalg.inv(transform_out_from_in)  # (B,4,4)
    pts_out_flat = pts_out.reshape(1, -1, 4).expand(B, -1, -1)  # (B, X*Y*Z, 4)
    pts_in = torch.bmm(pts_out_flat, transform_in_from_out.transpose(1, 2))  # (B, X*Y*Z, 4)
    pts_in = pts_in[..., :3].reshape(B, X, Y, Z, 3)

    nx = _normalize(pts_in[..., 0], point_cloud_range[0], point_cloud_range[3])
    ny = _normalize(pts_in[..., 1], point_cloud_range[1], point_cloud_range[4])
    nz = _normalize(pts_in[..., 2], point_cloud_range[2], point_cloud_range[5])

    # grid_sample on a 5D input (N,C,D,H,W) expects grid (N,D,H,W,3) with the last
    # dim ordered (x -> W, y -> H, z -> D). Our tensor layout is (B,C,X,Y,Z) with
    # X,Y,Z playing the role of (D,H,W) respectively, so the sample grid's last
    # dim must be ordered (z_norm, y_norm, x_norm) -- reversed from (x,y,z).
    grid = torch.stack([nz, ny, nx], dim=-1)  # (B,X,Y,Z,3)

    warped = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    return warped


def compose_future_transforms(ego_motion: Tensor, present_idx: int, num_future: int) -> Tensor:
    """Compose cumulative present->future ego transforms.

    Args:
        ego_motion: `(B, T_seq, 4, 4)`, `ego_motion[:, t] = T_{t+1<-t}`.
        present_idx: Index of the present frame within the raw frame timeline
            that `ego_motion` is defined over (`T_p - 1`).
        num_future: Number of future output frames `T_o` to produce.

    Returns:
        `future_ego`: `(B, num_future, 4, 4)` where `future_ego[:, k] =
        T_{present+k+1 <- present}`, i.e. index `k=0` is the transform to the
        first future frame. If `ego_motion` does not extend far enough to
        cover all `num_future` steps, the remaining steps use the identity
        (no further motion is known -- a documented, not silent, limitation).
    """
    if ego_motion.dim() != 4 or ego_motion.shape[-2:] != (4, 4):
        raise ValueError(f"ego_motion must be (B,T_seq,4,4), got shape {tuple(ego_motion.shape)}")
    if present_idx < 0:
        raise ValueError(f"present_idx must be >= 0, got {present_idx}")
    if num_future < 0:
        raise ValueError(f"num_future must be >= 0, got {num_future}")

    B, T_seq = ego_motion.shape[0], ego_motion.shape[1]
    device, dtype = ego_motion.device, ego_motion.dtype
    eye = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4)

    out: List[Tensor] = []
    cum = eye
    for k in range(num_future):
        idx = present_idx + k
        step = ego_motion[:, idx] if idx < T_seq else eye
        cum = torch.bmm(step, cum)
        out.append(cum)
    if num_future == 0:
        return torch.zeros(B, 0, 4, 4, device=device, dtype=dtype)
    return torch.stack(out, dim=1)


def cumulative_warp_to_present(
    feats: Tensor,
    ego_motion: Tensor,
    present_idx: int,
    point_cloud_range: Sequence[float],
    align_corners: bool = False,
) -> Tensor:
    """Warp past-frame latent volumes into the present frame's coordinates.

    Builds, for each past frame `t < present_idx`, the cumulative transform
    `T_{present<-t} = T_{present<-present-1} @ ... @ T_{t+1<-t}` from the
    per-step `ego_motion`, then resamples `feats[:, t]` through its inverse so
    that the output is expressed in present-frame coordinates. The present
    frame itself (`t == present_idx`) passes through unchanged.

    Args:
        feats: `(B, T_p, C, X, Y, Z)`, each timestep's features in its own
            frame's coordinates.
        ego_motion: `(B, T_seq, 4, 4)`, `ego_motion[:, t] = T_{t+1<-t}`. Only
            entries `[0, present_idx)` are used.
        present_idx: Index of the present frame within `feats`'s time
            dimension (`T_p - 1`).
        point_cloud_range: `[x_min,y_min,z_min,x_max,y_max,z_max]`.
        align_corners: Forwarded to `grid_sample`.

    Returns:
        `(B, T_p, C, X, Y, Z)`, all timesteps expressed in present-frame
        coordinates.
    """
    if feats.dim() != 6:
        raise ValueError(f"feats must be (B,T_p,C,X,Y,Z), got shape {tuple(feats.shape)}")
    B, T_p = feats.shape[0], feats.shape[1]
    if present_idx < 0 or present_idx >= T_p:
        raise ValueError(f"present_idx={present_idx} out of range for T_p={T_p}")
    if ego_motion.shape[0] != B:
        raise ValueError(f"ego_motion batch {ego_motion.shape[0]} != feats batch {B}")
    if ego_motion.shape[1] < present_idx:
        raise ValueError(
            f"ego_motion has {ego_motion.shape[1]} steps, need at least {present_idx} to reach the present frame"
        )

    device, dtype = feats.device, feats.dtype
    eye = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4)

    cum_present_from_t = [None] * (present_idx + 1)
    cum_present_from_t[present_idx] = eye
    cum = eye
    for t in range(present_idx - 1, -1, -1):
        # T_{present<-t} = T_{present<-t+1} @ T_{t+1<-t}
        cum = torch.bmm(cum, ego_motion[:, t])
        cum_present_from_t[t] = cum

    out: List[Tensor] = []
    for t in range(T_p):
        if t == present_idx:
            out.append(feats[:, t])
        elif t < present_idx:
            out.append(_warp_feature_volume(feats[:, t], cum_present_from_t[t], point_cloud_range, align_corners))
        else:
            raise ValueError(
                f"cumulative_warp_to_present only handles past/present frames (t <= present_idx={present_idx}), "
                f"but feats has T_p={T_p} frames including index t={t}"
            )
    return torch.stack(out, dim=1)
