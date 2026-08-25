"""Observer: aggregate multi-frame fused latents into a spacetime representation.

REUSED from OccProphet. Origin: OccProphet (ICLR 2025).

Wraps :class:`~drift.models.e4a.EfficientAggregation4D` with an ego-motion conditioning step:
each past frame's relative ego transform is converted to a 6-DoF pose vector, broadcast
spatially, and concatenated onto the fused latent as 6 extra channels before running E4A.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from drift.models.e4a import EfficientAggregation4D


def _mat2pose_vec(mat: Tensor, eps: float = 1e-7) -> Tensor:
    """Convert a batch of 4x4 rigid transforms to 6-DoF vectors [tx,ty,tz,rx,ry,rz].

    The rotation is represented as an axis-angle vector (`angle * unit_axis`), extracted from
    the rotation matrix via the standard Rodrigues formula. This is a local, self-contained
    reimplementation kept in-file rather than imported from `drift.data.ego_motion` (see
    interface note in this repo's implementation report): both should ultimately share one
    definition once that module lands.

    Args:
        mat: (..., 4, 4) rigid transforms.
        eps: numerical safety margin for the near-zero-rotation case.

    Returns:
        (..., 6) tensor of [tx, ty, tz, rx, ry, rz].
    """
    if mat.shape[-2:] != (4, 4):
        raise ValueError(f"_mat2pose_vec expects (...,4,4) input, got {tuple(mat.shape)}.")
    trans = mat[..., :3, 3]
    rot = mat[..., :3, :3]

    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    axis = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta).clamp_min(eps)
    axis = axis / (2.0 * sin_theta).unsqueeze(-1)

    # near-zero rotation -> axis-angle vector is ~0 regardless of (undefined) axis direction
    small = (theta.abs() < 1e-4).unsqueeze(-1)
    rot_vec = torch.where(small, torch.zeros_like(axis), axis * theta.unsqueeze(-1))

    return torch.cat([trans, rot_vec], dim=-1)


class Observer(nn.Module):
    """Aggregate multi-frame fused latents into a spacetime representation. REUSED (OccProphet).

    Args:
        in_channels: Channel count of `fused` (before any ego-motion channels are appended).
        embed_dims: Output channel width.
        timesteps: Expected `T_p`; forwarded to the internal E4A as a hint only.
        with_ego_channels: If True (default), append 6 ego-pose channels before running E4A,
            so the internal E4A sees `in_channels + 6` channels.
        **e4a_kwargs: Forwarded to `EfficientAggregation4D` (e.g. `downsample_layers`).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        timesteps: int,
        with_ego_channels: bool = True,
        **e4a_kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.with_ego_channels = with_ego_channels

        if "multiscale_output" in e4a_kwargs and e4a_kwargs["multiscale_output"]:
            raise ValueError(
                "Observer requires a single-tensor E4A output; pass multiscale_output=False "
                "(or omit it) in e4a_kwargs."
            )
        e4a_kwargs = dict(e4a_kwargs)
        e4a_kwargs["multiscale_output"] = False

        e4a_in_channels = in_channels + 6 if with_ego_channels else in_channels
        self.e4a = EfficientAggregation4D(
            in_channels=e4a_in_channels, embed_dims=embed_dims, timesteps=timesteps,
            **e4a_kwargs,
        )

    def forward(self, fused: Tensor, ego_motion: Tensor) -> Tensor:
        """
        Args:
            fused: (B, T_p, C, X, Y, Z), already warped to the present frame. C == in_channels.
            ego_motion: (B, T_p, 4, 4). Entry `t` is `T_{t+1<-t}`; the last entry (present ->
                next/future) is not used here.

        Returns:
            O_obs: (B, T_p, embed_dims, X, Y, Z)
        """
        if fused.dim() != 6:
            raise ValueError(
                f"Observer.forward expects fused with shape (B,T_p,C,X,Y,Z), got "
                f"{tuple(fused.shape)}."
            )
        b, t_p, c, x_dim, y_dim, z_dim = fused.shape
        if c != self.in_channels:
            raise ValueError(
                f"Observer: fused has {c} channels but in_channels={self.in_channels}."
            )

        if self.with_ego_channels:
            if ego_motion.shape[:2] != (b, t_p):
                raise ValueError(
                    f"Observer: ego_motion batch/time ({tuple(ego_motion.shape[:2])}) does not "
                    f"match fused ({(b, t_p)})."
                )
            if t_p >= 2:
                pose_vecs = _mat2pose_vec(ego_motion[:, : t_p - 1])  # (B, T_p-1, 6)
            else:
                pose_vecs = ego_motion.new_zeros(b, 0, 6)
            zero = pose_vecs.new_zeros(b, 1, 6)
            pose_vecs = torch.cat([zero, pose_vecs], dim=1)  # (B, T_p, 6)

            pose_map = pose_vecs.reshape(b, t_p, 6, 1, 1, 1).expand(
                b, t_p, 6, x_dim, y_dim, z_dim
            )
            x = torch.cat([fused, pose_map], dim=2)
        else:
            x = fused

        return self.e4a(x)
