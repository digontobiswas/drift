"""Instance-flow supervision loss.

Cam4DOcc flow ground truth is a *backward centroid-offset* field: every voxel
belonging to instance k at frame t stores `centroid_k(t-1) - own_index(t)`, in
units of latent voxels (0.8 m), with `255` marking a voxel that should not be
supervised (static/background voxels and out-of-instance voxels). See
docs/DESIGN_SPEC.md SS0 and SS3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["flow_loss"]

_IGNORE_VALUE = 255.0


def flow_loss(
    pred: Tensor,
    target: Tensor,
    ignore_value: float = _IGNORE_VALUE,
    beta: float = 1.0,
) -> Tensor:
    """Smooth-L1 loss between predicted and ground-truth instance flow.

    A voxel is included in the loss only if **all three** flow channels are
    `!= ignore_value` at that voxel (a partially-ignored channel still marks
    the whole voxel as ignore, since the GT is written atomically per-voxel by
    the rasterizer). If no voxel is valid (e.g. an empty scene with no dynamic
    instances) this returns an autograd-connected zero rather than the NaN
    that `0/0` would otherwise produce -- this path fires often in practice
    since most frames have long empty-flow stretches.

    Args:
        pred: Predicted flow `(B, T, 3, X, Y, Z)`, latent-voxel units.
        target: Ground-truth flow `(B, T, 3, X, Y, Z)`, latent-voxel units,
            `ignore_value` = ignore.
        ignore_value: Sentinel value marking an ignored channel.
        beta: Smooth-L1 transition point.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.shape[2] != 3:
        raise ValueError(f"expected 3 flow channels at dim=2, got shape {tuple(pred.shape)}")

    valid = (target != ignore_value).all(dim=2)  # (B,T,X,Y,Z)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        # NaN guard: keep the graph connected to `pred` with an exact-zero loss.
        return pred.sum() * 0.0

    valid_c = valid.unsqueeze(2).expand_as(pred)
    pred_valid = pred.masked_select(valid_c)
    target_valid = target.masked_select(valid_c)
    return F.smooth_l1_loss(pred_valid, target_valid, beta=beta, reduction="mean")
