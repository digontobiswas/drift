"""Heteroscedastic uncertainty loss for the per-voxel uncertainty head. NOVEL.

Follows the standard heteroscedastic-attenuation trick (Kendall & Gal, NeurIPS
2017): the network predicts a per-voxel log-variance `s`, and the base
per-voxel loss `L` is reweighted as `exp(-s) * L + 0.5 * s`. The `0.5 * s` term
is what prevents the well-known failure mode where the network simply drives
`s -> -inf` (or `+inf`) to trivially zero out the attenuated loss without
actually improving `L` -- omitting it (as a naive implementation might) lets
the loss "collapse".
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["uncertainty_nll"]

_IGNORE_INDEX = 255


def uncertainty_nll(
    logits: Tensor,
    log_var: Tensor,
    target: Tensor,
    ignore_index: int = _IGNORE_INDEX,
    log_var_clamp: Tuple[float, float] = (-10.0, 10.0),
) -> Tensor:
    """Heteroscedastic negative log-likelihood over per-voxel occupancy classes.

    `per_voxel_loss = exp(-log_var) * CE(logits, target) + 0.5 * log_var`,
    averaged over valid (non-ignored) voxels. `log_var` is clamped before use
    purely for numerical stability (`exp(-log_var)` would otherwise overflow);
    the clamp bounds are wide enough not to affect a well-behaved fit.

    Args:
        logits: Occupancy class logits `(B, T, num_classes, X, Y, Z)`.
        log_var: Raw (unbounded) predicted log-variance `(B, T, 1, X, Y, Z)`.
        target: Integer labels `(B, T, X, Y, Z)`, `ignore_index` = ignore.
        ignore_index: Label value excluded from the loss.
        log_var_clamp: `(min, max)` clamp applied to `log_var` before use.

    Returns:
        Scalar loss tensor; autograd-connected zero if nothing is valid.
    """
    if logits.dim() != 6:
        raise ValueError(f"expected logits (B,T,C,X,Y,Z), got shape {tuple(logits.shape)}")
    if log_var.shape[2] != 1:
        raise ValueError(f"expected log_var with a singleton class dim (B,T,1,X,Y,Z), got {tuple(log_var.shape)}")
    B, T, C = logits.shape[:3]
    spatial = logits.shape[3:]
    if log_var.shape != (B, T, 1) + tuple(spatial):
        raise ValueError(
            f"log_var shape {tuple(log_var.shape)} does not match logits spatial layout "
            f"{(B, T, 1) + tuple(spatial)}"
        )
    if target.shape != (B, T) + tuple(spatial):
        raise ValueError(f"target shape {tuple(target.shape)} does not match logits {(B, T) + tuple(spatial)}")

    valid = target != ignore_index
    n_valid = valid.sum()
    if int(n_valid.item()) == 0:
        return logits.sum() * 0.0 + log_var.sum() * 0.0

    ce = F.cross_entropy(
        logits.reshape(B * T, C, *spatial),
        target.reshape(B * T, *spatial),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(B, T, *spatial)

    log_var_c = log_var.squeeze(2).clamp(*log_var_clamp)
    precision = torch.exp(-log_var_c)
    per_voxel = precision * ce + 0.5 * log_var_c

    valid_f = valid.to(per_voxel.dtype)
    return (per_voxel * valid_f).sum() / n_valid.to(per_voxel.dtype)
