"""Instance-flow evaluation metrics: EPE, angular error, magnitude error.

**Units.** Flow is predicted and stored (ground truth) in *latent voxel*
units (0.8 m per voxel, see docs/DESIGN_SPEC.md SS1/SS0). Every metric in this
file reports its distance-valued outputs in **metres**, by multiplying the
raw latent-voxel values by `voxel_size` (default `0.8`) *before* computing any
error. Do not feed already-metric-scaled flow into these functions with a
non-1.0 `voxel_size`, or the conversion will double-apply.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

__all__ = ["flow_epe", "FlowEPEMetric"]

_IGNORE_VALUE = 255.0
_LATENT_VOXEL_METRES = 0.8


def flow_epe(
    pred: Tensor,
    target: Tensor,
    voxel_size: float = _LATENT_VOXEL_METRES,
    ignore_value: float = _IGNORE_VALUE,
) -> Dict[str, Tensor]:
    """End-point error, angular error, and magnitude error between predicted and GT flow.

    Args:
        pred: Predicted flow `(..., 3, X, Y, Z)`, latent-voxel units (channel
            dim at position `-4`).
        target: Ground-truth flow, same shape, latent-voxel units,
            `ignore_value` = ignore.
        voxel_size: Metres per latent voxel; both `pred` and `target` are
            multiplied by this before any error is computed, so all returned
            distances are in metres.
        ignore_value: Sentinel marking an ignored channel; a voxel is scored
            only if all 3 channels are `!= ignore_value`.

    Returns:
        Dict with `epe` (mean L2 error, metres), `angular_error` (mean angle
        between predicted/GT vectors, radians), `magnitude_error` (mean
        absolute difference of vector norms, metres), and `n_valid` (voxel
        count scored). All are autograd-connected zeros (not NaN) if no voxel
        is valid.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.shape[-4] != 3:
        raise ValueError(f"expected 3 flow channels at dim=-4, got shape {tuple(pred.shape)}")

    valid = (target != ignore_value).all(dim=-4)
    n_valid = valid.sum()
    if int(n_valid.item()) == 0:
        zero = pred.sum() * 0.0
        return {
            "epe": zero,
            "angular_error": zero,
            "magnitude_error": zero,
            "n_valid": torch.zeros((), dtype=torch.long, device=pred.device),
        }

    pred_m = pred * voxel_size
    target_m = target * voxel_size

    diff = pred_m - target_m
    l2 = diff.norm(dim=-4)
    epe = l2[valid].mean()

    pred_norm = pred_m.norm(dim=-4)
    target_norm = target_m.norm(dim=-4)
    magnitude_error = (pred_norm - target_norm).abs()[valid].mean()

    eps = 1e-6
    cos = (pred_m * target_m).sum(dim=-4) / (pred_norm.clamp_min(eps) * target_norm.clamp_min(eps))
    cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angular_error = torch.acos(cos)[valid].mean()

    return {"epe": epe, "angular_error": angular_error, "magnitude_error": magnitude_error, "n_valid": n_valid}


class FlowEPEMetric:
    """Accumulates flow EPE/angular/magnitude error across batches (pooled, not averaged-of-batches).

    Mirrors `OccupancyIoUMetric`'s pooling philosophy: each `update()` call's
    per-voxel errors and valid-voxel count are summed, and `compute()` divides
    the running sums once at the end -- this is a voxel-weighted (micro)
    average, not a mean of per-batch means (which would over-weight batches
    with few valid voxels).

    Args:
        voxel_size: Metres per latent voxel.
        ignore_value: Sentinel marking an ignored channel.
    """

    def __init__(self, voxel_size: float = _LATENT_VOXEL_METRES, ignore_value: float = _IGNORE_VALUE) -> None:
        self.voxel_size = voxel_size
        self.ignore_value = ignore_value
        self.reset()

    def reset(self) -> None:
        """Zero every running sum."""
        self._sum_epe = 0.0
        self._sum_ang = 0.0
        self._sum_mag = 0.0
        self._n_valid = 0

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor) -> None:
        """Accumulate one batch's flow error sums.

        Args:
            pred: Predicted flow `(..., 3, X, Y, Z)`, latent-voxel units.
            target: Ground-truth flow, same shape.
        """
        out = flow_epe(pred, target, voxel_size=self.voxel_size, ignore_value=self.ignore_value)
        n = int(out["n_valid"].item())
        if n == 0:
            return
        self._sum_epe += float(out["epe"].item()) * n
        self._sum_ang += float(out["angular_error"].item()) * n
        self._sum_mag += float(out["magnitude_error"].item()) * n
        self._n_valid += n

    def compute(self) -> Dict[str, float]:
        """Return pooled mean EPE / angular error / magnitude error over every `update()` call so far."""
        if self._n_valid == 0:
            return {"epe": 0.0, "angular_error": 0.0, "magnitude_error": 0.0, "n_valid": 0}
        return {
            "epe": self._sum_epe / self._n_valid,
            "angular_error": self._sum_ang / self._n_valid,
            "magnitude_error": self._sum_mag / self._n_valid,
            "n_valid": self._n_valid,
        }
