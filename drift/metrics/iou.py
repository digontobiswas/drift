"""Pooled semantic-occupancy IoU metric, numerically matching the Cam4DOcc/OccProphet protocol.

The defining property (docs/DESIGN_SPEC.md SS5) is that a *single* confusion
matrix is accumulated across every frame and every sample, and IoU is computed
once from that pooled matrix -- a micro-average over voxels, macro over
classes -- which is **not** the same number as averaging per-sample or
per-frame IoUs. Logits are trilinear-upsampled from latent to full resolution
and only then argmax'd (never argmax first, then upsampled labels), since
upsampling one-hot/argmax labels would introduce interpolation artifacts a
raw-logit upsample does not have.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["fast_hist", "cm_to_ious", "OccupancyIoUMetric"]

ArrayLike = Union[Tensor, np.ndarray]


def _as_long_tensor(x: ArrayLike) -> Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.reshape(-1).long()


def fast_hist(pred: ArrayLike, label: ArrayLike, num_cls: int) -> Tensor:
    """Pooled confusion matrix via a single bincount.

    `hist[label, pred] = bincount(num_cls * label + pred)`. Entries with
    `label` outside `[0, num_cls)` (e.g. the `255` ignore sentinel) are
    excluded from the count.

    Args:
        pred: Predicted class indices, any shape (flattened internally).
        label: Ground-truth class indices, same shape as `pred`.
        num_cls: Number of classes.

    Returns:
        `(num_cls, num_cls)` int64 tensor, `hist[i, j]` = count of voxels with
        ground truth `i` and prediction `j`.
    """
    pred_t = _as_long_tensor(pred)
    label_t = _as_long_tensor(label)
    if pred_t.shape != label_t.shape:
        raise ValueError(f"fast_hist: pred/label size mismatch: {pred_t.shape} vs {label_t.shape}")
    valid = (label_t >= 0) & (label_t < num_cls) & (pred_t >= 0) & (pred_t < num_cls)
    label_v = label_t[valid]
    pred_v = pred_t[valid]
    idx = num_cls * label_v + pred_v
    hist = torch.bincount(idx, minlength=num_cls * num_cls)
    return hist.reshape(num_cls, num_cls)


def cm_to_ious(hist: ArrayLike, eps: float = 1e-6) -> Tensor:
    """Per-class IoU from a pooled confusion matrix.

    `IoU_c = hist[c,c] / (row_sum[c] + col_sum[c] - hist[c,c])`.

    Args:
        hist: `(num_cls, num_cls)` confusion matrix (rows = ground truth,
            columns = prediction), as produced by `fast_hist`.
        eps: Added to the denominator; a class with zero support in both
            prediction and ground truth reports IoU `0` (not `NaN`) under
            this convention.

    Returns:
        `(num_cls,)` float tensor of per-class IoU.
    """
    if isinstance(hist, np.ndarray):
        hist = torch.from_numpy(hist)
    hist = hist.float()
    diag = torch.diagonal(hist)
    denom = hist.sum(dim=1) + hist.sum(dim=0) - diag
    return diag / (denom + eps)


class OccupancyIoUMetric:
    """Accumulates one pooled confusion matrix per bucket; computes IoU once at the end.

    Bucket semantics (docs/DESIGN_SPEC.md SS5):
      - "present" bucket: pools only the model's present-frame reconstruction
        (output index `present_index`, default `0` -- the model's `T_o`
        output frames are `[present_reconstruction, future_1, ..., future_{T_o-1}]`,
        see the "Interface concerns" note in the implementation report).
      - "future" bucket: pools every non-present output frame.
      - per-horizon cumulative buckets: bucket `k` (`k = 1 .. T_o-1`) pools
        every future frame up to and including horizon `k`, so `IoU` at
        bucket `k` reflects accumulated error through that horizon, not the
        error of frame `k` in isolation.

    `IOU_mean` excludes class `0` (free) from the per-class average, per spec.

    Args:
        num_classes: Number of occupancy classes (including free = class 0).
        num_future: `T_o`, the number of output frames each `update` call
            provides.
        upsample_size: Full-resolution `(X, Y, Z)` grid to upsample logits to
            before argmax (default `(512, 512, 40)`).
        present_index: Index within the `T_o` dimension treated as the
            "current occupancy" reconstruction for `IoU_c`.
    """

    def __init__(
        self,
        num_classes: int,
        num_future: int,
        upsample_size: Tuple[int, int, int] = (512, 512, 40),
        present_index: int = 0,
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if num_future < 1:
            raise ValueError(f"num_future must be >= 1, got {num_future}")
        if not (0 <= present_index < num_future):
            raise ValueError(f"present_index={present_index} out of range for num_future={num_future}")
        self.num_classes = num_classes
        self.num_future = num_future
        self.upsample_size = tuple(upsample_size)
        self.present_index = present_index
        self.reset()

    def reset(self) -> None:
        """Zero every accumulated confusion matrix."""
        nc = self.num_classes
        self._hist_present = torch.zeros(nc, nc, dtype=torch.int64)
        self._hist_future_total = torch.zeros(nc, nc, dtype=torch.int64)
        n_future_steps = max(self.num_future - 1, 0)
        self._hist_cumulative: List[Tensor] = [torch.zeros(nc, nc, dtype=torch.int64) for _ in range(n_future_steps)]

    @torch.no_grad()
    def update(self, pred_logits: Tensor, gt_labels: Tensor) -> None:
        """Accumulate one batch's confusion matrices into every bucket.

        Args:
            pred_logits: `(B, T_o, num_classes, x, y, z)` raw logits at any
                spatial resolution (typically the latent grid).
            gt_labels: `(B, T_o, 512, 512, 40)` (or `upsample_size`) int64
                ground truth; `255` (or any value `>= num_classes`) is
                treated as ignore.
        """
        if pred_logits.dim() != 6:
            raise ValueError(f"pred_logits must be (B,T_o,C,x,y,z), got shape {tuple(pred_logits.shape)}")
        B, T_o, C = pred_logits.shape[:3]
        if T_o != self.num_future:
            raise ValueError(f"update() got T_o={T_o}, but this metric was built with num_future={self.num_future}")
        if C != self.num_classes:
            raise ValueError(f"update() got {C} classes, but this metric was built with num_classes={self.num_classes}")
        if gt_labels.shape[:2] != (B, T_o) or tuple(gt_labels.shape[2:]) != self.upsample_size:
            raise ValueError(
                f"gt_labels shape {tuple(gt_labels.shape)} must be (B={B},T_o={T_o},*upsample_size="
                f"{self.upsample_size})"
            )

        flat_logits = pred_logits.reshape(B * T_o, C, *pred_logits.shape[3:]).float()
        up = F.interpolate(flat_logits, size=self.upsample_size, mode="trilinear", align_corners=False)
        pred_labels = up.argmax(dim=1).reshape(B, T_o, *self.upsample_size)

        hists = [fast_hist(pred_labels[:, t], gt_labels[:, t], self.num_classes) for t in range(T_o)]

        # `fast_hist` produces its result on whatever device `pred`/`label` were on --
        # i.e. wherever the caller is running eval, typically CUDA. The accumulators
        # were made in `reset()` with no device (CPU, by torch's default), so the
        # in-place `+=` below would raise "Expected all tensors to be on the same
        # device" the first time `update` is ever called under CUDA. Moved here
        # rather than in `reset()` because the metric is constructed before the
        # caller has necessarily chosen a device.
        device = hists[0].device
        if self._hist_present.device != device:
            self._hist_present = self._hist_present.to(device)
            self._hist_future_total = self._hist_future_total.to(device)
            self._hist_cumulative = [h.to(device) for h in self._hist_cumulative]

        cum = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=device)
        future_k = 0
        for t in range(T_o):
            if t == self.present_index:
                self._hist_present += hists[t]
            else:
                self._hist_future_total += hists[t]
                cum = cum + hists[t]
                self._hist_cumulative[future_k] += cum
                future_k += 1

    def compute(self) -> Dict[str, Any]:
        """Compute IoU from the accumulated pooled confusion matrices.

        Returns:
            Dict with:
              `IoU_c` (float): present-frame mean IoU, classes `1:` averaged.
              `IoU_f` (float): all-future-frames-pooled mean IoU, classes `1:`.
              `per_class_present`, `per_class_future`: `(num_classes,)` tensors.
              `per_horizon_IoU`: length `T_o-1` list of mean IoU (classes `1:`),
                cumulative through each future horizon.
        """
        ious_present = cm_to_ious(self._hist_present)
        ious_future = cm_to_ious(self._hist_future_total)
        per_horizon = [cm_to_ious(h) for h in self._hist_cumulative]
        return {
            "IoU_c": float(ious_present[1:].mean().item()) if self.num_classes > 1 else float(ious_present[0].item()),
            "IoU_f": float(ious_future[1:].mean().item()) if self.num_classes > 1 else float(ious_future[0].item()),
            "per_class_present": ious_present,
            "per_class_future": ious_future,
            "per_horizon_IoU": [float(h[1:].mean().item()) if self.num_classes > 1 else float(h[0].item()) for h in per_horizon],
            "hist_present": self._hist_present.clone(),
            "hist_future": self._hist_future_total.clone(),
        }
