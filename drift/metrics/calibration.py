"""Calibration metrics for the per-voxel uncertainty head: ECE + reliability curve. NOVEL.

Standard multiclass Expected Calibration Error (Guo et al., ICML 2017):
predictions are bucketed by their max-softmax confidence into `num_bins`
equal-width bins, and ECE is the confidence-weighted average gap between each
bin's mean confidence and its mean accuracy.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

__all__ = ["expected_calibration_error", "reliability_curve"]

_IGNORE_INDEX = 255


def _prepare(
    probs_or_logits: Tensor, labels: Tensor, from_logits: bool, ignore_index: int
) -> tuple[Tensor, Tensor]:
    if probs_or_logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"probs_or_logits {tuple(probs_or_logits.shape)} and labels {tuple(labels.shape)} "
            "must share every dim except probs_or_logits's trailing class dim"
        )
    C = probs_or_logits.shape[-1]
    probs = torch.softmax(probs_or_logits, dim=-1) if from_logits else probs_or_logits
    probs_flat = probs.reshape(-1, C)
    labels_flat = labels.reshape(-1)
    valid = labels_flat != ignore_index
    return probs_flat[valid], labels_flat[valid]


def expected_calibration_error(
    probs_or_logits: Tensor,
    labels: Tensor,
    num_bins: int = 15,
    from_logits: bool = False,
    ignore_index: int = _IGNORE_INDEX,
) -> Dict[str, Tensor]:
    """Expected Calibration Error over per-voxel class predictions.

    Args:
        probs_or_logits: `(..., num_classes)` predicted class probabilities
            (or logits, if `from_logits=True`).
        labels: `(...)` int64 ground-truth class labels, same leading shape.
        num_bins: Number of equal-width confidence bins in `[0, 1]`.
        from_logits: If `True`, applies softmax to `probs_or_logits` first.
        ignore_index: Label value excluded from the computation.

    Returns:
        Dict with `ece` (scalar), `bin_confidence`, `bin_accuracy`,
        `bin_count` (each `(num_bins,)`) and `bin_edges` (`(num_bins+1,)`).
        `ece` is `0.0` if there are no valid voxels.
    """
    probs, labels_v = _prepare(probs_or_logits, labels, from_logits, ignore_index)
    n = probs.shape[0]
    device = probs.device
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)

    if n == 0:
        zeros = torch.zeros(num_bins, device=device)
        return {
            "ece": torch.zeros((), device=device),
            "bin_confidence": zeros,
            "bin_accuracy": zeros.clone(),
            "bin_count": zeros.clone(),
            "bin_edges": bin_edges,
        }

    confidence, pred = probs.max(dim=-1)
    correct = (pred == labels_v).float()

    bin_conf = torch.zeros(num_bins, device=device)
    bin_acc = torch.zeros(num_bins, device=device)
    bin_count = torch.zeros(num_bins, device=device)
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        cnt = int(mask.sum().item())
        bin_count[i] = cnt
        if cnt > 0:
            bin_conf[i] = confidence[mask].mean()
            bin_acc[i] = correct[mask].mean()

    ece = (bin_count / n * (bin_acc - bin_conf).abs()).sum()
    return {"ece": ece, "bin_confidence": bin_conf, "bin_accuracy": bin_acc, "bin_count": bin_count, "bin_edges": bin_edges}


def reliability_curve(
    probs_or_logits: Tensor,
    labels: Tensor,
    num_bins: int = 15,
    from_logits: bool = False,
    ignore_index: int = _IGNORE_INDEX,
) -> Dict[str, Tensor]:
    """Reliability-diagram data (alias returning the same per-bin arrays as `expected_calibration_error`).

    Kept as a separate, explicitly-named entry point since "give me the curve
    to plot" and "give me the scalar ECE" are different call sites even
    though they share every intermediate computation.

    Args:
        See `expected_calibration_error`.

    Returns:
        Dict with `bin_confidence`, `bin_accuracy`, `bin_count`, `bin_edges`.
    """
    out = expected_calibration_error(probs_or_logits, labels, num_bins, from_logits, ignore_index)
    return {k: v for k, v in out.items() if k != "ece"}
