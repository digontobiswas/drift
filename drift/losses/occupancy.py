"""Occupancy (semantic scene completion) losses.

Combines four complementary terms over the latent-resolution occupancy prediction:
cross-entropy, Lovasz-softmax, geometric scene-class affinity ("geo_scal"), and
semantic scene-class affinity ("sem_scal"). The geo/sem-scal formulations follow
MonoScene (Cao & de Charette, CVPR 2022); the combination policy (all four terms
live and independently config-weighted) is a deliberate departure from the
OccProphet public release, which silently runs CE only with a hard-coded 0.5
weight on the others -- see docs/DESIGN_SPEC.md SS3.

All functions in this file operate at whatever spatial resolution `pred` and
`target` are given at (in DRIFT this is always the latent grid, 128x128x10);
`downsample_target` is provided separately to produce a latent-resolution
target from the full-resolution (512x512x40) ground truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "occupancy_loss",
    "downsample_target",
    "lovasz_softmax",
    "geo_scal_loss",
    "sem_scal_loss",
    "CE_ssc_loss",
]

_IGNORE_INDEX = 255


def _flatten_logits_and_target(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten `(B, T, C, X, Y, Z)` logits and `(B, T, X, Y, Z)` labels to `(N, C)` / `(N,)`.

    Args:
        pred: Logits of shape `(B, T, C, X, Y, Z)` (or any shape with the class
            dimension at position 2 followed by spatial dims).
        target: Integer labels of shape `(B, T, X, Y, Z)`, same leading/trailing
            dims as `pred` minus the class dimension.

    Returns:
        Tuple of `(logits_flat, target_flat)` with shapes `(N, C)` and `(N,)`.
    """
    if pred.dim() < 3:
        raise ValueError(f"pred must have at least 3 dims (B,C,...), got shape {tuple(pred.shape)}")
    if pred.shape[0] != target.shape[0] or pred.shape[1] != target.shape[1]:
        raise ValueError(
            f"pred/target leading (B,T) dims must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    num_classes = pred.shape[2]
    if pred.shape[3:] != target.shape[2:]:
        raise ValueError(
            f"pred spatial dims {tuple(pred.shape[3:])} must match target spatial dims "
            f"{tuple(target.shape[2:])}"
        )
    # (B,T,C,X,Y,Z) -> (B,T,X,Y,Z,C) -> (N,C)
    perm = [0, 1] + list(range(3, pred.dim())) + [2]
    logits_flat = pred.permute(*perm).reshape(-1, num_classes)
    target_flat = target.reshape(-1)
    return logits_flat, target_flat


def CE_ssc_loss(
    pred: Tensor,
    target: Tensor,
    class_weights: Optional[Tensor] = None,
    ignore_index: int = _IGNORE_INDEX,
) -> Tensor:
    """Weighted cross-entropy over per-voxel occupancy classes.

    Args:
        pred: Logits `(B, T, num_classes, X, Y, Z)`.
        target: Integer labels `(B, T, X, Y, Z)`, `ignore_index` = ignore.
        class_weights: Optional `(num_classes,)` per-class weight tensor.
        ignore_index: Label value excluded from the loss.

    Returns:
        Scalar loss tensor. Returns an autograd-connected zero if every voxel
        is ignored (avoids the `F.cross_entropy` NaN from an empty valid set).
    """
    logits_flat, target_flat = _flatten_logits_and_target(pred, target)
    valid = target_flat != ignore_index
    if not bool(valid.any()):
        return pred.sum() * 0.0
    weight = class_weights.to(dtype=logits_flat.dtype, device=logits_flat.device) if class_weights is not None else None
    return F.cross_entropy(logits_flat, target_flat, weight=weight, ignore_index=ignore_index, reduction="mean")


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Gradient of the Lovasz extension w.r.t sorted binary errors (Berman et al., CVPR 2018)."""
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-12)
    if p > 1:
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
    return jaccard


def _lovasz_softmax_flat(probas: Tensor, labels: Tensor, classes: str = "present") -> Tensor:
    """Multi-class Lovasz-Softmax loss on already-flattened, ignore-filtered inputs.

    Args:
        probas: `(N, C)` class probabilities (post-softmax).
        labels: `(N,)` integer ground-truth labels.
        classes: `"present"` averages only over classes that occur in `labels`;
            `"all"` averages over every class.

    Returns:
        Scalar loss tensor.
    """
    if probas.numel() == 0:
        return probas.sum() * 0.0
    num_classes = probas.shape[1]
    losses: List[Tensor] = []
    for c in range(num_classes):
        fg = (labels == c).to(probas.dtype)
        if classes == "present" and float(fg.sum()) == 0.0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    pred: Tensor,
    target: Tensor,
    classes: str = "present",
    ignore_index: int = _IGNORE_INDEX,
) -> Tensor:
    """Lovasz-Softmax loss (Berman et al., CVPR 2018), a direct surrogate for mean IoU.

    Args:
        pred: Logits `(B, T, num_classes, X, Y, Z)`.
        target: Integer labels `(B, T, X, Y, Z)`, `ignore_index` = ignore.
        classes: See `_lovasz_softmax_flat`.
        ignore_index: Label value excluded from the loss.

    Returns:
        Scalar loss tensor; autograd-connected zero if nothing is valid.
    """
    logits_flat, target_flat = _flatten_logits_and_target(pred, target)
    valid = target_flat != ignore_index
    if not bool(valid.any()):
        return pred.sum() * 0.0
    probas = F.softmax(logits_flat[valid], dim=1)
    return _lovasz_softmax_flat(probas, target_flat[valid], classes=classes)


def geo_scal_loss(pred: Tensor, target: Tensor, ignore_index: int = _IGNORE_INDEX) -> Tensor:
    """Geometric scene-class affinity loss (binary occupied-vs-free precision/recall/specificity).

    Follows MonoScene (Cao & de Charette, CVPR 2022), Eq. geo_scal.

    Args:
        pred: Logits `(B, T, num_classes, X, Y, Z)`. Class 0 is "free".
        target: Integer labels `(B, T, X, Y, Z)`, `ignore_index` = ignore.
        ignore_index: Label value excluded from the loss.

    Returns:
        Scalar loss tensor; autograd-connected zero if nothing is valid.
    """
    logits_flat, target_flat = _flatten_logits_and_target(pred, target)
    valid = target_flat != ignore_index
    if not bool(valid.any()):
        return pred.sum() * 0.0
    probs = F.softmax(logits_flat[valid], dim=1)
    empty_probs = probs[:, 0]
    nonempty_probs = 1.0 - empty_probs
    nonempty_target = (target_flat[valid] != 0).to(probs.dtype)

    eps = 1e-6
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum().clamp_min(eps)
    recall = intersection / nonempty_target.sum().clamp_min(eps)
    n_empty_target = (1.0 - nonempty_target).sum()
    specificity = ((1.0 - nonempty_target) * empty_probs).sum() / n_empty_target.clamp_min(eps)

    loss = (
        -torch.log(precision.clamp_min(eps))
        - torch.log(recall.clamp_min(eps))
        - torch.log(specificity.clamp_min(eps))
    )
    return loss


def sem_scal_loss(pred: Tensor, target: Tensor, ignore_index: int = _IGNORE_INDEX) -> Tensor:
    """Semantic scene-class affinity loss (per-class precision/recall/specificity).

    Follows MonoScene (Cao & de Charette, CVPR 2022), Eq. sem_scal.

    Args:
        pred: Logits `(B, T, num_classes, X, Y, Z)`.
        target: Integer labels `(B, T, X, Y, Z)`, `ignore_index` = ignore.
        ignore_index: Label value excluded from the loss.

    Returns:
        Scalar loss tensor; autograd-connected zero if nothing is valid or no
        class has any positive support in this batch.
    """
    logits_flat, target_flat = _flatten_logits_and_target(pred, target)
    valid = target_flat != ignore_index
    if not bool(valid.any()):
        return pred.sum() * 0.0
    probs = F.softmax(logits_flat[valid], dim=1)
    labels = target_flat[valid]
    num_classes = probs.shape[1]

    eps = 1e-6
    losses: List[Tensor] = []
    for c in range(num_classes):
        p = probs[:, c]
        completion_target = (labels == c).to(probs.dtype)
        n_pos = completion_target.sum()
        if float(n_pos) == 0.0:
            continue
        nominator = (p * completion_target).sum()
        loss_class = probs.sum() * 0.0
        p_sum = p.sum()
        if float(p_sum.detach()) > 0.0:
            precision = nominator / p_sum.clamp_min(eps)
            loss_class = loss_class - torch.log(precision.clamp_min(eps))
        recall = nominator / n_pos.clamp_min(eps)
        loss_class = loss_class - torch.log(recall.clamp_min(eps))
        n_neg = (1.0 - completion_target).sum()
        if float(n_neg) > 0.0:
            specificity = ((1.0 - p) * (1.0 - completion_target)).sum() / n_neg.clamp_min(eps)
            loss_class = loss_class - torch.log(specificity.clamp_min(eps))
        losses.append(loss_class)
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def downsample_target(target: Tensor, ratio: int = 4) -> Tensor:
    """Downsample a full-resolution occupancy target by masked majority vote.

    Splits the trailing three spatial dims into non-overlapping `ratio^3` blocks.
    Per block:
      - if every voxel in the block is empty (label 0), the block label is `0`.
      - otherwise, let the "majority" be the most frequent non-zero label among
        the block's voxels. If that label's count is a strict majority (more
        than half) of the block's voxels, the block takes that label.
      - otherwise (non-empty, but no strict majority among all label values in
        the block -- e.g. an even split between two object classes) the block
        is `255` (ignore).

    Args:
        target: Integer labels `(..., X, Y, Z)` with `X, Y, Z` each divisible by
            `ratio`. Leading dims (e.g. `(B, T)`) are arbitrary.
        ratio: Block edge length (e.g. `4` for 512 -> 128).

    Returns:
        Integer labels `(..., X/ratio, Y/ratio, Z/ratio)` as `int64`, ready to use
        directly as a `cross_entropy` target. The output is at latent resolution and
        therefore small, so it is always widened to `int64` rather than echoing
        `target`'s dtype -- callers pass a compact `uint8` full-resolution grid (see
        `Cam4DOccDataset`), and `F.cross_entropy` requires a `Long` target.
    """
    if target.dim() < 3:
        raise ValueError(f"target must have at least 3 dims (...,X,Y,Z), got shape {tuple(target.shape)}")
    *lead, X, Y, Z = target.shape
    if X % ratio != 0 or Y % ratio != 0 or Z % ratio != 0:
        raise ValueError(
            f"spatial dims {(X, Y, Z)} must all be divisible by ratio={ratio}, got shape {tuple(target.shape)}"
        )
    Xl, Yl, Zl = X // ratio, Y // ratio, Z // ratio
    device = target.device

    num_classes = int(target.max().item()) + 1 if target.numel() > 0 else 1
    num_classes = max(num_classes, 1)

    # (*, X, Y, Z) -> (*, Xl, r, Yl, r, Zl, r) -> (*, Xl, Yl, Zl, r, r, r) -> (M, K)
    blocks = target.reshape(*lead, Xl, ratio, Yl, ratio, Zl, ratio)
    n_lead = len(lead)
    perm = list(range(n_lead)) + [
        n_lead + 0,  # Xl
        n_lead + 2,  # Yl
        n_lead + 4,  # Zl
        n_lead + 1,  # r (x)
        n_lead + 3,  # r (y)
        n_lead + 5,  # r (z)
    ]
    blocks = blocks.permute(*perm).contiguous()
    K = ratio * ratio * ratio
    # Deliberately NOT widened to int64 here: at the real resolution this tensor is
    # (B*T_o*Xl*Yl*Zl, 64) == every voxel of the full-res grid, so a `.long()` copy of
    # it costs ~500 MB on its own.
    flat = blocks.reshape(-1, K)

    # Per-block class histogram, one class at a time.
    #
    # The obvious `F.one_hot(flat, num_classes).sum(dim=1)` allocates an
    # (M, K, num_classes) int64 intermediate -- at the real Cam4DOcc resolution that is
    # ~1.5 GB for a single batch element, enough on its own to push a 16 GB V100 into
    # OOM. Comparing against one class at a time costs a transient (M, K) bool instead
    # (~63 MB) and is reused across iterations by the caching allocator. num_classes is
    # small (3 for GMO, 17 for lidarseg), so the loop is short.
    counts = torch.stack(
        [(flat == c).sum(dim=1) for c in range(num_classes)], dim=1
    )  # (M, num_classes), int64
    nonempty_count = K - counts[:, 0]
    counts_nonzero = counts.clone()
    counts_nonzero[:, 0] = -1  # never selected as the majority class
    mode_count, mode_class = counts_nonzero.max(dim=1)

    all_empty = nonempty_count == 0
    has_majority = (mode_count * 2 > nonempty_count) & (~all_empty)

    out = torch.full_like(mode_class, fill_value=_IGNORE_INDEX)
    out = torch.where(all_empty, torch.zeros_like(out), out)
    out = torch.where(has_majority, mode_class, out)

    out = out.reshape(*lead, Xl, Yl, Zl).to(dtype=torch.long, device=device)
    return out


def occupancy_loss(
    pred: Tensor,
    target: Tensor,
    class_weights: Optional[Tensor] = None,
    weights_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, Tensor]:
    """Combined occupancy loss: CE + Lovasz-softmax + geo_scal + sem_scal, all live.

    Unlike the OccProphet public release (CE only, others hard-coded to a dead
    0.5 weight), every term here is computed and independently weighted via
    `weights_cfg`. `pred` and `target` must already be at the same (typically
    latent) spatial resolution -- downsample the full-resolution ground truth
    with `downsample_target` first.

    Args:
        pred: Logits `(B, T, num_classes, X, Y, Z)`.
        target: Integer labels `(B, T, X, Y, Z)`, `255` = ignore.
        class_weights: Optional `(num_classes,)` weights for the CE term. If
            `None`, defaults to `w[0]=1.0`, `w[1:]=5.0` per spec SS3.
        weights_cfg: Optional dict with keys `"ce"`, `"lovasz"`, `"geo_scal"`,
            `"sem_scal"` (default all `1.0`) scaling each term.

    Returns:
        Dict of already-weighted scalar terms: `loss_occ_ce`, `loss_occ_lovasz`,
        `loss_occ_geo_scal`, `loss_occ_sem_scal`. Sum the values for the total
        occupancy loss.
    """
    if pred.dim() != 6 or target.dim() != 5:
        raise ValueError(
            f"expected pred (B,T,C,X,Y,Z) and target (B,T,X,Y,Z), got {tuple(pred.shape)} / {tuple(target.shape)}"
        )
    num_classes = pred.shape[2]
    if class_weights is None:
        class_weights = torch.full((num_classes,), 5.0, dtype=pred.dtype, device=pred.device)
        class_weights[0] = 1.0
    cfg = {"ce": 1.0, "lovasz": 1.0, "geo_scal": 1.0, "sem_scal": 1.0}
    if weights_cfg is not None:
        cfg.update(weights_cfg)

    loss_ce = CE_ssc_loss(pred, target, class_weights=class_weights) * cfg["ce"]
    loss_lovasz = lovasz_softmax(pred, target) * cfg["lovasz"]
    loss_geo = geo_scal_loss(pred, target) * cfg["geo_scal"]
    loss_sem = sem_scal_loss(pred, target) * cfg["sem_scal"]

    return {
        "loss_occ_ce": loss_ce,
        "loss_occ_lovasz": loss_lovasz,
        "loss_occ_geo_scal": loss_geo,
        "loss_occ_sem_scal": loss_sem,
    }
