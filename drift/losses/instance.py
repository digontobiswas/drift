"""Instance-query auxiliary loss: Hungarian-matched detection + trajectory ADE. NOVEL.

Supervises the dynamic-agent instance-query path (`drift/models/instance_path.py`):
queries are matched to ground-truth boxes at the first forecasted horizon via
Hungarian assignment (focal classification cost + L1 box cost), then that
assignment is followed by ground-truth `track_id` across the remaining
horizons to score trajectory (average displacement) error -- since the
`MotionForecaster` rolls a *fixed* set of queries forward, index `q` refers to
the same putative track at every horizon.

`InstanceState` (from `drift/models/instance_path.py`) and `BoxSet` (from
`drift/data/cam4docc_dataset.py`) are imported lazily / only for type checking
to avoid a circular import with the models package; every access to their
fields is duck-typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from drift.data.cam4docc_dataset import BoxSet
    from drift.models.instance_path import InstanceState

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

__all__ = ["instance_loss"]


def _sigmoid_focal_loss(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """Elementwise sigmoid focal loss (Lin et al., ICCV 2017), unreduced.

    Args:
        logits: Arbitrary-shape logits.
        targets: Same-shape binary/soft targets in `[0, 1]`.
        alpha: Positive-class balancing weight; `< 0` disables balancing.
        gamma: Focusing exponent.

    Returns:
        Elementwise loss, same shape as `logits`.
    """
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * (1 - p_t).clamp_min(0.0).pow(gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def _linear_sum_assignment_with_fallback(cost: Tensor) -> Tuple[List[int], List[int]]:
    """Solve (or greedily approximate) the linear assignment problem.

    Uses `scipy.optimize.linear_sum_assignment` when scipy is importable;
    otherwise falls back to a greedy nearest-cost matcher (globally suboptimal
    but dependency-free and adequate for a training-time auxiliary loss).

    Args:
        cost: `(N_query, N_gt)` cost matrix.

    Returns:
        `(query_indices, gt_indices)`, each of length `min(N_query, N_gt)`.
    """
    cost_np = cost.detach().cpu().double().numpy()
    n_q, n_g = cost_np.shape
    if n_q == 0 or n_g == 0:
        return [], []
    if _HAS_SCIPY:
        row, col = linear_sum_assignment(cost_np)
        return row.tolist(), col.tolist()

    # Greedy fallback: repeatedly take the globally cheapest remaining pair.
    flat = [(cost_np[i, j], i, j) for i in range(n_q) for j in range(n_g)]
    flat.sort(key=lambda x: x[0])
    used_q, used_g = set(), set()
    rows, cols = [], []
    for _, i, j in flat:
        if i in used_q or j in used_g:
            continue
        used_q.add(i)
        used_g.add(j)
        rows.append(i)
        cols.append(j)
    return rows, cols


DEFAULT_PC_RANGE: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def _norm_center(center: Tensor, pc_range: Optional[List[float]]) -> Tensor:
    """Normalize metric xyz to roughly [0, 1] using the point-cloud range.

    Box regression on raw metric coordinates is badly scaled: over a +/-51.2 m range a typical
    early-training error is tens of metres, so the L1 term dominates every other loss by two
    orders of magnitude. Normalizing by the range extent (the DETR3D / BEVFormer convention)
    puts the box term on the same scale as the classification and occupancy terms.
    """
    if pc_range is None:
        return center
    lo = center.new_tensor(pc_range[:3])
    extent = center.new_tensor(pc_range[3:]) - lo
    return (center - lo) / extent


def _norm_size(size: Tensor, pc_range: Optional[List[float]]) -> Tensor:
    """Normalize box dimensions by the same extent, so sizes are also O(1)."""
    if pc_range is None:
        return size
    extent = size.new_tensor(pc_range[3:]) - size.new_tensor(pc_range[:3])
    return size / extent


def _box_cost_matrix(
    pred_center: Tensor, pred_size: Tensor, pred_yaw: Tensor, gt_center: Tensor, gt_size: Tensor,
    gt_yaw: Tensor, pc_range: Optional[List[float]] = None,
) -> Tensor:
    """Pairwise L1 box cost, `(N_query, N_gt)`, on range-normalized coordinates."""
    return (
        torch.cdist(_norm_center(pred_center, pc_range), _norm_center(gt_center, pc_range), p=1)
        + torch.cdist(_norm_size(pred_size, pc_range), _norm_size(gt_size, pc_range), p=1)
        + torch.cdist(pred_yaw, gt_yaw, p=1)
    )


def _cls_cost_matrix(pred_logits: Tensor, gt_label: Tensor) -> Tensor:
    """Pairwise classification cost `(N_query, N_gt)`: negative predicted prob of the gt class."""
    probs = torch.sigmoid(pred_logits)  # (N_query, num_classes)
    # cost[q, g] = -probs[q, gt_label[g]]  (lower cost = query more confidently predicts that class)
    return -probs[:, gt_label]


def instance_loss(
    states: List["InstanceState"],
    gt_boxes: List[List["BoxSet"]],
    cls_weight: float = 2.0,
    box_weight: float = 5.0,
    traj_weight: float = 1.0,
    score_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    point_cloud_range: Optional[List[float]] = None,
) -> Dict[str, Tensor]:
    """Hungarian-matched instance detection + trajectory loss.

    Matching is performed once per batch element, at the first forecast
    horizon (`states[0]`), using a cost combining sigmoid-focal classification
    cost and L1 box cost. That query<->gt assignment is then carried forward:
    for each later horizon `t`, the matched ground-truth track (identified by
    `track_id`) is looked up in `gt_boxes[b][t]` if it is still in-range at
    that frame (tracks that leave the scene simply stop contributing to the
    trajectory term, matching the "objects whose OBB is not fully inside
    pc_range are dropped for that frame" data convention).

    Args:
        states: Length-`T_o` list of `InstanceState`, one per forecast
            horizon, each with `(B, Q, ...)` fields sharing query identity
            across the list (as produced by `MotionForecaster`).
        gt_boxes: `gt_boxes[b][t]` is the `BoxSet` of ground-truth objects for
            batch element `b` at forecast horizon `t`. Must duck-type
            `center (N,3)`, `size (N,3)`, `yaw (N,1)`, `velocity (N,3)`,
            `label (N,)`, `track_id (N,)`.
        cls_weight: Weight on the matched-frame classification term.
        box_weight: Weight on the matched-frame box regression term.
        traj_weight: Weight on the cross-horizon trajectory ADE term.
        score_weight: Weight on the objectness-score BCE term.
        focal_alpha: Focal loss alpha.
        focal_gamma: Focal loss gamma.

    Returns:
        Dict with `loss_instance_cls`, `loss_instance_box`, `loss_instance_traj`,
        `loss_instance_score`, each an already-weighted scalar tensor.
    """
    if len(states) == 0:
        raise ValueError("states must be a non-empty list of InstanceState")
    T_o = len(states)
    if point_cloud_range is None:
        point_cloud_range = DEFAULT_PC_RANGE
    pcr = point_cloud_range

    B, Q = states[0].center.shape[:2]
    device = states[0].center.device
    dtype = states[0].center.dtype
    num_classes = states[0].logits.shape[-1]

    zero = torch.zeros((), device=device, dtype=dtype)
    loss_cls = zero.clone()
    loss_box = zero.clone()
    loss_traj = zero.clone()
    loss_score = zero.clone()
    n_cls_terms = 0
    n_box_terms = 0
    n_traj_terms = 0
    n_score_terms = 0

    for b in range(B):
        pred0 = states[0]
        gt0 = gt_boxes[b][0] if len(gt_boxes[b]) > 0 else None
        n_gt0 = 0 if gt0 is None else gt0.center.shape[0]

        cls_target = torch.zeros(Q, num_classes, device=device, dtype=dtype)
        score_target = torch.zeros(Q, 1, device=device, dtype=dtype)
        matched_q: List[int] = []
        matched_g: List[int] = []

        if n_gt0 > 0:
            cost_box = _box_cost_matrix(
                pred0.center[b], pred0.size[b], pred0.yaw[b], gt0.center, gt0.size,
                gt0.yaw.reshape(-1, 1), pc_range=pcr,
            )
            cost_cls = _cls_cost_matrix(pred0.logits[b], gt0.label.long())
            cost = box_weight * cost_box + cls_weight * cost_cls
            matched_q, matched_g = _linear_sum_assignment_with_fallback(cost)

            if matched_q:
                q_idx = torch.as_tensor(matched_q, device=device, dtype=torch.long)
                g_idx = torch.as_tensor(matched_g, device=device, dtype=torch.long)
                cls_target[q_idx] = F.one_hot(gt0.label.long()[g_idx], num_classes=num_classes).to(dtype)
                score_target[q_idx] = 1.0

                box_l1 = (
                    F.l1_loss(
                        _norm_center(pred0.center[b][q_idx], pcr),
                        _norm_center(gt0.center[g_idx], pcr), reduction="mean")
                    + F.l1_loss(
                        _norm_size(pred0.size[b][q_idx], pcr),
                        _norm_size(gt0.size[g_idx], pcr), reduction="mean")
                    + F.l1_loss(pred0.yaw[b][q_idx], gt0.yaw.reshape(-1, 1)[g_idx], reduction="mean")
                )
                loss_box = loss_box + box_l1
                n_box_terms += 1

        # Classification / objectness are computed for every query (matched -> its class,
        # unmatched -> background) so gradient reaches all Q queries even with zero GT.
        loss_cls = loss_cls + _sigmoid_focal_loss(
            pred0.logits[b], cls_target, alpha=focal_alpha, gamma=focal_gamma
        ).mean()
        n_cls_terms += 1
        loss_score = loss_score + F.binary_cross_entropy_with_logits(
            _score_to_logit(pred0.score[b]), score_target, reduction="mean"
        )
        n_score_terms += 1

        if not matched_q:
            continue

        track_ids = gt0.track_id.long()[torch.as_tensor(matched_g, device=device, dtype=torch.long)]
        q_idx = torch.as_tensor(matched_q, device=device, dtype=torch.long)

        for t in range(1, T_o):
            frame_boxes = gt_boxes[b][t] if t < len(gt_boxes[b]) else None
            if frame_boxes is None or frame_boxes.center.shape[0] == 0:
                continue
            frame_track_ids = frame_boxes.track_id.long()
            # For each matched query, find whether its track is still present at frame t.
            match_mask = track_ids.unsqueeze(1) == frame_track_ids.unsqueeze(0)  # (n_matched, n_frame)
            present = match_mask.any(dim=1)
            if not bool(present.any()):
                continue
            gt_pos = match_mask[present].float().argmax(dim=1)
            pred_pos_t = states[t].center[b][q_idx[present]]
            gt_center_t = frame_boxes.center[gt_pos]
            loss_traj = loss_traj + F.l1_loss(
                _norm_center(pred_pos_t, pcr), _norm_center(gt_center_t, pcr), reduction="mean")
            n_traj_terms += 1

    loss_cls = (loss_cls / max(n_cls_terms, 1)) * cls_weight
    loss_box = (loss_box / max(n_box_terms, 1)) * box_weight
    loss_traj = (loss_traj / max(n_traj_terms, 1)) * traj_weight
    loss_score = (loss_score / max(n_score_terms, 1)) * score_weight

    return {
        "loss_instance_cls": loss_cls,
        "loss_instance_box": loss_box,
        "loss_instance_traj": loss_traj,
        "loss_instance_score": loss_score,
    }


def _score_to_logit(score: Tensor, eps: float = 1e-4) -> Tensor:
    """Treat `InstanceState.score` (in `[0,1]`) as a probability and recover its logit.

    `score` is documented as "objectness in [0,1]" -- if it is produced by a
    sigmoid already, callers computing an NLL directly on it would double
    apply the nonlinearity. We instead invert it back to a logit so the BCE
    term above is well-posed regardless of whether the head returned a raw
    logit or a sigmoid-activated score; when `score` is already outside
    `(0,1)` (i.e. it *was* a raw logit) this is a no-op modulo the clamp.
    """
    s = score.clamp(eps, 1.0 - eps)
    if bool(((score < 0) | (score > 1)).any()):
        return score
    return torch.log(s / (1.0 - s))
