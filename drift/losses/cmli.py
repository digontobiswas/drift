"""Cross-Modal Latent Imagination (CMLI) consistency loss. NOVEL.

Supervises `drift/models/cmli.py`'s `CrossModalLatentImagination` generator: at
frames where a modality's real latent *was* observed, that real latent is a
free teacher signal for what the "imagined" reconstruction of that modality
should have produced (whether or not the imagined value was actually
substituted into the fused pipeline at that frame -- the generator runs on
every frame regardless of the dropout mask). This trains the imagination
network to be a faithful cross-modal predictor, which is what lets the model
degrade gracefully when a modality really is missing at inference time.

Interface contract (see the "Interface concerns" note in the accompanying
report): `aux` is expected to carry, in addition to the
`CrossModalLatentImagination.forward` outputs (`imagined_lidar`,
`imagined_cam`, `disagreement`), the *real* pre-substitution latents
`real_lidar` / `real_cam` that the imagined ones are compared against. These
are not emitted by `CrossModalLatentImagination.forward` per its SS2.4 contract,
so the model assembly code (`drift/models/drift.py`) is expected to add them to
`aux` before calling this loss -- this file degrades gracefully (skips the
missing term, does not raise) if they are absent, so it stays usable while
that wiring is finalized by the model-side agent.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["cmli_consistency_loss"]


def _masked_l2(imagined: Tensor, real: Tensor, mask: Tensor) -> Optional[Tensor]:
    """Mean squared error between `imagined` and `real`, restricted to frames where `mask == 1`.

    Args:
        imagined: `(B, T, C, X, Y, Z)`.
        real: `(B, T, C, X, Y, Z)`.
        mask: `(B, T)` float in `{0, 1}`.

    Returns:
        Scalar tensor, or `None` if no frame is masked in (nothing to supervise).
    """
    if imagined.shape != real.shape:
        raise ValueError(f"imagined/real shape mismatch: {tuple(imagined.shape)} vs {tuple(real.shape)}")
    m = mask.to(imagined.dtype)
    n_valid = m.sum()
    if float(n_valid.item()) == 0.0:
        return None
    sq_err = (imagined - real).pow(2).mean(dim=2)  # (B,T,X,Y,Z), mean over channels
    sq_err = sq_err.flatten(2).mean(dim=2)  # (B,T), mean over space
    return (sq_err * m).sum() / n_valid


def cmli_consistency_loss(
    aux: Dict[str, Tensor],
    masks: Dict[str, Tensor],
    occ_weight: float = 0.1,
) -> Dict[str, Tensor]:
    """L2 imagination-consistency loss (+ optional occupancy-consistency term).

    Args:
        aux: Dict expected to contain (see module docstring):
            `imagined_lidar`, `imagined_cam` (always required, from
            `CrossModalLatentImagination.forward`'s aux output), and
            optionally `real_lidar`, `real_cam` (the pre-substitution
            latents) and `occ_mask_logits_from_imagined` /
            `occ_mask_logits_from_real` (both `(B,T,1,X,Y,Z)`, for the
            occupancy-consistency term).
        masks: Dict with `lidar_mask`, `cam_mask`, each `(B, T)` float in
            `{0, 1}` -- 1 where that modality was actually observed (so the
            teacher signal is valid there).
        occ_weight: Weight on the occupancy-consistency term.

    Returns:
        Dict with `loss_cmli_lidar`, `loss_cmli_cam`, `loss_cmli_occ`. A term
        is an autograd-connected zero (not omitted) when its required inputs
        are absent from `aux`, so the returned dict shape is stable.
    """
    for key in ("imagined_lidar", "imagined_cam"):
        if key not in aux:
            raise ValueError(f"cmli_consistency_loss requires aux['{key}']; got keys {list(aux.keys())}")
    for key in ("lidar_mask", "cam_mask"):
        if key not in masks:
            raise ValueError(f"cmli_consistency_loss requires masks['{key}']; got keys {list(masks.keys())}")

    device = aux["imagined_lidar"].device
    dtype = aux["imagined_lidar"].dtype
    zero = aux["imagined_lidar"].sum() * 0.0 + aux["imagined_cam"].sum() * 0.0

    if "real_lidar" in aux:
        lidar_term = _masked_l2(aux["imagined_lidar"], aux["real_lidar"], masks["lidar_mask"])
        loss_lidar = lidar_term if lidar_term is not None else zero
    else:
        loss_lidar = zero

    if "real_cam" in aux:
        cam_term = _masked_l2(aux["imagined_cam"], aux["real_cam"], masks["cam_mask"])
        loss_cam = cam_term if cam_term is not None else zero
    else:
        loss_cam = zero

    if "occ_mask_logits_from_imagined" in aux and "occ_mask_logits_from_real" in aux:
        pred_logits = aux["occ_mask_logits_from_imagined"]
        target_logits = aux["occ_mask_logits_from_real"]
        target_probs = torch.sigmoid(target_logits).detach()
        joint_mask = (masks["lidar_mask"] * masks["cam_mask"]).to(dtype)
        n_valid = joint_mask.sum()
        if float(n_valid.item()) > 0.0:
            per_frame = F.binary_cross_entropy_with_logits(pred_logits, target_probs, reduction="none")
            per_frame = per_frame.flatten(2).mean(dim=2)  # (B,T)
            loss_occ = (per_frame * joint_mask).sum() / n_valid * occ_weight
        else:
            loss_occ = zero
    else:
        loss_occ = zero

    return {
        "loss_cmli_lidar": loss_lidar,
        "loss_cmli_cam": loss_cam,
        "loss_cmli_occ": loss_occ,
    }
