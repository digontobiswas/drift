"""Robustness evaluation: run a model under a grid of sensor-dropout scenarios. NOVEL.

Exercises the same dropout mechanism `drift.models.cmli.ModalityDropout` uses
at train time, but applied deterministically at eval time over a fixed grid
of `(lidar_dropout, cam_dropout)` scenarios, so callers can quantify how much
`IoU_c`/`IoU_f` degrade as each modality becomes less reliable -- directly
measuring what `CrossModalLatentImagination` is supposed to buy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from drift.metrics.iou import OccupancyIoUMetric

__all__ = ["DropoutScenario", "DEFAULT_SCENARIOS", "apply_modality_dropout", "evaluate_robustness"]


@dataclass
class DropoutScenario:
    """One point in the sensor-dropout evaluation grid.

    Attributes:
        name: Human-readable scenario label (used as the degradation-table row key).
        lidar_dropout: Per-frame probability of the LiDAR modality being marked absent.
        cam_dropout: Per-frame probability of the camera modality being marked absent.
    """

    name: str
    lidar_dropout: float
    cam_dropout: float


DEFAULT_SCENARIOS: List[DropoutScenario] = [
    DropoutScenario("clean", 0.0, 0.0),
    DropoutScenario("lidar_only", 0.0, 1.0),
    DropoutScenario("camera_only", 1.0, 0.0),
    DropoutScenario("moderate_dropout", 0.3, 0.3),
    DropoutScenario("severe_dropout", 0.7, 0.7),
]


def apply_modality_dropout(
    batch: Dict[str, Any],
    lidar_dropout: float,
    cam_dropout: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Return a shallow copy of `batch` with `lidar_mask`/`cam_mask` stochastically zeroed.

    Only the mask tensors are modified (the model, via
    `drift.models.cmli.CrossModalLatentImagination`, is expected to consult
    these masks to decide which modality's latent to substitute -- this
    function does not itself need to zero `points`/`imgs`, since a `mask==0`
    frame's raw features are expected to be ignored downstream regardless of
    their content). Every other batch entry is passed through by reference.

    Args:
        batch: A DRIFT batch dict (spec SS4) containing `lidar_mask` and
            `cam_mask`, each `(B, T_p)`.
        lidar_dropout: Per-frame probability of dropping LiDAR.
        cam_dropout: Per-frame probability of dropping the camera.
        generator: Optional `torch.Generator` for reproducible sampling.

    Returns:
        A new dict (shallow copy of `batch`) with `lidar_mask`/`cam_mask`
        replaced.
    """
    if "lidar_mask" not in batch or "cam_mask" not in batch:
        raise ValueError("apply_modality_dropout requires 'lidar_mask' and 'cam_mask' in batch")
    out = dict(batch)
    lidar_mask = batch["lidar_mask"]
    cam_mask = batch["cam_mask"]
    if lidar_dropout > 0:
        keep = torch.rand(lidar_mask.shape, generator=generator, device=lidar_mask.device) >= lidar_dropout
        out["lidar_mask"] = lidar_mask * keep.to(lidar_mask.dtype)
    if cam_dropout > 0:
        keep = torch.rand(cam_mask.shape, generator=generator, device=cam_mask.device) >= cam_dropout
        out["cam_mask"] = cam_mask * keep.to(cam_mask.dtype)
    return out


def evaluate_robustness(
    model: Callable[[Dict[str, Any]], Dict[str, Any]],
    dataloader: Iterable[Dict[str, Any]],
    num_classes: int,
    num_future: int,
    scenarios: Optional[List[DropoutScenario]] = None,
    pred_key: str = "occ_logits",
    gt_key: str = "gt_occ",
    upsample_size: tuple = (512, 512, 40),
    present_index: int = 0,
    max_batches: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Run `model` under each dropout scenario and report the IoU degradation table.

    Args:
        model: Callable mapping a batch dict to an output dict containing
            `pred_key` (occupancy logits, `(B, T_o, num_classes, x, y, z)`).
        dataloader: Iterable of DRIFT batch dicts (spec SS4); consumed fresh
            for every scenario (must support being iterated multiple times,
            e.g. a `DataLoader`, not a one-shot generator).
        num_classes: Occupancy class count, passed to `OccupancyIoUMetric`.
        num_future: `T_o`, passed to `OccupancyIoUMetric`.
        scenarios: Dropout grid to evaluate; defaults to `DEFAULT_SCENARIOS`.
            The first scenario is used as the "clean" baseline for the
            reported degradation percentage regardless of its name.
        pred_key: Key in the model's output dict holding occupancy logits.
        gt_key: Key in the batch dict holding full-resolution occupancy ground truth.
        upsample_size: Forwarded to `OccupancyIoUMetric`.
        present_index: Forwarded to `OccupancyIoUMetric`.
        max_batches: If set, only the first `max_batches` batches of
            `dataloader` are evaluated per scenario (for a fast smoke run).
        seed: Base seed for the dropout RNG (offset per scenario so scenarios
            don't share dropout draws).

    Returns:
        Dict keyed by scenario name, each value a dict with `IoU_c`, `IoU_f`,
        `per_horizon_IoU`, and (for every scenario after the first)
        `IoU_f_degradation_pct` = the relative drop in `IoU_f` from the first
        (baseline) scenario.
    """
    scenarios = scenarios if scenarios is not None else DEFAULT_SCENARIOS
    if len(scenarios) == 0:
        raise ValueError("evaluate_robustness requires at least one scenario")

    table: Dict[str, Dict[str, Any]] = {}
    baseline_iou_f: Optional[float] = None

    for s_i, scenario in enumerate(scenarios):
        generator = torch.Generator().manual_seed(seed + s_i)
        metric = OccupancyIoUMetric(
            num_classes=num_classes, num_future=num_future, upsample_size=upsample_size, present_index=present_index
        )
        with torch.no_grad():
            for b_i, batch in enumerate(dataloader):
                if max_batches is not None and b_i >= max_batches:
                    break
                dropped = apply_modality_dropout(batch, scenario.lidar_dropout, scenario.cam_dropout, generator)
                outputs = model(dropped)
                if pred_key not in outputs:
                    raise ValueError(f"evaluate_robustness: model output is missing '{pred_key}'")
                if gt_key not in dropped:
                    raise ValueError(f"evaluate_robustness: batch is missing '{gt_key}'")
                metric.update(outputs[pred_key], dropped[gt_key])

        result = metric.compute()
        row = {
            "lidar_dropout": scenario.lidar_dropout,
            "cam_dropout": scenario.cam_dropout,
            "IoU_c": result["IoU_c"],
            "IoU_f": result["IoU_f"],
            "per_horizon_IoU": result["per_horizon_IoU"],
        }
        if s_i == 0:
            baseline_iou_f = result["IoU_f"]
            row["IoU_f_degradation_pct"] = 0.0
        else:
            if baseline_iou_f is not None and abs(baseline_iou_f) > 1e-8:
                row["IoU_f_degradation_pct"] = 100.0 * (baseline_iou_f - result["IoU_f"]) / baseline_iou_f
            else:
                row["IoU_f_degradation_pct"] = float("nan")
        table[scenario.name] = row

    return table
