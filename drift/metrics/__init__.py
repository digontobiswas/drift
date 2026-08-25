"""DRIFT evaluation metrics. See docs/DESIGN_SPEC.md SS5."""

from drift.metrics.calibration import expected_calibration_error, reliability_curve
from drift.metrics.flow_epe import FlowEPEMetric, flow_epe
from drift.metrics.iou import OccupancyIoUMetric, cm_to_ious, fast_hist
from drift.metrics.robustness import DEFAULT_SCENARIOS, DropoutScenario, apply_modality_dropout, evaluate_robustness

__all__ = [
    "fast_hist",
    "cm_to_ious",
    "OccupancyIoUMetric",
    "flow_epe",
    "FlowEPEMetric",
    "expected_calibration_error",
    "reliability_curve",
    "DropoutScenario",
    "DEFAULT_SCENARIOS",
    "apply_modality_dropout",
    "evaluate_robustness",
]
