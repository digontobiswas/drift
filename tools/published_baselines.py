"""Published Cam4DOcc-protocol numbers, transcribed once so no table is retyped by hand.

Every value here is a number *another paper reported*, not something this repository
measured. That distinction is the whole reason this file exists as data rather than as a
paragraph in a markdown table: a transcribed baseline that drifts out of sync with its source,
or gets quietly mixed in with measured numbers, is the kind of error that survives all the way
into a submitted table.

Three things about these numbers will break a comparison if they are not handled explicitly,
and all three are encoded below rather than left to whoever builds the table:

1. **Units.** Published Cam4DOcc numbers are percentages (OCFNet's inflated-GMO ``IoU_c`` is
   ``31.30``, meaning 31.30%). ``tools/eval.py`` reports fractions in ``[0, 1]``
   (``0.1357``). Putting the two in one column without scaling produces a table where the
   baseline appears 230x better than it is. ``as_percent`` below is the only conversion
   anything should use.

2. **Task setting.** Cam4DOcc reports *inflated GMO* and *fine-grained GMO* as separate
   tasks with very different numbers (OCFNet: 31.30 vs 11.45 ``IoU_c``). ``configs.cam4docc_gmo``
   is a 3-class preset -- free / GSO / GMO, one unified movable-object class -- which is the
   **inflated GMO** setting. Comparing a DRIFT GMO number against the fine-grained column, or
   vice versa, is comparing two different benchmarks.

3. **``IoU_c`` is not the same quantity in both papers.** Cam4DOcc defines ``IoU_c`` as
   occupancy estimation at the *current* moment, t=0 -- a reconstruction of a frame the model
   has already observed. DRIFT's data/model contract has no present-frame slot at all: every
   one of its ``T_o`` outputs is a genuine future frame, and ``present_index=0`` merely labels
   the nearest one, t=+0.5s (see ``tools/eval.py``'s module docstring and README Known
   Limitations). So DRIFT's ``IoU_c`` is a forecast and Cam4DOcc's is a reconstruction. The
   forecast is the harder task, which means this comparison is *conservative* for DRIFT -- but
   it is still not apples-to-apples, and every table built from this file carries the caveat.

``IoU_f`` is the safer of the two columns: both papers mean "averaged over the future frames
out to +2.0s", and DRIFT's ``cam4docc_2s``/``cam4docc_gmo`` presets are built to that horizon.

Sources
-------
Primary: Cam4DOcc, CVPR 2024, arXiv:2311.17663, Tables 2 and 3.
Secondary (agrees with the primary on every shared value, and adds OccProphet):
OccProphet, arXiv:2502.15180.

Unverified
----------
``OCFNet†`` is transcribed exactly as printed. The dagger's meaning is defined in the source
table's caption and could not be confirmed from the abstract page; ``needs_verification`` is
set on those rows so any table built from them says so rather than implying it was checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "PublishedResult",
    "INFLATED_GMO",
    "FINE_GRAINED_GMO",
    "MODEL_COMPLEXITY",
    "as_percent",
    "for_protocol",
]


@dataclass(frozen=True)
class PublishedResult:
    """One row of another paper's results table, transcribed verbatim.

    Attributes:
        method: Method name exactly as printed in the source table.
        protocol: ``"inflated_gmo"`` or ``"fine_grained_gmo"`` -- which Cam4DOcc task setting.
        iou_c: Published ``IoU_c`` as a PERCENTAGE (0-100), or ``None`` where the source
            printed a failure rather than a number.
        iou_f: Published ``IoU_f`` as a PERCENTAGE (0-100), or ``None``.
        source: Short citation for the table this row came from.
        needs_verification: True when something about this row is transcribed but unconfirmed
            (see the module docstring). Tables must surface this rather than swallow it.
        note: Anything a reader of the comparison table needs in order not to misread the row.
    """

    method: str
    protocol: str
    iou_c: Optional[float]
    iou_f: Optional[float]
    source: str
    needs_verification: bool = False
    note: str = ""


# Cam4DOcc Table 2 -- nuScenes, inflated GMO (vehicle and human as one movable class).
# This is the setting `configs.cam4docc_gmo` matches: 3 classes, free / GSO / GMO.
INFLATED_GMO: List[PublishedResult] = [
    PublishedResult(
        "OpenOccupancy-C", "inflated_gmo", 11.45, 11.74,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 2",
    ),
    PublishedResult(
        "SPC (point-cloud prediction)", "inflated_gmo", 1.27, None,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 2",
        note="Source prints a failure rather than an IoU_f value.",
    ),
    PublishedResult(
        "PowerBEV-3D", "inflated_gmo", 23.08, 21.25,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 2",
    ),
    PublishedResult(
        "OCFNet", "inflated_gmo", 27.86, 23.89,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 2",
        note="The benchmark's own reference model.",
    ),
    PublishedResult(
        "OCFNet(dagger)", "inflated_gmo", 31.30, 26.82,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 2",
        needs_verification=True,
        note="Dagger variant, transcribed as printed; check the source caption for what the "
             "dagger denotes before citing this row.",
    ),
    PublishedResult(
        "OccProphet", "inflated_gmo", 34.36, 26.94,
        "OccProphet (arXiv:2502.15180), 4D occupancy forecasting comparison table",
        note="Camera-only; the strongest published number on this task at transcription time.",
    ),
]

# Cam4DOcc Table 3 -- nuScenes-Occupancy, fine-grained GMO (movable objects split into
# per-category classes). Recorded for completeness; `cam4docc_gmo` is NOT this setting.
FINE_GRAINED_GMO: List[PublishedResult] = [
    PublishedResult(
        "OpenOccupancy-C", "fine_grained_gmo", 10.82, 8.02,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 3",
    ),
    PublishedResult(
        "SPC (point-cloud prediction)", "fine_grained_gmo", 5.85, 1.08,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 3",
    ),
    PublishedResult(
        "PowerBEV-3D", "fine_grained_gmo", 5.91, 5.25,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 3",
    ),
    PublishedResult(
        "OCFNet", "fine_grained_gmo", 10.15, 8.35,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 3",
    ),
    PublishedResult(
        "OCFNet(dagger)", "fine_grained_gmo", 11.45, 9.68,
        "Cam4DOcc (CVPR 2024, arXiv:2311.17663), Table 3",
        needs_verification=True,
        note="Dagger variant, transcribed as printed; check the source caption.",
    ),
    PublishedResult(
        "OccProphet", "fine_grained_gmo", 15.38, 10.69,
        "OccProphet (arXiv:2502.15180), 4D occupancy forecasting comparison table",
    ),
]

# Cost figures, for the accuracy-efficiency Pareto plot. Reported on the publishing authors'
# own hardware, which is not this cluster's V100s -- so these are a rough frame of reference
# for magnitude, not a controlled latency comparison against `tools/benchmark_latency.py`.
MODEL_COMPLEXITY: List[dict] = [
    {
        "method": "Cam4DOcc (OCFNet)", "params_m": 370.0, "gpu_memory_gb": 57.0,
        "gflops": 6434.0, "fps": 1.7,
        "source": "OccProphet (arXiv:2502.15180), model-complexity table",
        "note": "Measured on the OccProphet authors' hardware, not on this cluster.",
    },
    {
        "method": "OccProphet", "params_m": 82.0, "gpu_memory_gb": 24.0,
        "gflops": 1985.0, "fps": 4.5,
        "source": "OccProphet (arXiv:2502.15180), model-complexity table",
        "note": "Measured on the OccProphet authors' hardware, not on this cluster.",
    },
]

# The caveat that has to travel with every table mixing these rows and DRIFT's own.
COMPARISON_CAVEAT = (
    "Published rows are transcribed from the cited papers, not re-run here. DRIFT rows are "
    "measured by tools/eval.py on this cluster. Cam4DOcc's IoU_c is occupancy estimation at "
    "t=0 (a frame the model observed); DRIFT has no present-frame slot, so its IoU_c column is "
    "a forecast at t=+0.5s -- a strictly harder quantity. The IoU_f columns are comparable in "
    "kind (mean over future frames to +2.0s). All values are percentages."
)


def as_percent(fraction: Optional[float]) -> Optional[float]:
    """Convert a ``tools/eval.py`` IoU (a fraction in ``[0, 1]``) to the published scale.

    This exists so the fraction-vs-percent conversion happens in exactly one place. Getting it
    wrong does not raise anything -- it produces a table that looks finished and is off by two
    orders of magnitude.

    Args:
        fraction: An IoU in ``[0, 1]``, or ``None``.

    Returns:
        The same value as a percentage, or ``None`` if ``fraction`` was ``None``.
    """
    return None if fraction is None else fraction * 100.0


def for_protocol(protocol: str) -> List[PublishedResult]:
    """Return the published rows for one Cam4DOcc task setting.

    Args:
        protocol: ``"inflated_gmo"`` or ``"fine_grained_gmo"``.

    Returns:
        The matching rows, in the source table's order.

    Raises:
        ValueError: On an unknown protocol, rather than returning an empty list that would
            render as a comparison table with no baselines in it.
    """
    table = {"inflated_gmo": INFLATED_GMO, "fine_grained_gmo": FINE_GRAINED_GMO}
    if protocol not in table:
        raise ValueError(
            f"Unknown protocol {protocol!r}; expected one of {sorted(table)}. "
            "configs.cam4docc_gmo (3 classes: free/GSO/GMO) is the 'inflated_gmo' setting."
        )
    return list(table[protocol])
