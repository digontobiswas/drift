#!/usr/bin/env python
"""Turn the evaluation grid's JSONs into CSVs that open in Excel, pandas, or a plotting script.

``slurm/04_eval.slurm`` writes one ``eval_<config>.json`` per ablation, ``slurm/05_robustness.slurm``
adds ``robustness_<config>.json`` and ``latency_<config>.json``. Those files are nested and
awkward to read by hand -- per-class IoU is a list, calibration is four parallel lists, the
robustness table is a dict of dicts. This flattens all of it into tidy CSVs: one row per
observation, every row carrying the columns that identify it, so a pivot table or a groupby
does the rest.

``tools/collect_results.py`` renders the same JSONs as markdown for pasting into a paper.
This is the machine-readable sibling: same inputs, no formatting decisions, nothing dropped.

    python tools/results_to_csv.py --results-dir $DRIFT_RESULTS_DIR
    python tools/results_to_csv.py --results-dir $DRIFT_RESULTS_DIR --out-dir ~/drift_csv

Written (only for the data actually present -- a half-finished grid produces the CSVs it can):

    summary.csv             one row per config: headline IoU, flow, calibration
    per_class_iou.csv       config x class: present and future IoU
    per_horizon_iou.csv     config x forecast horizon: the degradation curve
    calibration.csv         config x confidence bin: the reliability diagram's data
    robustness.csv          config x sensor-dropout scenario
    latency.csv             config: latency, memory, parameters, FLOPs
    baseline_comparison.csv DRIFT beside the published Cam4DOcc numbers, in percent
    README_CSV.md           what each column means and which caveats apply

Units
-----
IoU columns in every file except ``baseline_comparison.csv`` are fractions in ``[0, 1]``,
exactly as ``tools/eval.py`` reports them. ``baseline_comparison.csv`` is the one file that
converts to percentages, because that is the scale the published numbers are on -- mixing the
two scales in one column is the failure mode that file exists to prevent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.published_baselines import (  # noqa: E402
    COMPARISON_CAVEAT,
    MODEL_COMPLEXITY,
    as_percent,
    for_protocol,
)

# Presentation order and the one-line reading each ablation supports. Mirrors
# tools/collect_results.py so the CSV and the markdown table order rows identically.
ABLATIONS: List[tuple] = [
    ("cam4docc_gmo", "Full model"),
    ("no_cmli", "- cross-modal latent imagination"),
    ("no_instance_path", "- instance queries (dense flow)"),
    ("no_uncertainty", "- per-voxel uncertainty head"),
    ("camera_only", "camera only (no LiDAR encoder)"),
    ("lidar_only", "LiDAR only (no camera encoder)"),
    ("fusion_sum", "sum fusion (not gated)"),
]

# `configs.cam4docc_gmo` is 3-class: 0 = free, 1 = GSO (static), 2 = GMO (movable). Index 2 is
# the quantity the Cam4DOcc benchmark actually reports, so it is pulled out as its own column
# rather than left for the reader to find in per_class_iou.csv.
GMO_CLASS_INDEX = 2
GMO_PRESET_NUM_CLASSES = 3


def _load(path: Path) -> Optional[Dict[str, Any]]:
    """Read one results JSON, or return None if it is absent or unreadable.

    A truncated JSON -- a job killed mid-write -- is treated as absent rather than raising, so
    one bad file does not stop the other six configs from being exported.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[csv] skipping unreadable {path.name}: {exc}", file=sys.stderr)
        return None


def _write(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> int:
    """Write one CSV and report how many data rows it got.

    Returns:
        The number of rows written, so the caller can skip announcing an empty file.
    """
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _horizon_seconds(bucket_index_1based: int) -> float:
    """Seconds into the future for cumulative per-horizon bucket ``k``.

    Mirrors ``tools/eval.py``'s ``format_report`` exactly (``0.5 * (k + 1)``, k starting at 1):
    bucket k excludes the present_index slot itself, so the first bucket is already +1.0s. The
    formula is duplicated rather than re-derived so the CSV and the printed report can never
    disagree about what a column means.
    """
    return 0.5 * (bucket_index_1based + 1)


def _discovered_configs(rdir: Path) -> List[tuple]:
    """Every config with an ``eval_*.json``, in ABLATIONS order, then any extras found on disk.

    A config the grid does not know about (a one-off eval of a renamed preset) still gets
    exported; it just sorts after the known ones and carries an empty description.
    """
    known = [name for name, _ in ABLATIONS]
    found = {p.stem[len("eval_"):] for p in rdir.glob("eval_*.json")}
    ordered = [(n, d) for n, d in ABLATIONS if n in found]
    ordered += [(n, "") for n in sorted(found - set(known))]
    return ordered


def build_summary_rows(rdir: Path, configs: Sequence[tuple]) -> List[Dict[str, Any]]:
    """One row per evaluated config: the numbers a results table leads with."""
    rows = []
    for name, desc in configs:
        data = _load(rdir / f"eval_{name}.json")
        if data is None:
            continue
        iou = data.get("iou") or {}
        flow = data.get("flow") or {}
        ece = data.get("ece") or {}
        per_present = iou.get("per_class_present") or []
        per_future = iou.get("per_class_future") or []
        n_classes = len(per_present)
        # Only a 3-class run is the GMO preset; on any other class count, index 2 means
        # something else entirely and must not be labelled GMO.
        is_gmo_preset = n_classes == GMO_PRESET_NUM_CLASSES
        rows.append({
            "config": name,
            "description": desc,
            "checkpoint": data.get("checkpoint", ""),
            "num_classes": n_classes,
            "num_samples": data.get("num_samples"),
            "num_batches": data.get("num_batches"),
            "elapsed_s": data.get("elapsed_s"),
            "IOU_mean": iou.get("IOU_mean"),
            "IoU_c": iou.get("IoU_c"),
            "IoU_f": iou.get("IoU_f"),
            "gmo_iou_present": per_present[GMO_CLASS_INDEX] if is_gmo_preset else None,
            "gmo_iou_future": per_future[GMO_CLASS_INDEX] if is_gmo_preset else None,
            "flow_epe_m": flow.get("epe"),
            "flow_angular_error_rad": flow.get("angular_error"),
            "flow_magnitude_error_m": flow.get("magnitude_error"),
            "flow_valid_voxels": flow.get("n_valid"),
            "ece": ece.get("ece"),
            "ece_valid_voxels": ece.get("n_valid"),
        })
    return rows


def build_per_class_rows(rdir: Path, configs: Sequence[tuple]) -> List[Dict[str, Any]]:
    """Long format: one row per (config, class), both IoU buckets side by side."""
    rows = []
    for name, _ in configs:
        data = _load(rdir / f"eval_{name}.json")
        if data is None:
            continue
        iou = data.get("iou") or {}
        present = iou.get("per_class_present") or []
        future = iou.get("per_class_future") or []
        names = data.get("class_names") or [f"class_{i}" for i in range(len(present))]
        for i, cls in enumerate(names):
            rows.append({
                "config": name,
                "class_index": i,
                "class_name": cls,
                "is_free_class": i == 0,
                "iou_present": present[i] if i < len(present) else None,
                "iou_future": future[i] if i < len(future) else None,
            })
    return rows


def build_per_horizon_rows(rdir: Path, configs: Sequence[tuple]) -> List[Dict[str, Any]]:
    """Long format: the cumulative IoU-vs-horizon curve, one row per point."""
    rows = []
    for name, _ in configs:
        data = _load(rdir / f"eval_{name}.json")
        if data is None:
            continue
        curve = (data.get("iou") or {}).get("per_horizon_IoU") or []
        for k, value in enumerate(curve, start=1):
            rows.append({
                "config": name,
                "bucket_index": k,
                "horizon_s": _horizon_seconds(k),
                "iou_cumulative": value,
            })
    return rows


def build_calibration_rows(rdir: Path, configs: Sequence[tuple]) -> List[Dict[str, Any]]:
    """Long format: the reliability diagram's underlying bins.

    ``gap`` (accuracy minus confidence) is precomputed because it is what the diagram actually
    draws and what ECE integrates: positive means the model is underconfident in that bin,
    negative means overconfident.
    """
    rows = []
    for name, _ in configs:
        data = _load(rdir / f"eval_{name}.json")
        if data is None:
            continue
        ece = data.get("ece") or {}
        conf = ece.get("bin_confidence") or []
        acc = ece.get("bin_accuracy") or []
        count = ece.get("bin_count") or []
        n_bins = len(count)
        if not n_bins:
            continue
        total = sum(count) or 1.0
        for i in range(n_bins):
            c = conf[i] if i < len(conf) else None
            a = acc[i] if i < len(acc) else None
            rows.append({
                "config": name,
                "bin_index": i,
                "bin_lo": i / n_bins,
                "bin_hi": (i + 1) / n_bins,
                "mean_confidence": c,
                "mean_accuracy": a,
                "gap_accuracy_minus_confidence": (a - c) if (a is not None and c is not None) else None,
                "voxel_count": count[i],
                "voxel_fraction": count[i] / total,
            })
    return rows


def build_robustness_rows(rdir: Path) -> List[Dict[str, Any]]:
    """Long format: one row per (config, sensor-dropout scenario).

    Reads ``payload["table"]``, which is the shape ``tools/run_robustness.py`` actually
    writes -- a dict keyed by scenario name, not a list.
    """
    rows = []
    for path in sorted(rdir.glob("robustness_*.json")):
        data = _load(path)
        if data is None:
            continue
        config = data.get("config") or path.stem[len("robustness_"):]
        for scenario, row in (data.get("table") or {}).items():
            degradation = row.get("IoU_f_degradation_pct")
            retained = None
            if isinstance(degradation, (int, float)) and not math.isnan(degradation):
                retained = 100.0 - degradation
            rows.append({
                "config": config,
                "scenario": scenario,
                "lidar_dropout": row.get("lidar_dropout"),
                "cam_dropout": row.get("cam_dropout"),
                "IoU_c": row.get("IoU_c"),
                "IoU_f": row.get("IoU_f"),
                "IoU_f_degradation_pct": degradation,
                "IoU_f_retained_pct": retained,
            })
    return rows


def build_latency_rows(rdir: Path) -> List[Dict[str, Any]]:
    """One row per benchmarked config.

    Reads the nested ``latency`` / ``memory`` / ``params`` / ``flops`` sub-dicts that
    ``tools/benchmark_latency.py`` actually writes.
    """
    rows = []
    for path in sorted(rdir.glob("latency_*.json")):
        data = _load(path)
        if data is None:
            continue
        lat = data.get("latency") or {}
        mem = data.get("memory") or {}
        params = data.get("params") or {}
        flops = data.get("flops") or {}
        peak_mb = mem.get("peak_mb")
        rows.append({
            "config": data.get("config") or path.stem[len("latency_"):],
            "device": data.get("device"),
            "hardware": data.get("hardware"),
            "batch_size": data.get("batch_size"),
            "T_p": data.get("T_p"),
            "T_o": data.get("T_o"),
            "latency_mean_ms": lat.get("mean_ms"),
            "latency_std_ms": lat.get("std_ms"),
            "latency_p95_ms": lat.get("p95_ms"),
            "peak_memory_mb": peak_mb,
            "peak_memory_gb": (peak_mb / 1024.0) if isinstance(peak_mb, (int, float)) else None,
            "memory_method": mem.get("method"),
            "params_total_m": params.get("total_m"),
            "params_trainable_m": params.get("trainable_m"),
            "gflops": flops.get("gflops"),
            "flops_backend": flops.get("backend"),
        })
    return rows


def build_baseline_comparison_rows(
    rdir: Path, configs: Sequence[tuple], protocol: str = "inflated_gmo"
) -> List[Dict[str, Any]]:
    """DRIFT's measured GMO IoU beside the published Cam4DOcc numbers, both in percent.

    Only 3-class (GMO-preset) evaluations produce a DRIFT row: on any other class count there
    is no column that means the same thing as the benchmark's GMO IoU, and inventing one by
    reusing ``IOU_mean`` would silently compare two different quantities.
    """
    rows: List[Dict[str, Any]] = []
    for result in for_protocol(protocol):
        rows.append({
            "method": result.method,
            "kind": "published",
            "protocol": result.protocol,
            "IoU_c_pct": result.iou_c,
            "IoU_f_pct": result.iou_f,
            "measured_here": False,
            "needs_verification": result.needs_verification,
            "source": result.source,
            "note": result.note,
        })

    for name, desc in configs:
        data = _load(rdir / f"eval_{name}.json")
        if data is None:
            continue
        iou = data.get("iou") or {}
        present = iou.get("per_class_present") or []
        future = iou.get("per_class_future") or []
        if len(present) != GMO_PRESET_NUM_CLASSES:
            continue
        rows.append({
            "method": f"DRIFT ({name})",
            "kind": "measured",
            "protocol": protocol,
            "IoU_c_pct": as_percent(present[GMO_CLASS_INDEX]),
            "IoU_f_pct": as_percent(future[GMO_CLASS_INDEX]),
            "measured_here": True,
            "needs_verification": False,
            "source": data.get("checkpoint", "tools/eval.py on this cluster"),
            "note": (desc + "; " if desc else "") + "IoU_c here is a +0.5s forecast, not a t=0 "
                    "reconstruction -- see README_CSV.md.",
        })
    return rows


def build_complexity_rows() -> List[Dict[str, Any]]:
    """Published parameter / FLOP / FPS figures, for the accuracy-efficiency plot."""
    return [
        {
            "method": entry["method"], "kind": "published",
            "params_m": entry["params_m"], "gpu_memory_gb": entry["gpu_memory_gb"],
            "gflops": entry["gflops"], "fps": entry["fps"],
            "source": entry["source"], "note": entry["note"],
        }
        for entry in MODEL_COMPLEXITY
    ]


README_TEMPLATE = """# DRIFT results, as CSV

Generated by `tools/results_to_csv.py` from the JSONs in `{results_dir}`.
Regenerate any time; nothing here is edited by hand.

## Units

IoU values are **fractions in [0, 1]** in every file except `baseline_comparison.csv`, which is
in **percent (0-100)** because that is the scale the published Cam4DOcc numbers use. Do not mix
the two in one column.

## Files

| File | One row is | Use it for |
|---|---|---|
| `summary.csv` | one evaluated config | the headline results table |
| `per_class_iou.csv` | config x semantic class | which classes the model actually gets right |
| `per_horizon_iou.csv` | config x forecast horizon | the accuracy-vs-time degradation curve |
| `calibration.csv` | config x confidence bin | the reliability diagram behind the ECE number |
| `robustness.csv` | config x sensor-dropout scenario | the graceful-degradation claim |
| `latency.csv` | benchmarked config | the accuracy-efficiency trade-off |
| `baseline_comparison.csv` | one method, published or measured | comparison against the literature |
| `model_complexity.csv` | one published method | published cost figures for context |

## Caveats that belong in the paper, not just here

{caveat}

`per_horizon_iou.csv`'s buckets are **cumulative**: bucket k pools every future frame out to
horizon k, which is the Cam4DOcc convention, so the curve is a running average and not a
per-frame score.

In `summary.csv`, `gmo_iou_present` / `gmo_iou_future` are filled only for 3-class runs (the
`cam4docc_gmo` preset, where class 2 is the unified movable-object class). They are blank for
any other class count, because index 2 does not mean GMO there.

`baseline_comparison.csv` contains a DRIFT row only for 3-class runs, for the same reason.
Rows with `needs_verification = True` are transcribed from a source whose table caption was
not confirmed -- check it before citing.
"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True, help="Directory holding eval_*.json etc.")
    p.add_argument("--out-dir", default=None, help="Where to write CSVs (default: <results-dir>/csv).")
    p.add_argument(
        "--protocol", default="inflated_gmo", choices=["inflated_gmo", "fine_grained_gmo"],
        help="Which Cam4DOcc task setting to compare against (cam4docc_gmo is inflated_gmo).",
    )
    args = p.parse_args(argv)

    rdir = Path(args.results_dir)
    if not rdir.is_dir():
        print(f"[csv] results dir not found: {rdir}", file=sys.stderr)
        return 1

    out = Path(args.out_dir) if args.out_dir else rdir / "csv"
    out.mkdir(parents=True, exist_ok=True)
    configs = _discovered_configs(rdir)

    if not configs:
        print(f"[csv] no eval_*.json found in {rdir} -- has the eval grid run?", file=sys.stderr)

    written: List[str] = []
    for filename, fields, rows in [
        ("summary.csv",
         ["config", "description", "checkpoint", "num_classes", "num_samples", "num_batches",
          "elapsed_s", "IOU_mean", "IoU_c", "IoU_f", "gmo_iou_present", "gmo_iou_future",
          "flow_epe_m", "flow_angular_error_rad", "flow_magnitude_error_m", "flow_valid_voxels",
          "ece", "ece_valid_voxels"],
         build_summary_rows(rdir, configs)),
        ("per_class_iou.csv",
         ["config", "class_index", "class_name", "is_free_class", "iou_present", "iou_future"],
         build_per_class_rows(rdir, configs)),
        ("per_horizon_iou.csv",
         ["config", "bucket_index", "horizon_s", "iou_cumulative"],
         build_per_horizon_rows(rdir, configs)),
        ("calibration.csv",
         ["config", "bin_index", "bin_lo", "bin_hi", "mean_confidence", "mean_accuracy",
          "gap_accuracy_minus_confidence", "voxel_count", "voxel_fraction"],
         build_calibration_rows(rdir, configs)),
        ("robustness.csv",
         ["config", "scenario", "lidar_dropout", "cam_dropout", "IoU_c", "IoU_f",
          "IoU_f_degradation_pct", "IoU_f_retained_pct"],
         build_robustness_rows(rdir)),
        ("latency.csv",
         ["config", "device", "hardware", "batch_size", "T_p", "T_o", "latency_mean_ms",
          "latency_std_ms", "latency_p95_ms", "peak_memory_mb", "peak_memory_gb",
          "memory_method", "params_total_m", "params_trainable_m", "gflops", "flops_backend"],
         build_latency_rows(rdir)),
        ("baseline_comparison.csv",
         ["method", "kind", "protocol", "IoU_c_pct", "IoU_f_pct", "measured_here",
          "needs_verification", "source", "note"],
         build_baseline_comparison_rows(rdir, configs, args.protocol)),
        ("model_complexity.csv",
         ["method", "kind", "params_m", "gpu_memory_gb", "gflops", "fps", "source", "note"],
         build_complexity_rows()),
    ]:
        n = _write(out / filename, fields, rows)
        if n:
            written.append(f"{filename} ({n} rows)")

    (out / "README_CSV.md").write_text(
        README_TEMPLATE.format(results_dir=rdir, caveat=COMPARISON_CAVEAT)
    )

    print(f"[csv] wrote {len(written)} files to {out}")
    for line in written:
        print(f"       - {line}")
    print("       - README_CSV.md")

    missing = [n for n, _ in ABLATIONS if not (rdir / f"eval_{n}.json").exists()]
    if missing:
        print(f"[csv] not evaluated yet: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
