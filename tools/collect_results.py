"""Collect the per-config JSONs written by the Slurm grid into markdown tables.

``slurm/04_eval.slurm`` writes one ``eval_<config>.json`` per ablation and
``slurm/05_robustness.slurm`` adds ``robustness_<config>.json`` /
``latency_<config>.json``. This turns whatever is present into the tables that
go straight into README §4.1 and PROBLEM_STATEMENT_AND_EXPERIMENTS §7-§8.

    python tools/collect_results.py --results-dir $DRIFT_RESULTS_DIR
    python tools/collect_results.py --results-dir $DRIFT_RESULTS_DIR --out results.md

Configs with no JSON yet are listed as pending rather than silently dropped, so
a half-finished grid never reads as a complete one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Presentation order and the one-line reading each ablation supports.
ABLATIONS = [
    ("cam4docc_gmo", "Full model"),
    ("no_cmli", "− cross-modal latent imagination"),
    ("no_instance_path", "− instance queries (dense flow)"),
    ("no_uncertainty", "− per-voxel uncertainty head"),
    ("camera_only", "camera only (no LiDAR encoder)"),
    ("lidar_only", "LiDAR only (no camera encoder)"),
    ("fusion_sum", "sum fusion (not gated)"),
]


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _fmt(v: Any, prec: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="Write markdown here instead of stdout.")
    args = p.parse_args(argv)

    rdir = Path(args.results_dir)
    if not rdir.is_dir():
        raise SystemExit(f"results dir not found: {rdir}")

    lines: List[str] = []
    pending: List[str] = []

    # --- ablation / main table -------------------------------------------
    lines.append("## Ablation study (nuScenes val, Cam4DOcc GMO protocol)\n")
    lines.append("| Config | What it removes | mIoU | IoU present | IoU future | Flow EPE (m) | ECE |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, desc in ABLATIONS:
        data = _load(rdir / f"eval_{name}.json")
        if data is None:
            pending.append(f"eval_{name}.json")
            lines.append(f"| `{name}` | {desc} | — | — | — | — | — |")
            continue
        iou = data.get("iou", {})
        flow = data.get("flow", {})
        ece = data.get("ece", {})
        # GMO (class 2) is the class the Cam4DOcc benchmark actually reports.
        pc_p = iou.get("per_class_present") or []
        pc_f = iou.get("per_class_future") or []
        gmo_p = pc_p[2] if len(pc_p) > 2 else None
        gmo_f = pc_f[2] if len(pc_f) > 2 else None
        lines.append(
            f"| `{name}` | {desc} | {_fmt(iou.get('IOU_mean'))} | {_fmt(gmo_p)} | "
            f"{_fmt(gmo_f)} | {_fmt(flow.get('epe'))} | {_fmt(ece.get('ece'))} |"
        )
    lines.append("\n_IoU columns are the GMO class (index 2) — the quantity the Cam4DOcc "
                 "benchmark reports. mIoU averages all non-free classes._\n")

    # --- robustness ------------------------------------------------------
    robust_files = sorted(rdir.glob("robustness_*.json"))
    if robust_files:
        lines.append("\n## Sensor-degradation robustness\n")
        first = _load(robust_files[0]) or {}
        scenarios = [s.get("name", f"s{i}") for i, s in enumerate(first.get("scenarios", []))]
        if scenarios:
            lines.append("| Config | " + " | ".join(scenarios) + " |")
            lines.append("|---" * (len(scenarios) + 1) + "|")
            for f in robust_files:
                data = _load(f) or {}
                cfg_name = data.get("config", f.stem.replace("robustness_", ""))
                cells = []
                for s in data.get("scenarios", []):
                    m = s.get("iou", {}).get("IOU_mean")
                    cells.append(_fmt(m))
                lines.append(f"| `{cfg_name}` | " + " | ".join(cells) + " |")
            lines.append("\n_The gap between `cam4docc_gmo` and `no_cmli` across these columns "
                         "is the cross-modal-imagination claim._\n")
    else:
        pending.append("robustness_*.json")

    # --- latency ---------------------------------------------------------
    lat_files = sorted(rdir.glob("latency_*.json"))
    if lat_files:
        lines.append("\n## Latency / memory (GPU)\n")
        lines.append("| Config | Latency (ms) | Peak memory (GB) | Params (M) |")
        lines.append("|---|---|---|---|")
        for f in lat_files:
            data = _load(f) or {}
            name = data.get("config", f.stem.replace("latency_", ""))
            lat = data.get("latency_ms", {})
            mean = lat.get("mean") if isinstance(lat, dict) else lat
            mem = data.get("peak_memory_gb") or data.get("peak_memory_mb")
            if mem and "peak_memory_mb" in data and "peak_memory_gb" not in data:
                mem = mem / 1024.0
            params = data.get("num_parameters")
            params_m = params / 1e6 if isinstance(params, (int, float)) else None
            lines.append(f"| `{name}` | {_fmt(mean, 2)} | {_fmt(mem, 2)} | {_fmt(params_m, 1)} |")
        lines.append("")
    else:
        pending.append("latency_*.json")

    if pending:
        lines.append("\n> **Incomplete grid.** Still missing: " + ", ".join(f"`{p}`" for p in pending))
        lines.append("> Rows above showing `—` have not been evaluated yet.\n")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[collect] wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
