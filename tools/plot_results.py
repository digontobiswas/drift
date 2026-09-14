#!/usr/bin/env python
"""Draw the paper's quantitative figures from the CSVs, so no number is retyped into a plot.

Reads whatever ``tools/results_to_csv.py`` produced and writes one PNG (for looking at) and
one PDF (vector, for the paper) per figure. Figures whose input CSV is absent are skipped with
a line saying so, so a half-finished grid yields the figures it can rather than an error.

    python tools/results_to_csv.py --results-dir $DRIFT_RESULTS_DIR
    python tools/plot_results.py   --csv-dir     $DRIFT_RESULTS_DIR/csv

Figures
-------
    fig_ablation                 what each component is worth (IoU panel + flow-EPE panel)
    fig_horizon_degradation      accuracy against forecast horizon -- the forecasting claim
    fig_per_class_iou            which classes the model actually gets right
    fig_calibration              reliability diagram behind the ECE number
    fig_robustness               graceful degradation under sensor dropout -- the CMLI claim
    fig_pareto                   accuracy against latency
    fig_baseline_comparison      DRIFT beside the published Cam4DOcc numbers

Design notes that are not taste
-------------------------------
The categorical palette is fixed and assigned in order, never cycled: a config keeps its
colour no matter which other configs are in the plot, so two figures in the same paper cannot
disagree about which line is ``no_cmli``. It is colourblind-safe (validated: worst adjacent
deutan Delta-E 11.0). Because three of the hues sit below 3:1 contrast on white, every figure
also carries a legend and, where there are few enough series, direct labels -- identity is
never carried by colour alone.

Measured and published numbers are never drawn as if they were the same kind of thing:
published bars are hatched and labelled in the legend as transcribed rather than re-run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")  # no display on a compute node
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised only on an env without matplotlib
    print(
        "[plot] matplotlib is not installed in this environment.\n"
        "       pip install matplotlib   (it is in requirements.txt)",
        file=sys.stderr,
    )
    raise SystemExit(1)

from tools.published_baselines import COMPARISON_CAVEAT  # noqa: E402

# Fixed categorical order. Index 0 is always the full model, so it is always the same blue.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#7B3294", "#56B4E9", "#A6761D"]

# Text and furniture never wear a series colour.
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#e4e4e4"

CONFIG_ORDER = [
    "cam4docc_gmo", "no_cmli", "no_instance_path", "no_uncertainty",
    "camera_only", "lidar_only", "fusion_sum",
]
SHORT_LABEL = {
    "cam4docc_gmo": "full", "no_cmli": "-CMLI", "no_instance_path": "-instance",
    "no_uncertainty": "-uncert.", "camera_only": "cam only", "lidar_only": "LiDAR only",
    "fusion_sum": "sum fusion",
}
# Sensor-availability scenarios, ordered worst-to-best along the degradation axis.
SCENARIO_ORDER = ["clean", "moderate_dropout", "severe_dropout", "lidar_only", "camera_only"]


def _style() -> None:
    """Recessive furniture: the data should be the darkest thing on the page."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8,
    })


def _read(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    """Parse one CSV cell as a float, treating blank and non-numeric as missing.

    Blank cells are deliberate in these CSVs -- `gmo_iou_present` is empty when the run was
    not the GMO preset -- so a blank must become None (a gap in the plot) and never 0.0 (a
    measured result of zero).
    """
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _colour_for(config: str) -> str:
    """A config's colour, fixed by its position in CONFIG_ORDER rather than by plot order."""
    if config in CONFIG_ORDER:
        return PALETTE[CONFIG_ORDER.index(config) % len(PALETTE)]
    return PALETTE[-1]


def _label_for(config: str) -> str:
    return SHORT_LABEL.get(config, config)


def _ordered(rows: Sequence[Dict[str, str]], key: str = "config") -> List[str]:
    """Distinct values of `key`, in CONFIG_ORDER first, then anything unexpected."""
    seen = list(dict.fromkeys(r[key] for r in rows))
    known = [c for c in CONFIG_ORDER if c in seen]
    return known + [c for c in seen if c not in CONFIG_ORDER]


def _save(fig, out_dir: Path, name: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- figures


def plot_ablation(csv_dir: Path, out_dir: Path) -> Optional[List[Path]]:
    """Two panels rather than two y-axes on one plot.

    IoU (a ratio) and flow EPE (metres) share no scale, and drawing them against twin axes
    would let the relative heights of the two measures be set by the axis limits rather than
    by the data. Separate panels make the comparison within each measure and refuse the
    comparison between them, which is the honest reading.
    """
    rows = _read(csv_dir / "summary.csv")
    if not rows:
        return None
    configs = _ordered(rows)
    by_config = {r["config"]: r for r in rows}

    fig, (ax_iou, ax_flow) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = range(len(configs))
    width = 0.38

    for offset, (key, label, alpha) in enumerate([
        ("IoU_c", "IoU$_c$ (nearest horizon)", 1.0),
        ("IoU_f", "IoU$_f$ (future pooled)", 0.55),
    ]):
        values = [_f(by_config[c], key) for c in configs]
        ax_iou.bar(
            [i + (offset - 0.5) * width for i in x],
            [v if v is not None else 0.0 for v in values],
            width, label=label, alpha=alpha,
            color=[_colour_for(c) for c in configs], edgecolor="white", linewidth=1.2,
        )
        for i, v in zip(x, values):
            if v is not None:
                ax_iou.text(i + (offset - 0.5) * width, v, f"{v:.3f}", ha="center",
                            va="bottom", fontsize=6.5, color=INK)

    ax_iou.set_ylabel("IoU")
    ax_iou.set_title("Occupancy accuracy per ablation", loc="left")
    # Above the axes rather than inside them: an in-axes legend lands on top of the tallest
    # bars, which are the full model's -- the row a reader looks at first.
    ax_iou.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)

    epe = [_f(by_config[c], "flow_epe_m") for c in configs]
    ax_flow.bar(list(x), [v if v is not None else 0.0 for v in epe], 0.62,
                color=[_colour_for(c) for c in configs], edgecolor="white", linewidth=1.2)
    for i, v in zip(x, epe):
        if v is not None:
            ax_flow.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax_flow.set_ylabel("Flow EPE (m)")
    ax_flow.set_title("Flow error per ablation (lower is better)", loc="left")

    for ax in (ax_iou, ax_flow):
        ax.set_xticks(list(x))
        ax.set_xticklabels([_label_for(c) for c in configs], rotation=30, ha="right")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    return _save(fig, out_dir, "fig_ablation")


def plot_horizon_degradation(csv_dir: Path, out_dir: Path) -> Optional[List[Path]]:
    """IoU against forecast horizon: the figure the whole forecasting claim rests on."""
    rows = _read(csv_dir / "per_horizon_iou.csv")
    if not rows:
        return None
    configs = _ordered(rows)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    # Direct labels only while there are few enough lines for them to stay legible; past that
    # the label column becomes its own thicket and the legend does the job alone.
    direct_label = len(configs) <= 4
    for config in configs:
        pts = sorted(
            ((_f(r, "horizon_s"), _f(r, "iou_cumulative")) for r in rows if r["config"] == config),
            key=lambda p: (p[0] if p[0] is not None else 0.0),
        )
        xs = [p[0] for p in pts if p[0] is not None and p[1] is not None]
        ys = [p[1] for p in pts if p[0] is not None and p[1] is not None]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=2.0,
                color=_colour_for(config), label=_label_for(config))
        if direct_label:
            # Identity should not rest on the legend alone when there is room to say it twice.
            ax.annotate(_label_for(config), (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=7.5, color=INK)

    ax.set_xlabel("Forecast horizon (s)")
    ax.set_ylabel("Cumulative IoU (classes 1:)")
    ax.set_title("Accuracy degrades with horizon", loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.margins(x=0.16 if direct_label else 0.02)
    if len(configs) > 1:
        ax.legend(loc="upper right", ncol=2)
    return _save(fig, out_dir, "fig_horizon_degradation")


def plot_per_class_iou(csv_dir: Path, out_dir: Path, config: Optional[str] = None) -> Optional[List[Path]]:
    """Per-class IoU for one config, free class excluded.

    Class 0 is free space, which dominates every voxel grid and scores high for reasons that
    have nothing to do with forecasting quality; including it would compress every class that
    matters into the left edge of the plot.
    """
    rows = _read(csv_dir / "per_class_iou.csv")
    if not rows:
        return None
    configs = _ordered(rows)
    config = config or (configs[0] if configs else None)
    subset = [r for r in rows if r["config"] == config and r.get("is_free_class") != "True"]
    if not subset:
        return None

    labels = [r["class_name"] for r in subset]
    present = [_f(r, "iou_present") or 0.0 for r in subset]
    future = [_f(r, "iou_future") or 0.0 for r in subset]

    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(labels) + 1.9))
    y = range(len(labels))
    h = 0.36
    ax.barh([i + h / 2 for i in y], present, h, color=PALETTE[0], edgecolor="white",
            linewidth=1.2, label="present bucket (IoU$_c$)")
    ax.barh([i - h / 2 for i in y], future, h, color=PALETTE[1], edgecolor="white",
            linewidth=1.2, label="future pooled (IoU$_f$)")
    for i, (p, f_) in enumerate(zip(present, future)):
        ax.text(p, i + h / 2, f" {p:.3f}", va="center", fontsize=6.5, color=INK)
        ax.text(f_, i - h / 2, f" {f_:.3f}", va="center", fontsize=6.5, color=INK)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("IoU")
    ax.set_title(f"Per-class IoU — {config} (free class excluded)", loc="left")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.margins(x=0.14)
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "fig_per_class_iou")


def plot_calibration(csv_dir: Path, out_dir: Path) -> Optional[List[Path]]:
    """Reliability diagram: predicted confidence against observed accuracy.

    The diagonal is perfect calibration. Bars below it are overconfident, above it
    underconfident. Bin opacity carries how much data is in each bin, because a bin holding
    0.1% of the voxels deviating wildly means far less than the bin holding half of them.
    """
    rows = _read(csv_dir / "calibration.csv")
    if not rows:
        return None
    configs = _ordered(rows)
    config = configs[0]
    subset = sorted((r for r in rows if r["config"] == config),
                    key=lambda r: int(r["bin_index"]))
    if not subset:
        return None

    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), linewidth=1.2, color=INK_MUTED,
            label="perfect calibration")

    width = 1.0 / len(subset)
    for r in subset:
        conf = _f(r, "mean_confidence")
        acc = _f(r, "mean_accuracy")
        frac = _f(r, "voxel_fraction") or 0.0
        if conf is None or acc is None:
            continue
        lo = _f(r, "bin_lo") or 0.0
        ax.bar(lo + width / 2, acc, width * 0.92, color=PALETTE[0],
               alpha=0.25 + 0.75 * min(frac * len(subset), 1.0),
               edgecolor="white", linewidth=1.0)
        ax.plot([conf], [acc], marker="o", markersize=5, color=PALETTE[1], zorder=3)

    ax.plot([], [], marker="o", linestyle="none", markersize=5, color=PALETTE[1],
            label="bin (confidence, accuracy)")
    ax.bar([], [], color=PALETTE[0], alpha=0.7, label="observed accuracy (opacity = voxel share)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram — {config}", loc="left")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    return _save(fig, out_dir, "fig_calibration")


def plot_robustness(csv_dir: Path, out_dir: Path) -> Optional[List[Path]]:
    """IoU_f across sensor-availability scenarios -- the slope, not the level, is the claim.

    The CMLI argument is not that cross-modal imagination raises clean-data accuracy; it is
    that it flattens this line. Drawing the full model and ``no_cmli`` on one axis is what
    makes that testable by eye.
    """
    rows = _read(csv_dir / "robustness.csv")
    if not rows:
        return None
    configs = _ordered(rows)
    present = [s for s in SCENARIO_ORDER if any(r["scenario"] == s for r in rows)]
    present += [r["scenario"] for r in rows if r["scenario"] not in SCENARIO_ORDER]
    scenarios = list(dict.fromkeys(present))

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for config in configs:
        by_scenario = {r["scenario"]: r for r in rows if r["config"] == config}
        xs, ys = [], []
        for i, s in enumerate(scenarios):
            v = _f(by_scenario[s], "IoU_f") if s in by_scenario else None
            if v is not None:
                xs.append(i)
                ys.append(v)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2.0,
                color=_colour_for(config), label=_label_for(config))

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios])
    ax.set_ylabel("IoU$_f$")
    ax.set_title("Degradation under sensor dropout", loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    return _save(fig, out_dir, "fig_robustness")


def plot_pareto(csv_dir: Path, out_dir: Path) -> Optional[List[Path]]:
    """Accuracy against latency, one point per config that has both measured."""
    summary = {r["config"]: r for r in _read(csv_dir / "summary.csv")}
    latency = {r["config"]: r for r in _read(csv_dir / "latency.csv")}
    shared = [c for c in _ordered([{"config": c} for c in summary]) if c in latency]
    if not shared:
        return None

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for config in shared:
        ms = _f(latency[config], "latency_mean_ms")
        iou = _f(summary[config], "IoU_f")
        if ms is None or iou is None:
            continue
        ax.scatter([ms], [iou], s=70, color=_colour_for(config), zorder=3,
                   edgecolor="white", linewidth=1.4, label=_label_for(config))
        ax.annotate(_label_for(config), (ms, iou), textcoords="offset points",
                    xytext=(8, 3), fontsize=7.5, color=INK)

    ax.set_xlabel("Latency (ms, batch size 1)")
    ax.set_ylabel("IoU$_f$")
    ax.set_title("Accuracy against cost", loc="left")
    ax.grid(True)
    ax.set_axisbelow(True)
    return _save(fig, out_dir, "fig_pareto")


def plot_baseline_comparison(csv_dir: Path, out_dir: Path) -> Optional[List[Path]]:
    """DRIFT beside the published numbers, with the published bars visibly marked as such.

    Hatching is not decoration here: a transcribed number and a number measured on this
    cluster are different kinds of evidence, and a bar chart that renders them identically
    invites the reader to treat the gap as a controlled result. The caveat about IoU_c
    meaning different things in the two protocols is written under the axes, not left to a
    caption someone may not copy across.
    """
    rows = _read(csv_dir / "baseline_comparison.csv")
    if not rows:
        return None

    published = [r for r in rows if r.get("measured_here") != "True"]
    measured = [r for r in rows if r.get("measured_here") == "True"]
    ordered = published + measured
    labels = [r["method"] for r in ordered]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = range(len(ordered))
    width = 0.38

    for offset, (key, label) in enumerate([("IoU_c_pct", "IoU$_c$"), ("IoU_f_pct", "IoU$_f$")]):
        for i, r in enumerate(ordered):
            v = _f(r, key)
            if v is None:
                continue
            is_measured = r.get("measured_here") == "True"
            ax.bar(
                i + (offset - 0.5) * width, v, width,
                color=PALETTE[0] if is_measured else INK_MUTED,
                alpha=1.0 if offset == 0 else 0.55,
                hatch="" if is_measured else "///",
                edgecolor="white", linewidth=1.2,
            )
            ax.text(i + (offset - 0.5) * width, v, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
        ax.bar([], [], color=INK_MUTED if offset else PALETTE[0], label=label, alpha=1.0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[0], edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=INK_MUTED, edgecolor="white", hatch="///"),
    ]
    ax.legend(handles, ["measured here (tools/eval.py)", "published (transcribed, not re-run)"],
              loc="upper left")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_title("Cam4DOcc inflated-GMO protocol", loc="left")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.text(0.0, -0.13, _wrap(COMPARISON_CAVEAT, 108), fontsize=6.6, color=INK_MUTED, va="top")
    return _save(fig, out_dir, "fig_baseline_comparison")


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


FIGURES = [
    ("ablation", plot_ablation),
    ("horizon_degradation", plot_horizon_degradation),
    ("per_class_iou", plot_per_class_iou),
    ("calibration", plot_calibration),
    ("robustness", plot_robustness),
    ("pareto", plot_pareto),
    ("baseline_comparison", plot_baseline_comparison),
]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv-dir", required=True, help="Directory written by tools/results_to_csv.py.")
    p.add_argument("--out-dir", default=None, help="Where to write figures (default: <csv-dir>/../figures).")
    p.add_argument("--only", default=None, help="Comma-separated figure names to draw.")
    args = p.parse_args(argv)

    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_dir():
        print(f"[plot] csv dir not found: {csv_dir} -- run tools/results_to_csv.py first",
              file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else csv_dir.parent / "figures"
    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    _style()

    drawn, skipped = [], []
    for name, fn in FIGURES:
        if wanted and name not in wanted:
            continue
        paths = fn(csv_dir, out_dir)
        (drawn if paths else skipped).append(name)

    print(f"[plot] wrote {len(drawn)} figures to {out_dir}")
    for name in drawn:
        print(f"       - fig_{name}.png / .pdf")
    if skipped:
        print(f"[plot] no data yet for: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
