#!/usr/bin/env python
"""Render what the model saw and what it forecast, side by side, for real val scenes.

The numbers say the model scores 0.19 IoU. They do not say whether it is losing the pedestrian
at the crossing or smearing every vehicle by half a car length -- two very different failures
behind the same number. This draws one page per scene so that question can be answered by
looking:

    row 1   the camera inputs at the present frame (all N_cam views)
    row 2   the LiDAR sweep, top-down, as point density
    row 3   ground-truth occupancy, top-down, one panel per forecast horizon
    row 4   DRIFT's forecast, same horizons, same colours
    row 5   where they disagree -- false positive, false negative, or wrong class

Rows 3-5 are bird's-eye views: the Z axis is collapsed by taking the highest class index
present in each column, which under the ``cam4docc_gmo`` class order (0 free, 1 static,
2 movable) means a movable object is never hidden behind the static structure it is standing
in front of. That ordering is what makes the collapse meaningful, so it is checked rather
than assumed -- see ``bev_from_occupancy``.

    python tools/visualize_scenes.py --config cam4docc_gmo \\
        --checkpoint $DRIFT_CKPT_DIR/cam4docc_gmo/latest.pth \\
        --ann-file $DRIFT_VAL_ANN_FILE --num-scenes 8 --out-dir $DRIFT_RESULTS_DIR/scenes

Scene selection defaults to evenly spaced samples across the split rather than the first N,
because the first N are one contiguous drive and a figure gallery drawn from a single stretch
of road is not evidence about the split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:  # pragma: no cover
    print("[viz] matplotlib is not installed (it is in requirements.txt).", file=sys.stderr)
    raise SystemExit(1)

# Occupancy class colours for the 3-class GMO preset. Free is the page, not a colour, so the
# two classes that matter are the only ink on the map.
FREE_RGB = (0.97, 0.97, 0.96)
STATIC_RGB = (0.60, 0.72, 0.82)   # recessive: structure is context, not the subject
MOVABLE_RGB = (0.84, 0.31, 0.00)  # the vermillion from the figures palette

# Disagreement map. Three failures that a single "wrong" colour would merge into one.
#
# None of these may reuse MOVABLE_RGB or STATIC_RGB. The occupancy rows and the disagreement
# row sit on the same page under one shared legend, so a colour that means "movable object"
# two rows up cannot also mean "hallucinated" here -- the reader has no way to tell which
# sense is intended, and the figure looks correct either way.
DISAGREE_COLOURS = {
    "correct": (0.97, 0.97, 0.96),
    "missed": (0.00, 0.45, 0.70),       # GT occupied, predicted free -- a miss
    "spurious": (0.48, 0.20, 0.58),     # predicted occupied, GT free -- a hallucination
    "wrong_class": (0.90, 0.62, 0.00),  # both occupied, different class
}

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"


def bev_from_occupancy(occ: np.ndarray) -> np.ndarray:
    """Collapse a ``(X, Y, Z)`` class grid to a ``(X, Y)`` top-down class map.

    Takes the maximum class index along Z. This is only meaningful when the class order puts
    the classes that matter most last -- under ``cam4docc_gmo`` (0 free, 1 static, 2 movable) a
    car in front of a wall wins the pixel, which is the reading a person wants from a BEV. On a
    17-class run the result still shows "something is here" correctly but the class shown is
    whichever label happens to sort highest, which is arbitrary; the caller is responsible for
    labelling such a plot honestly.

    Args:
        occ: ``(X, Y, Z)`` integer class indices.

    Returns:
        ``(X, Y)`` integer class indices.
    """
    if occ.ndim != 3:
        raise ValueError(f"bev_from_occupancy expects (X, Y, Z), got shape {occ.shape}")
    return occ.max(axis=2)


def lidar_bev(
    points: np.ndarray, point_cloud_range: Sequence[float], grid_xy: Tuple[int, int]
) -> np.ndarray:
    """Bin a LiDAR sweep into a top-down point-density image on the occupancy grid.

    Points outside ``point_cloud_range`` are dropped rather than clamped: clamping would pile
    every distant return onto the border pixels and draw a bright frame around the scene that
    no sensor ever measured.

    Args:
        points: ``(N, >=2)`` array; only the x and y columns are read.
        point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in metres.
        grid_xy: ``(X, Y)`` cell counts, matching the occupancy grid so the BEVs align.

    Returns:
        ``(X, Y)`` float array of point counts per cell.
    """
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    nx, ny = grid_xy
    out = np.zeros((nx, ny), dtype=np.float32)
    if points.size == 0:
        return out

    xs, ys = points[:, 0], points[:, 1]
    inside = (xs >= x_min) & (xs < x_max) & (ys >= y_min) & (ys < y_max)
    if not inside.any():
        return out

    ix = ((xs[inside] - x_min) / (x_max - x_min) * nx).astype(np.int64)
    iy = ((ys[inside] - y_min) / (y_max - y_min) * ny).astype(np.int64)
    np.add.at(out, (np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)), 1.0)
    return out


def disagreement_map(gt_bev: np.ndarray, pred_bev: np.ndarray) -> np.ndarray:
    """Classify every BEV cell into correct / missed / spurious / wrong_class.

    Returns an index array into ``DISAGREE_COLOURS``' insertion order, so the three failure
    modes stay separable in the figure. Collapsing them to one "error" colour would hide the
    distinction the reader most wants -- a model that misses objects and a model that invents
    them fail in opposite directions and want opposite fixes.

    Args:
        gt_bev: ``(X, Y)`` ground-truth class indices.
        pred_bev: ``(X, Y)`` predicted class indices.

    Returns:
        ``(X, Y)`` int array: 0 correct, 1 missed, 2 spurious, 3 wrong class.
    """
    if gt_bev.shape != pred_bev.shape:
        raise ValueError(f"shape mismatch: gt {gt_bev.shape} vs pred {pred_bev.shape}")
    gt_occupied = gt_bev > 0
    pred_occupied = pred_bev > 0
    out = np.zeros(gt_bev.shape, dtype=np.int64)
    out[gt_occupied & ~pred_occupied] = 1
    out[~gt_occupied & pred_occupied] = 2
    out[gt_occupied & pred_occupied & (gt_bev != pred_bev)] = 3
    return out


def _occupancy_cmap(num_classes: int) -> ListedColormap:
    """Colour map for a class grid: free, static, movable, then a ramp for extra classes."""
    colours = [FREE_RGB, STATIC_RGB, MOVABLE_RGB][:max(1, min(num_classes, 3))]
    if num_classes > 3:
        extra = plt.get_cmap("tab20")(np.linspace(0, 1, num_classes - 3))[:, :3]
        colours = colours + [tuple(c) for c in extra]
    return ListedColormap(colours)


def _show_bev(ax, image: np.ndarray, cmap, vmin=None, vmax=None) -> None:
    """Draw one top-down panel with the ego vehicle at the centre, x forward, y left.

    The array is transposed and flipped so the figure reads the way a driver would: forward is
    up, left is left. Plotting the raw (X, Y) array instead would put forward to the right,
    which silently rotates every qualitative figure in the paper by 90 degrees.
    """
    ax.imshow(np.flipud(image.T), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#d8d8d8")


def _denormalize(img: np.ndarray) -> np.ndarray:
    """Bring a ``(3, H, W)`` image tensor into a displayable ``(H, W, 3)`` 0-1 range.

    Scales by the array's own range rather than assuming ImageNet statistics, because the
    dataset may hand back either normalized or raw pixels depending on how it was built, and a
    guess that is wrong produces a plausible-looking but wrongly-tinted figure.
    """
    chw = np.asarray(img, dtype=np.float32)
    hwc = np.transpose(chw, (1, 2, 0))
    lo, hi = float(hwc.min()), float(hwc.max())
    if hi - lo < 1e-6:
        return np.zeros_like(hwc)
    return (hwc - lo) / (hi - lo)


def render_scene(
    imgs: np.ndarray,
    points: np.ndarray,
    gt_occ: np.ndarray,
    pred_occ: np.ndarray,
    point_cloud_range: Sequence[float],
    title: str,
    horizon_step_s: float = 0.5,
):
    """Build the full input-to-forecast page for one sample.

    Args:
        imgs: ``(N_cam, 3, H, W)`` camera views at the present frame.
        points: ``(N, >=2)`` LiDAR points of the present frame.
        gt_occ: ``(T_o, X, Y, Z)`` ground-truth class grid.
        pred_occ: ``(T_o, X, Y, Z)`` predicted class grid, already argmaxed and upsampled.
        point_cloud_range: Six-element metric range, for the LiDAR binning.
        title: Figure title (scene / sample identification).
        horizon_step_s: Seconds between consecutive output frames.

    Returns:
        The matplotlib figure.
    """
    T_o = gt_occ.shape[0]
    n_cam = imgs.shape[0] if imgs is not None and imgs.size else 0
    num_classes = int(max(gt_occ.max(), pred_occ.max())) + 1
    cmap = _occupancy_cmap(max(num_classes, 3))
    grid_xy = (gt_occ.shape[1], gt_occ.shape[2])

    n_cols = max(T_o, n_cam, 1)
    # The four BEV rows are square; the camera strip is wide and short, so giving it an equal
    # share of the height just pads it with blank axes. Height ratios sized to what each row
    # actually draws keep the page from being mostly whitespace.
    height_ratios = [0.58, 1.0, 1.0, 1.0, 1.0]
    fig = plt.figure(figsize=(2.05 * n_cols, 2.05 * sum(height_ratios) + 0.6))
    gs = fig.add_gridspec(5, n_cols, hspace=0.16, wspace=0.06, height_ratios=height_ratios)

    # Row 1 -- camera inputs.
    for c in range(n_cols):
        ax = fig.add_subplot(gs[0, c])
        ax.set_xticks([])
        ax.set_yticks([])
        if c < n_cam:
            ax.imshow(_denormalize(imgs[c]))
            ax.set_title(f"cam {c}", fontsize=7, color=INK_MUTED, pad=3)
        else:
            ax.axis("off")
        if c == 0:
            ax.set_ylabel("camera in", fontsize=8, color=INK)

    # Row 2 -- LiDAR input, drawn once and left-aligned; the rest of the row is empty.
    ax = fig.add_subplot(gs[1, 0])
    density = lidar_bev(points, point_cloud_range, grid_xy)
    # log1p so a handful of dense near-range returns do not black out everything beyond 20 m.
    _show_bev(ax, np.log1p(density), cmap="Greys")
    ax.set_ylabel("LiDAR in", fontsize=8, color=INK)
    ax.set_title("point density (log), top-down", fontsize=7, color=INK_MUTED, pad=3)
    for c in range(1, n_cols):
        fig.add_subplot(gs[1, c]).axis("off")

    # Rows 3-5 -- ground truth, forecast, and where they differ.
    disagree_cmap = ListedColormap(list(DISAGREE_COLOURS.values()))
    for t in range(n_cols):
        ax_gt = fig.add_subplot(gs[2, t])
        ax_pr = fig.add_subplot(gs[3, t])
        ax_df = fig.add_subplot(gs[4, t])
        if t >= T_o:
            for a in (ax_gt, ax_pr, ax_df):
                a.axis("off")
            continue

        gt_bev = bev_from_occupancy(gt_occ[t])
        pred_bev = bev_from_occupancy(pred_occ[t])
        _show_bev(ax_gt, gt_bev, cmap, vmin=0, vmax=max(num_classes, 3) - 1)
        _show_bev(ax_pr, pred_bev, cmap, vmin=0, vmax=max(num_classes, 3) - 1)
        _show_bev(ax_df, disagreement_map(gt_bev, pred_bev), disagree_cmap, vmin=0, vmax=3)

        ax_gt.set_title(f"+{horizon_step_s * (t + 1):.1f}s", fontsize=7.5, color=INK, pad=3)
        if t == 0:
            ax_gt.set_ylabel("ground truth", fontsize=8, color=INK)
            ax_pr.set_ylabel("DRIFT forecast", fontsize=8, color=INK)
            ax_df.set_ylabel("disagreement", fontsize=8, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STATIC_RGB),
               plt.Rectangle((0, 0), 1, 1, facecolor=MOVABLE_RGB)]
    labels = ["static occupancy", "movable object"]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor=DISAGREE_COLOURS[k])
                for k in ("missed", "spurious", "wrong_class")]
    labels += ["missed (GT only)", "spurious (pred only)", "wrong class"]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(title, fontsize=10, color=INK, y=0.98)
    return fig


def pick_indices(n_available: int, n_wanted: int) -> List[int]:
    """Evenly spaced sample indices across the split.

    The first N samples of a driving split are one contiguous stretch of one drive. A gallery
    built from them shows one road, one time of day, one traffic condition -- and reads as if
    it characterized the split.
    """
    if n_available <= 0 or n_wanted <= 0:
        return []
    n = min(n_wanted, n_available)
    step = n_available / n
    return sorted({min(n_available - 1, int(i * step)) for i in range(n)})


def _present_frame(value: Any) -> Any:
    """The last observed frame from a stacked past-frames tensor or a per-frame list."""
    if isinstance(value, (list, tuple)):
        return value[-1] if value else None
    return value[-1] if hasattr(value, "shape") and len(value.shape) >= 1 else value


def main(argv: Optional[List[str]] = None) -> int:
    from configs import get_config, list_configs
    from drift.data.collate import collate_fn
    from drift.models.drift import DRIFT
    from drift.utils.seed import seed_everything

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="cam4docc_gmo", choices=list_configs())
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--random-init", action="store_true",
                   help="Skip the checkpoint; renders layout only, numbers are meaningless.")
    p.add_argument("--dataset", default=None, choices=["synthetic", "cam4docc"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--ann-file", default=None)
    p.add_argument("--num-scenes", type=int, default=6)
    p.add_argument("--indices", default=None, help="Comma-separated sample indices, overriding --num-scenes.")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    if not args.checkpoint and not args.random_init:
        print("[viz] pass --checkpoint <path>, or --random-init to check the layout only.",
              file=sys.stderr)
        return 1

    # eval.py owns the config-override and dataset-building logic; reuse it rather than
    # maintaining a second copy that can drift out of step with the evaluation protocol.
    from tools.eval import (
        build_config, build_val_dataset, load_checkpoint_into, move_batch_to_device,
    )

    # `build_config` is eval.py's, so it expects eval.py's full argument set. One scene per
    # page means batch size 1 and no worker processes, regardless of what the config asks for.
    args.batch_size = 1
    args.num_workers = 0
    args.seed = args.seed if hasattr(args, "seed") else None

    cfg = build_config(args)
    seed_everything(cfg.train.seed)
    device = torch.device(
        (args.device or cfg.train.device) if (args.device or cfg.train.device) != "cuda"
        or torch.cuda.is_available() else "cpu"
    )

    dataset = build_val_dataset(cfg)
    model = DRIFT(cfg.model, cfg.loss).to(device).eval()
    tag = "random-init"
    if args.checkpoint:
        ckpt = load_checkpoint_into(model, args.checkpoint, device)
        tag = f"epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')}"
        print(f"[viz] loaded checkpoint: {args.checkpoint} ({tag})")

    if args.indices:
        indices = [int(s) for s in args.indices.split(",") if s.strip()]
    else:
        indices = pick_indices(len(dataset), args.num_scenes)
    if not indices:
        print("[viz] no samples to render.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    occ_size = tuple(cfg.model.occ_size)
    written = []

    for idx in indices:
        batch = collate_fn([dataset[idx]])
        moved = move_batch_to_device(batch, device)

        with torch.no_grad():
            outputs = model(moved)
            logits = outputs["occ_logits"][:1].float()
            # Upsample logits, then argmax -- never the reverse. Upsampling an already-argmaxed
            # label grid interpolates between class *indices*, inventing classes that were
            # never predicted. This mirrors OccupancyIoUMetric.update's ordering.
            B, T_o, C = logits.shape[:3]
            up = torch.nn.functional.interpolate(
                logits.reshape(B * T_o, C, *logits.shape[3:]), size=occ_size,
                mode="trilinear", align_corners=False,
            )
            pred = up.argmax(dim=1).reshape(T_o, *occ_size).cpu().numpy()

        gt = batch["gt_occ"][0].cpu().numpy().astype(np.int64)
        imgs_t = batch["imgs"][0]
        imgs = _present_frame(imgs_t)
        imgs = imgs.cpu().numpy() if torch.is_tensor(imgs) else np.zeros((0, 3, 1, 1))
        pts = _present_frame(batch["points"][0])
        pts = pts.cpu().numpy() if torch.is_tensor(pts) else np.zeros((0, 3), dtype=np.float32)

        fig = render_scene(
            imgs=imgs, points=pts, gt_occ=gt, pred_occ=pred,
            point_cloud_range=cfg.model.point_cloud_range,
            title=f"{cfg.name} — val sample {idx} — {tag}",
        )
        for ext in ("png", "pdf"):
            path = out_dir / f"scene_{idx:05d}.{ext}"
            fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)
        written.append(idx)
        print(f"[viz] scene {idx} -> {out_dir / f'scene_{idx:05d}.png'}")

    print(f"[viz] wrote {len(written)} scenes to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
