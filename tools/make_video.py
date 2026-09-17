#!/usr/bin/env python
"""Animate the forecast, either as it unrolls in time or as the drive progresses.

`tools/visualize_scenes.py` renders one page per scene, deliberately sampling scenes spread
across the whole val split. Those pages are the paper figures, and they are exactly the wrong
thing to stitch into a video: consecutive pages are unrelated pieces of road, so the result is
a slideshow that looks like a tracking failure.

A video has to be built from frames that genuinely belong in sequence, and there are two
sequences worth watching:

    --mode rollout    one sample, animated across the forecast horizons (+0.5s ... +3.0s).
                      This is the 4D claim itself: the model is asked about a future it has
                      not seen, and the animation shows how the prediction holds up as the
                      horizon stretches. Ground truth and forecast run side by side, with the
                      disagreement map underneath.

    --mode temporal   consecutive keyframes of ONE drive, animated as the vehicle moves, each
                      frame showing the nearest-horizon forecast. This is what a person means
                      by "does it work" -- the world moving, and the model keeping up.

Neither belongs in a conference PDF, which cannot hold video. They are for the supplementary
material, the project page, and the talk.

    python tools/make_video.py --mode rollout --config cam4docc_gmo \\
        --checkpoint $DRIFT_CKPT_DIR/cam4docc_gmo/latest.pth.bak \\
        --ann-file $DRIFT_VAL_ANN_FILE --index 0 --out-dir $DRIFT_RESULTS_DIR/video

    python tools/make_video.py --mode temporal --config cam4docc_gmo \\
        --checkpoint $DRIFT_CKPT_DIR/cam4docc_gmo/latest.pth.bak \\
        --ann-file $DRIFT_VAL_ANN_FILE --index 0 --max-frames 40 \\
        --out-dir $DRIFT_RESULTS_DIR/video

Output is a GIF, written with Pillow, which is already a dependency -- no ffmpeg needed. An
MP4 is written as well when ffmpeg happens to be available, since a GIF of forty frames is
large and MP4 is what a submission site will want.

On finding a run of consecutive frames
-------------------------------------
Index order is not time order. ``tools/prepare_nuscenes.py`` shards the sample list with
``sample_tokens[shard::num_shards]``, so a merged annotation file interleaves shards: entry n
and entry n+1 are usually seconds apart in a drive, or in different drives altogether. Walking
the index and stopping at the first discontinuity therefore yields exactly one frame.

The annotation file has no scene id either, but every sample's LiDAR filename carries the
nuScenes log prefix and a microsecond timestamp
(``n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin``). That is enough to
reconstruct the truth: group the samples by drive, sort each group by capture time, and take
the longest stretch whose consecutive gaps are about the 0.5 s keyframe interval. The animation
then covers real consecutive keyframes and stops at the end of the drive, wherever those
samples happen to sit in the file.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
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
    from PIL import Image
except ImportError:  # pragma: no cover
    print("[video] needs matplotlib and Pillow (both in requirements.txt).", file=sys.stderr)
    raise SystemExit(1)

from tools.visualize_scenes import (  # noqa: E402
    DISAGREE_COLOURS,
    _denormalize,
    _occupancy_cmap,
    _show_bev,
    bev_from_occupancy,
    disagreement_map,
    legend_handles,
)

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"

# nuScenes LiDAR filenames: <log prefix>__LIDAR_TOP__<microsecond timestamp>.pcd.bin
_LIDAR_NAME = re.compile(r"^(?P<drive>.+?)__LIDAR_TOP__(?P<ts>\d+)\.pcd(?:\.bin)?$")

# Keyframes are 2 Hz. Allow a generous margin -- a dropped keyframe should not be read as a
# new drive, but a jump of several seconds should.
KEYFRAME_INTERVAL_US = 500_000
MAX_GAP_US = 1_500_000


def drive_and_timestamp(lidar_path: str) -> Optional[Tuple[str, int]]:
    """Pull the drive id and capture timestamp out of a nuScenes LiDAR filename.

    Args:
        lidar_path: Path to a ``.pcd.bin`` sweep, absolute or relative.

    Returns:
        ``(drive_id, timestamp_microseconds)``, or ``None`` if the name does not follow the
        nuScenes convention -- which is the case for synthetic data, and is not an error.
    """
    match = _LIDAR_NAME.match(Path(lidar_path).name)
    if match is None:
        return None
    return match.group("drive"), int(match.group("ts"))


def temporal_run(
    entries: Sequence[Dict[str, Any]], max_frames: int, start: Optional[int] = None
) -> List[int]:
    """Indices of a genuinely consecutive stretch of one drive, returned in time order.

    Index order is not time order, and assuming it was is what produced a one-frame
    "animation". ``tools/prepare_nuscenes.py`` shards the sample list with
    ``sample_tokens[shard::num_shards]``, so a merged annotation file interleaves shards: entry
    n and entry n+1 are typically several seconds apart within a drive, or in different drives
    altogether. Walking the index directly finds no neighbours at all -- which the boundary
    check correctly reported rather than splicing unrelated frames together.

    The run is therefore recovered from the timestamps: group every sample by its drive, sort
    each group by capture time, and take the longest stretch whose consecutive gaps are within
    a keyframe interval. Where those samples happen to sit in the annotation file does not
    matter.

    Args:
        entries: The dataset's annotation entries.
        max_frames: Cap on the number of frames returned.
        start: Optional index to anchor on. When given and that sample belongs to a run, the
            run is taken from there onward; otherwise the longest run in the split is used.

    Returns:
        Dataset indices in ascending time order. Falls back to a plain consecutive range when
        filenames carry no timestamps (synthetic data), where there are no drives to respect.
    """
    stamped: List[Tuple[str, int, int]] = []
    for i, entry in enumerate(entries):
        paths = entry.get("points_paths") or []
        got = drive_and_timestamp(paths[-1]) if paths else None
        if got is not None:
            stamped.append((got[0], got[1], i))

    if not stamped:
        begin = start or 0
        return list(range(begin, min(begin + max_frames, len(entries))))

    by_drive: Dict[str, List[Tuple[int, int]]] = {}
    for drive, ts, i in stamped:
        by_drive.setdefault(drive, []).append((ts, i))

    runs: List[List[int]] = []
    for frames in by_drive.values():
        frames.sort()
        current = [frames[0][1]]
        for (prev_ts, _), (ts, idx) in zip(frames, frames[1:]):
            if 0 < ts - prev_ts <= MAX_GAP_US:
                current.append(idx)
            else:
                runs.append(current)
                current = [idx]
        runs.append(current)

    if start is not None:
        for run in runs:
            if start in run:
                return run[run.index(start):][:max_frames]
    return max(runs, key=len)[:max_frames]


def contiguous_run(entries: Sequence[Dict[str, Any]], start: int, max_frames: int) -> List[int]:
    """Indices from ``start`` that belong to the same continuous drive.

    Stops at the first sample whose drive id differs or whose timestamp jumps -- the end of
    the scene. Without this the animation would cut from one city to another mid-playback
    while still looking like continuous footage, which reads as the model teleporting.

    Args:
        entries: The dataset's annotation entries, in file order.
        start: First index to include.
        max_frames: Hard cap on the number of frames.

    Returns:
        Consecutive indices, always including ``start``, never crossing a scene boundary.
        Falls back to a plain consecutive range when filenames carry no timestamp (synthetic
        data), since there are no scenes to cross.
    """
    if start < 0 or start >= len(entries):
        return []

    def stamp(i: int) -> Optional[Tuple[str, int]]:
        paths = entries[i].get("points_paths") or []
        return drive_and_timestamp(paths[-1]) if paths else None

    first = stamp(start)
    if first is None:
        return list(range(start, min(start + max_frames, len(entries))))

    run = [start]
    prev_drive, prev_ts = first
    for i in range(start + 1, len(entries)):
        if len(run) >= max_frames:
            break
        here = stamp(i)
        if here is None:
            break
        drive, ts = here
        if drive != prev_drive or not (0 < ts - prev_ts <= MAX_GAP_US):
            break
        run.append(i)
        prev_drive, prev_ts = drive, ts
    return run


def _frame_to_image(fig) -> Image.Image:
    """Rasterize one figure at a fixed size.

    Deliberately not ``bbox_inches="tight"``: that trims to the drawn content, so frames come
    out at slightly different sizes and the animation jitters or is rejected outright by the
    encoder. Video frames must all be identical dimensions.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_rollout_frame(
    gt_occ: np.ndarray, pred_occ: np.ndarray, t: int, horizon_step_s: float, title: str
):
    """One horizon of the forecast: ground truth, prediction, and where they part."""
    num_classes = max(int(max(gt_occ.max(), pred_occ.max())) + 1, 3)
    cmap = _occupancy_cmap(num_classes)
    gt_bev = bev_from_occupancy(gt_occ[t])
    pred_bev = bev_from_occupancy(pred_occ[t])

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    _show_bev(axes[0], gt_bev, cmap, vmin=0, vmax=num_classes - 1)
    _show_bev(axes[1], pred_bev, cmap, vmin=0, vmax=num_classes - 1)
    _show_bev(
        axes[2], disagreement_map(gt_bev, pred_bev),
        matplotlib.colors.ListedColormap(list(DISAGREE_COLOURS.values())), vmin=0, vmax=3,
    )
    for ax, label in zip(axes, ["ground truth", "DRIFT forecast", "disagreement"]):
        ax.set_title(label, fontsize=9, color=INK, pad=4)

    handles, labels = legend_handles(num_classes)
    # Fixed figure coordinates, never `tight` -- the legend must not change the frame size
    # from one horizon to the next (see _frame_to_image).
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{title}    horizon +{horizon_step_s * (t + 1):.1f}s", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.13, wspace=0.05)
    return fig


def render_rollout_figure(
    gt_occ: np.ndarray, pred_occ: np.ndarray, horizon_step_s: float, title: str
):
    """The paper figure: every horizon at once, ground truth over forecast over disagreement.

    The animation is for a talk; this is what goes in a PDF. Horizons run left to right as
    columns so a reader can follow one object across the forecast, and the three rows stack
    vertically so the comparison at any single horizon is a straight downward glance. This is
    the "BEV occupancy rollout" figure specified in
    ``docs/PROBLEM_STATEMENT_AND_EXPERIMENTS.md`` section 10.

    Args:
        gt_occ: ``(T_o, X, Y, Z)`` ground-truth class grid.
        pred_occ: ``(T_o, X, Y, Z)`` predicted class grid, same shape.
        horizon_step_s: Seconds between consecutive output frames.
        title: Figure title identifying config, sample and checkpoint.

    Returns:
        The matplotlib figure.
    """
    if gt_occ.shape != pred_occ.shape:
        raise ValueError(f"shape mismatch: gt {gt_occ.shape} vs pred {pred_occ.shape}")
    T_o = gt_occ.shape[0]
    num_classes = max(int(max(gt_occ.max(), pred_occ.max())) + 1, 3)
    cmap = _occupancy_cmap(num_classes)
    disagree_cmap = matplotlib.colors.ListedColormap(list(DISAGREE_COLOURS.values()))

    fig, axes = plt.subplots(3, T_o, figsize=(1.85 * T_o, 6.0), squeeze=False)
    for t in range(T_o):
        gt_bev = bev_from_occupancy(gt_occ[t])
        pred_bev = bev_from_occupancy(pred_occ[t])
        _show_bev(axes[0][t], gt_bev, cmap, vmin=0, vmax=num_classes - 1)
        _show_bev(axes[1][t], pred_bev, cmap, vmin=0, vmax=num_classes - 1)
        _show_bev(axes[2][t], disagreement_map(gt_bev, pred_bev), disagree_cmap, vmin=0, vmax=3)
        axes[0][t].set_title(f"+{horizon_step_s * (t + 1):.1f}s", fontsize=9, color=INK, pad=4)

    for row, label in enumerate(["ground truth", "DRIFT forecast", "disagreement"]):
        axes[row][0].set_ylabel(label, fontsize=9, color=INK)

    handles, labels = legend_handles(num_classes)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=10, color=INK, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.08, wspace=0.05, hspace=0.06)
    return fig


def render_temporal_frame(
    imgs: np.ndarray, gt_occ: np.ndarray, pred_occ: np.ndarray, horizon: int,
    frame_no: int, n_frames: int, horizon_step_s: float, title: str,
):
    """One moment of the drive: what the front camera sees, and the nearest-horizon forecast."""
    num_classes = max(int(max(gt_occ.max(), pred_occ.max())) + 1, 3)
    cmap = _occupancy_cmap(num_classes)
    gt_bev = bev_from_occupancy(gt_occ[horizon])
    pred_bev = bev_from_occupancy(pred_occ[horizon])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    if imgs is not None and imgs.size:
        axes[0].imshow(_denormalize(imgs[0]))  # camera 0 is CAM_FRONT
        axes[0].set_title("front camera", fontsize=9, color=INK, pad=4)
    else:
        axes[0].axis("off")

    _show_bev(axes[1], gt_bev, cmap, vmin=0, vmax=num_classes - 1)
    _show_bev(axes[2], pred_bev, cmap, vmin=0, vmax=num_classes - 1)
    axes[1].set_title("ground truth", fontsize=9, color=INK, pad=4)
    axes[2].set_title(f"DRIFT forecast (+{horizon_step_s * (horizon + 1):.1f}s)",
                      fontsize=9, color=INK, pad=4)

    # Occupancy classes only: this layout draws no disagreement panel, and a legend entry for
    # a colour that never appears in the frame sends the reader looking for it.
    handles, labels = legend_handles(num_classes, include_disagreement=False)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{title}    frame {frame_no + 1}/{n_frames}"
                 f"    t = {frame_no * horizon_step_s:.1f}s", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.12, wspace=0.08)
    return fig


def write_gif(frames: List[Image.Image], path: Path, fps: float) -> None:
    """Write the frames as a looping GIF via Pillow (no ffmpeg required)."""
    if not frames:
        raise ValueError("write_gif: no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=int(round(1000.0 / max(fps, 0.1))), loop=0, optimize=True,
    )


def write_mp4(frames: List[Image.Image], path: Path, fps: float) -> bool:
    """Write an MP4 if ffmpeg is on PATH; report whether it happened.

    A forty-frame GIF is tens of megabytes and most submission sites want MP4, but ffmpeg is
    not guaranteed on a compute node, so this is best-effort and never fatal: the GIF is
    already written by the time this runs.
    """
    if not frames:
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps}",
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(path)],
            input=b"".join(_png_bytes(f) for f in frames),
            capture_output=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        print(f"[video] ffmpeg failed, GIF still written: "
              f"{proc.stderr.decode('utf-8', 'replace')[:300]}", file=sys.stderr)
        return False
    return True


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _predict(model, batch, device, occ_size) -> np.ndarray:
    """Run the model and return ``(T_o, X, Y, Z)`` predicted class indices at full resolution.

    Upsamples the logits and only then takes the argmax, mirroring
    ``OccupancyIoUMetric.update``. Upsampling an already-argmaxed label grid interpolates
    between class *indices* and invents classes the model never predicted.
    """
    from tools.eval import move_batch_to_device

    with torch.no_grad():
        outputs = model(move_batch_to_device(batch, device))
        logits = outputs["occ_logits"][:1].float()
        B, T_o, C = logits.shape[:3]
        up = torch.nn.functional.interpolate(
            logits.reshape(B * T_o, C, *logits.shape[3:]), size=occ_size,
            mode="trilinear", align_corners=False,
        )
        return up.argmax(dim=1).reshape(T_o, *occ_size).cpu().numpy()


def main(argv: Optional[List[str]] = None) -> int:
    from configs import get_config, list_configs
    from drift.data.collate import collate_fn
    from drift.models.drift import DRIFT
    from drift.utils.seed import seed_everything

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["figure", "rollout", "temporal"], default="rollout",
                   help="figure: one static PNG/PDF for the paper. rollout/temporal: animation.")
    p.add_argument("--config", default="cam4docc_gmo", choices=list_configs())
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--dataset", default=None, choices=["synthetic", "cam4docc"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--ann-file", default=None)
    p.add_argument("--index", type=int, default=None,
                   help="Sample to render (figure/rollout) or to anchor on (temporal). "
                        "Omit in temporal mode to use the longest continuous drive in the split.")
    p.add_argument("--max-frames", type=int, default=40, help="temporal mode: frame cap.")
    p.add_argument("--horizon", type=int, default=0,
                   help="temporal mode: which forecast horizon to display (0 = nearest).")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    if not args.checkpoint and not args.random_init:
        print("[video] pass --checkpoint <path>, or --random-init to check the layout only.",
              file=sys.stderr)
        return 1

    from tools.eval import build_config, build_val_dataset, load_checkpoint_into

    args.batch_size, args.num_workers = 1, 0
    args.seed = getattr(args, "seed", None)
    cfg = build_config(args)
    seed_everything(cfg.train.seed)

    want = args.device or cfg.train.device
    device = torch.device(want if want != "cuda" or torch.cuda.is_available() else "cpu")

    dataset = build_val_dataset(cfg)
    model = DRIFT(cfg.model, cfg.loss).to(device).eval()
    tag = "random-init"
    if args.checkpoint:
        ckpt = load_checkpoint_into(model, args.checkpoint, device)
        tag = f"epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')}"
        print(f"[video] loaded checkpoint: {args.checkpoint} ({tag})")

    occ_size = tuple(cfg.model.occ_size)
    out_dir = Path(args.out_dir)
    frames: List[Image.Image] = []

    # figure and rollout address one sample, so an unspecified index means the first.
    sample_index = 0 if args.index is None else args.index

    if args.mode == "figure":
        if not 0 <= sample_index < len(dataset):
            print(f"[figure] --index {sample_index} is outside the split (0..{len(dataset) - 1}).",
                  file=sys.stderr)
            return 1
        batch = collate_fn([dataset[sample_index]])
        pred = _predict(model, batch, device, occ_size)
        gt = batch["gt_occ"][0].cpu().numpy().astype(np.int64)
        fig = render_rollout_figure(
            gt, pred, 0.5, f"{cfg.name} — val sample {sample_index} — {tag}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for ext in ("png", "pdf"):
            path = out_dir / f"rollout_figure_{sample_index:05d}.{ext}"
            # `tight` is right here and wrong for video frames: a static figure should be
            # trimmed to its content, and nothing depends on two of them matching in size.
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
            written.append(str(path))
        plt.close(fig)
        print("[figure] wrote " + " and ".join(written))
        return 0

    if args.mode == "rollout":
        if not 0 <= sample_index < len(dataset):
            print(f"[video] --index {sample_index} is outside the split (0..{len(dataset) - 1}).",
                  file=sys.stderr)
            return 1
        batch = collate_fn([dataset[sample_index]])
        pred = _predict(model, batch, device, occ_size)
        gt = batch["gt_occ"][0].cpu().numpy().astype(np.int64)
        title = f"{cfg.name} — sample {sample_index} — {tag}"
        for t in range(gt.shape[0]):
            frames.append(_frame_to_image(render_rollout_frame(gt, pred, t, 0.5, title)))
            print(f"[video] horizon {t + 1}/{gt.shape[0]}")
        stem = f"rollout_{sample_index:05d}"
    else:
        entries = getattr(dataset, "_index", None) or [{}] * len(dataset)
        indices = temporal_run(entries, args.max_frames, args.index)
        if not indices:
            print("[video] found no frames to animate.", file=sys.stderr)
            return 1
        if len(indices) == 1:
            print(
                "[video] only one frame belongs to this run, so there is nothing to animate. "
                "The annotation file is not in time order (tools/prepare_nuscenes.py shards "
                "the sample list), and no drive in it has consecutive keyframes. Re-run "
                "without --index to search the whole split.",
                file=sys.stderr,
            )
        print(f"[video] {len(indices)} consecutive keyframes of one drive "
              f"(samples {indices[0]}..{indices[-1]})")
        title = f"{cfg.name} — {tag}"
        for n, idx in enumerate(indices):
            batch = collate_fn([dataset[idx]])
            pred = _predict(model, batch, device, occ_size)
            gt = batch["gt_occ"][0].cpu().numpy().astype(np.int64)
            imgs_t = batch["imgs"][0]
            imgs = imgs_t[-1].cpu().numpy() if torch.is_tensor(imgs_t) else np.zeros((0, 3, 1, 1))
            frames.append(_frame_to_image(render_temporal_frame(
                imgs, gt, pred, args.horizon, n, len(indices), 0.5, title,
            )))
            print(f"[video] frame {n + 1}/{len(indices)} (sample {idx})")
        stem = f"temporal_{indices[0]:05d}_{len(indices)}f"

    gif_path = out_dir / f"{stem}.gif"
    write_gif(frames, gif_path, args.fps)
    print(f"[video] wrote {gif_path} ({len(frames)} frames at {args.fps} fps)")

    mp4_path = out_dir / f"{stem}.mp4"
    if write_mp4(frames, mp4_path, args.fps):
        print(f"[video] wrote {mp4_path}")
    else:
        print("[video] ffmpeg unavailable -- GIF only. Convert locally if an MP4 is needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
