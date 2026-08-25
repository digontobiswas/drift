#!/usr/bin/env python3
"""DRIFT evaluation script. See ``docs/DESIGN_SPEC.md`` §5 for the metric contracts.

Loads a checkpoint (or, with ``--random-init``, builds a freshly-initialized model for a
smoke test) and runs it over one dataset split, reporting:

- The Cam4DOcc-style occupancy IoU table (``drift.metrics.iou.OccupancyIoUMetric``):
  per-class IoU, ``IOU_mean`` (classes ``1:`` averaged, class 0 = free excluded),
  ``IoU_c`` (present-index bucket) and ``IoU_f`` (all-future-frames-pooled bucket), plus the
  cumulative per-horizon buckets.
- Flow EPE / angular error / magnitude error in metres (``drift.metrics.flow_epe``).
- Expected Calibration Error (``drift.metrics.calibration``), pooled across the whole split.

**On "IoU_c".** Per ``drift/models/drift.py``'s SEAM #5 docstring, DRIFT's data/model
contract has *no* present-frame reconstruction slot: every one of the ``T_o`` output
indices is a genuine future frame (index ``k`` = ``(k+1) * 0.5s`` after the present).
``OccupancyIoUMetric``'s ``present_index`` (default ``0``, matched here) therefore just
labels the *nearest*-horizon (``t=+0.5s``) frame as "present" for bucketing purposes -- it
is not a literal same-timestamp reconstruction. See the README's Known Limitations section.

Examples:
    CPU smoke test on synthetic data with a random-init model (no checkpoint needed)::

        python tools/eval.py --config tiny --random-init --device cpu

    Evaluate a trained checkpoint on the Cam4DOcc-protocol val split::

        python tools/eval.py --config cam4docc_2s --checkpoint work_dirs/drift/latest.pth \\
            --dataset cam4docc --data-root /data/cam4docc --ann-file val.json \\
            --json-out results/eval_cam4docc_2s.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from configs import get_config, list_configs
from configs.base import DriftConfig
from drift.data.cam4docc_dataset import Cam4DOccDataset, SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.metrics.calibration import expected_calibration_error
from drift.metrics.flow_epe import FlowEPEMetric
from drift.metrics.iou import OccupancyIoUMetric
from drift.models.drift import DRIFT
from drift.utils.seed import seed_everything

# Approximate nuScenes-occupancy / Cam4DOcc 17-class semantic vocabulary (class 0 = free).
# This is a display convenience only -- if a config's `num_classes` doesn't match this list's
# length, or an `ann_file`'s actual label map differs, generic "class_<i>" names are used
# instead (see `_class_names`). Nothing in the metrics computation depends on these names.
_CAM4DOCC_17_CLASS_NAMES = [
    "free", "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface", "other_flat",
    "sidewalk", "terrain", "manmade", "vegetation",
]

# Synthetic dataset has no on-disk train/val split; a val "split" is simulated by drawing
# samples from a seed range disjoint from whatever `--seed`/`cfg.data.seed` training used.
_SYNTHETIC_VAL_SEED_OFFSET = 1_000_000


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse ``tools/eval.py`` command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate a DRIFT model/checkpoint.")
    p.add_argument("--config", type=str, default="tiny", choices=list_configs())
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a tools/train.py checkpoint (.pth).")
    p.add_argument(
        "--random-init", action="store_true",
        help="Skip checkpoint loading; evaluate a freshly-initialized model (smoke test only).",
    )
    p.add_argument("--dataset", type=str, default=None, choices=["synthetic", "cam4docc"])
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ann-file", type=str, default=None, help="For --dataset cam4docc: the val split's annotation file.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--num-batches", type=int, default=None, help="Evaluate only the first N batches (fast smoke run).")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--present-index", type=int, default=0, help="Output index treated as 'present' for IoU bucketing.")
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--json-out", type=str, default=None, help="Optional path to dump the full results dict as JSON.")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> DriftConfig:
    """Load the named preset and apply CLI overrides (mirrors ``tools/train.py``)."""
    cfg = get_config(args.config)
    if args.dataset is not None:
        cfg.data.dataset = args.dataset
    if args.data_root is not None:
        cfg.data.data_root = args.data_root
    if args.ann_file is not None:
        cfg.data.ann_file = args.ann_file
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.device is not None:
        cfg.train.device = args.device
    # Eval never applies training-time modality dropout augmentation on top of the dataset's
    # own masks (ModalityDropout is already a no-op in eval mode -- see drift/models/cmli.py --
    # but zero this out too so `--dataset synthetic`'s own stochastic masks stay the only
    # source of missing-modality frames unless the user is explicitly running robustness).
    cfg.data.modality_dropout_p = 0.0
    return cfg


def build_val_dataset(cfg: DriftConfig):
    """Build the val-split dataset for evaluation. See ``tools/train.py``'s ``build_dataset``."""
    d, m = cfg.data, cfg.model
    if d.dataset == "synthetic":
        return SyntheticOccDataset(
            num_samples=d.num_samples, T_p=m.T_p, T_f=m.T_f, T_o=m.T_o, N_cam=m.N_cam,
            H_img=d.H_img, W_img=d.W_img, num_classes=m.num_classes, latent_size=m.latent_size,
            occ_size=m.occ_size, point_cloud_range=m.point_cloud_range, in_channels=d.in_channels,
            num_points_range=d.num_points_range, num_boxes_range=d.num_boxes_range,
            modality_dropout_p=0.0, seed=d.seed + _SYNTHETIC_VAL_SEED_OFFSET,
        )
    if d.dataset == "cam4docc":
        if not d.data_root or not d.ann_file:
            raise ValueError(
                "--dataset cam4docc requires --data-root and --ann-file pointing at a "
                "pre-processed Cam4DOcc-protocol val archive (see "
                "drift.data.cam4docc_dataset.Cam4DOccDataset's docstring)."
            )
        return Cam4DOccDataset(
            data_root=d.data_root,
            ann_file=d.ann_file,
            in_channels=d.in_channels,
            img_hw=(d.H_img, d.W_img),
            point_dims_on_disk=d.point_dims_on_disk,
        )
    raise ValueError(f"Unknown dataset '{d.dataset}'.")


def _move(obj: Any, device: torch.device) -> Any:
    """Move one leaf to ``device``. Tensors get ``non_blocking``; other ``.to()``-ables do not."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor / tensor-container in a DRIFT batch dict to ``device``.

    Identical logic to ``tools/train.py``'s helper of the same name (duplicated rather than
    imported since ``tools/`` is not a package and each tool is meant to run standalone).
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to") and callable(v.to):  # CameraParams
            out[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            out[k] = [[_move(t, device) for t in sample] for sample in v]
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "to"):
            out[k] = [_move(sample, device) for sample in v]
        else:
            out[k] = v
    return out


def load_checkpoint_into(model: torch.nn.Module, path: str, device: torch.device) -> Dict[str, Any]:
    """Load a ``tools/train.py``-style checkpoint's model weights in place.

    Args:
        model: The ``DRIFT`` instance to load weights into.
        path: Checkpoint path.
        device: Map-location device.

    Returns:
        The raw checkpoint dict (for reporting ``epoch``/``global_step``/``config_name``).
    """
    ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint at {path} has no 'model' key; not a tools/train.py checkpoint?")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    del missing, unexpected  # strict=True already raises on mismatch; kept for clarity
    return ckpt


class ECEAccumulator:
    """Pools per-batch ECE bin statistics into one split-wide Expected Calibration Error.

    Mirrors the pooling philosophy of ``OccupancyIoUMetric`` / ``FlowEPEMetric``: rather than
    averaging each batch's own scalar ECE (which over-weights small batches), the per-bin
    confidence-sum / accuracy-sum / count from every batch are accumulated and combined once,
    at ``compute()`` time, into a single voxel-weighted (micro-average) ECE.

    Args:
        num_bins: Number of equal-width confidence bins in ``[0, 1]``.
    """

    def __init__(self, num_bins: int = 15) -> None:
        self.num_bins = num_bins
        self._conf_sum = torch.zeros(num_bins, dtype=torch.float64)
        self._acc_sum = torch.zeros(num_bins, dtype=torch.float64)
        self._count = torch.zeros(num_bins, dtype=torch.float64)
        self._bin_edges: Optional[Tensor] = None

    @torch.no_grad()
    def update(self, probs: Tensor, labels: Tensor) -> None:
        """Accumulate one batch's calibration statistics.

        Args:
            probs: ``(..., num_classes)`` predicted class probabilities.
            labels: ``(...)`` int64 ground-truth labels, ``255`` = ignore.
        """
        out = expected_calibration_error(probs, labels, num_bins=self.num_bins, from_logits=False)
        if self._bin_edges is None:
            self._bin_edges = out["bin_edges"].cpu()
        count = out["bin_count"].double().cpu()
        self._conf_sum += out["bin_confidence"].double().cpu() * count
        self._acc_sum += out["bin_accuracy"].double().cpu() * count
        self._count += count

    def compute(self) -> Dict[str, Any]:
        """Return the pooled ECE and its reliability-curve bin statistics."""
        n = float(self._count.sum().item())
        if n == 0:
            return {"ece": 0.0, "n_valid": 0, "bin_confidence": [], "bin_accuracy": [], "bin_count": []}
        safe_count = self._count.clamp_min(1.0)
        bin_conf = self._conf_sum / safe_count
        bin_acc = self._acc_sum / safe_count
        ece = float(((self._count / n) * (bin_acc - bin_conf).abs()).sum().item())
        return {
            "ece": ece,
            "n_valid": int(n),
            "bin_confidence": bin_conf.tolist(),
            "bin_accuracy": bin_acc.tolist(),
            "bin_count": self._count.tolist(),
        }


def _upsample_probs(occ_logits: Tensor, upsample_size: "tuple[int, int, int]") -> Tensor:
    """Trilinear-upsample per-horizon occupancy logits to full resolution, then softmax.

    Mirrors ``drift.metrics.iou.OccupancyIoUMetric.update``'s upsample-then-decide ordering
    (upsample the *logits*, never upsample an already-argmax'd/one-hot label), applied here
    for calibration instead of argmax.

    Args:
        occ_logits: ``(B, T_o, num_classes, x, y, z)`` raw logits at latent resolution.
        upsample_size: Target ``(X, Y, Z)``.

    Returns:
        ``(B, T_o, X, Y, Z, num_classes)`` softmax probabilities.
    """
    B, T_o, C = occ_logits.shape[:3]
    flat = occ_logits.reshape(B * T_o, C, *occ_logits.shape[3:]).float()
    up = F.interpolate(flat, size=upsample_size, mode="trilinear", align_corners=False)
    probs = torch.softmax(up, dim=1)
    return probs.reshape(B, T_o, C, *upsample_size).permute(0, 1, 3, 4, 5, 2)


def _class_names(num_classes: int) -> List[str]:
    if num_classes == len(_CAM4DOCC_17_CLASS_NAMES):
        return list(_CAM4DOCC_17_CLASS_NAMES)
    return [f"class_{i}" for i in range(num_classes)]


def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: DriftConfig,
    device: torch.device,
    present_index: int = 0,
    ece_bins: int = 15,
    num_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full evaluation loop and return the assembled results dict.

    Args:
        model: A ``DRIFT`` instance (weights already loaded / random-init as desired).
        loader: Val-split ``DataLoader`` (``collate_fn``-collated DRIFT batches).
        cfg: The resolved ``DriftConfig``.
        device: Device to run inference on.
        present_index: Forwarded to ``OccupancyIoUMetric``.
        ece_bins: Number of ECE confidence bins.
        num_batches: If set, stop after this many batches (fast smoke run).

    Returns:
        Dict with keys ``iou`` (the ``OccupancyIoUMetric.compute()`` dict, tensors converted
        to lists/floats), ``flow`` (``FlowEPEMetric.compute()`` dict), ``ece`` (the
        ``ECEAccumulator.compute()`` dict), ``class_names``, ``num_batches``, ``num_samples``,
        and ``elapsed_s``.
    """
    model.eval()
    iou_metric = OccupancyIoUMetric(
        num_classes=cfg.model.num_classes, num_future=cfg.model.T_o,
        upsample_size=tuple(cfg.model.occ_size), present_index=present_index,
    )
    flow_metric = FlowEPEMetric()
    ece_metric = ECEAccumulator(num_bins=ece_bins)

    n_samples = 0
    n_batches_seen = 0
    t0 = time.time()
    with torch.no_grad():
        for b_i, batch in enumerate(loader):
            if num_batches is not None and b_i >= num_batches:
                break
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)

            iou_metric.update(outputs["occ_logits"], batch["gt_occ"])
            flow_metric.update(outputs["flow_pred"], batch["gt_flow"])

            probs = _upsample_probs(outputs["occ_logits"], tuple(cfg.model.occ_size))
            ece_metric.update(probs, batch["gt_occ"])

            n_samples += batch["imgs"].shape[0]
            n_batches_seen += 1
    elapsed = time.time() - t0

    iou_result = iou_metric.compute()
    iou_result["per_class_present"] = iou_result["per_class_present"].tolist()
    iou_result["per_class_future"] = iou_result["per_class_future"].tolist()
    iou_result["IOU_mean"] = (
        sum(iou_result["per_class_present"][1:]) / len(iou_result["per_class_present"][1:])
        if cfg.model.num_classes > 1 else iou_result["per_class_present"][0]
    )
    del iou_result["hist_present"], iou_result["hist_future"]

    return {
        "iou": iou_result,
        "flow": flow_metric.compute(),
        "ece": ece_metric.compute(),
        "class_names": _class_names(cfg.model.num_classes),
        "num_batches": n_batches_seen,
        "num_samples": n_samples,
        "elapsed_s": elapsed,
    }


def format_report(results: Dict[str, Any], cfg: DriftConfig, checkpoint_desc: str) -> str:
    """Render ``run_eval``'s results dict as a clean, fixed-width text report."""
    lines: List[str] = []
    W = 78
    lines.append("=" * W)
    lines.append(f"DRIFT evaluation -- config={cfg.name}  checkpoint={checkpoint_desc}")
    lines.append(
        f"T_p={cfg.model.T_p} T_o={cfg.model.T_o} num_classes={cfg.model.num_classes} "
        f"latent_size={tuple(cfg.model.latent_size)} occ_size={tuple(cfg.model.occ_size)}"
    )
    lines.append(f"samples={results['num_samples']}  batches={results['num_batches']}  "
                 f"wall={results['elapsed_s']:.2f}s")
    lines.append("=" * W)

    iou = results["iou"]
    names = results["class_names"]
    lines.append("")
    lines.append("Occupancy IoU (Cam4DOcc-style pooled confusion matrix)")
    lines.append("-" * W)
    lines.append(f"{'class':<22}{'IoU_present':>14}{'IoU_future':>14}")
    for i, name in enumerate(names):
        lines.append(f"{name:<22}{iou['per_class_present'][i]:>14.4f}{iou['per_class_future'][i]:>14.4f}")
    lines.append("-" * W)
    lines.append(f"{'IOU_mean (classes 1:)':<22}{iou['IOU_mean']:>14.4f}")
    lines.append(f"{'IoU_c (present bucket)':<22}{iou['IoU_c']:>14.4f}")
    lines.append(f"{'IoU_f (future pooled)':<22}{iou['IoU_f']:>14.4f}")
    lines.append("")
    lines.append("Cumulative per-horizon IoU (bucket k pools every future frame up to horizon k)")
    lines.append("-" * W)
    lines.append(f"{'horizon':<12}{'IoU':>10}")
    for k, v in enumerate(iou["per_horizon_IoU"], start=1):
        horizon_s = 0.5 * (k + 1)  # +1 since bucket k excludes the present_index slot itself
        lines.append(f"{'+' + f'{horizon_s:.1f}s':<12}{v:>10.4f}")

    lines.append("")
    lines.append("Flow error (metres; masked by GT != 255)")
    lines.append("-" * W)
    flow = results["flow"]
    lines.append(f"{'EPE (m)':<22}{flow['epe']:>14.4f}")
    lines.append(f"{'Angular error (rad)':<22}{flow['angular_error']:>14.4f}")
    lines.append(f"{'Magnitude error (m)':<22}{flow['magnitude_error']:>14.4f}")
    lines.append(f"{'valid voxels':<22}{flow['n_valid']:>14d}")

    lines.append("")
    lines.append("Calibration (Expected Calibration Error, per-voxel max-softmax confidence)")
    lines.append("-" * W)
    ece = results["ece"]
    lines.append(f"{'ECE':<22}{ece['ece']:>14.4f}")
    lines.append(f"{'valid voxels':<22}{ece['n_valid']:>14d}")
    lines.append("=" * W)
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    if not args.checkpoint and not args.random_init:
        raise ValueError(
            "tools/eval.py requires either --checkpoint <path> or --random-init "
            "(for a smoke test with a freshly-initialized model)."
        )

    cfg = build_config(args)
    seed_everything(cfg.train.seed)

    requested_device = cfg.train.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA requested but not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    print(f"[eval] config={cfg.name} device={device} dataset={cfg.data.dataset}")

    dataset = build_val_dataset(cfg)
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )

    model = DRIFT(cfg.model, cfg.loss).to(device)

    checkpoint_desc = "random-init (smoke test)"
    if args.checkpoint:
        ckpt = load_checkpoint_into(model, args.checkpoint, device)
        checkpoint_desc = (
            f"{args.checkpoint} (epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}, "
            f"trained_config={ckpt.get('config_name')})"
        )
        print(f"[eval] loaded checkpoint: {checkpoint_desc}")
    else:
        print("[eval] --random-init: evaluating a freshly-initialized model (smoke test only, "
              "numbers are meaningless as a quality measure).")

    results = run_eval(
        model, loader, cfg, device,
        present_index=args.present_index, ece_bins=args.ece_bins, num_batches=args.num_batches,
    )

    report = format_report(results, cfg, checkpoint_desc)
    print()
    print(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": cfg.name,
            "checkpoint": checkpoint_desc,
            "device": str(device),
            **results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[eval] wrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
