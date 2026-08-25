#!/usr/bin/env python3
"""DRIFT sensor-dropout robustness study. See ``drift/metrics/robustness.py``.

Runs a model under a grid of ``(lidar_dropout, cam_dropout)`` scenarios
(``drift.metrics.robustness.DropoutScenario``) and reports the occupancy-IoU degradation
table: absolute ``IoU_c``/``IoU_f`` per scenario, and percent retained vs. the both-present
("clean") baseline. This is what directly measures what ``CrossModalLatentImagination``
(★NOVEL, spec §2.4) is supposed to buy: graceful degradation rather than a cliff when a
modality drops out.

Scenario grid (``drift.metrics.robustness.DEFAULT_SCENARIOS``, selectable with ``--scenarios``):

- ``clean``: both modalities always present (the baseline every other row is normalized against).
- ``lidar_only``: camera dropped every frame (``cam_dropout=1.0``).
- ``camera_only``: LiDAR dropped every frame (``lidar_dropout=1.0``).
- ``moderate_dropout`` / ``severe_dropout``: partial, per-frame Bernoulli dropout of both
  modalities (0.3 / 0.7 probability each) -- a "flaky sensor" scenario rather than a clean
  ablation, and the scenario that best isolates CMLI's imagination-under-partial-loss behaviour
  from its full-substitution behaviour.

Examples:
    CPU smoke test on synthetic data with a random-init model::

        python tools/run_robustness.py --config tiny --random-init --device cpu --num-batches 4

    Full grid on a trained checkpoint, dumping the table to JSON::

        python tools/run_robustness.py --config cam4docc_2s --checkpoint work_dirs/drift/latest.pth \\
            --dataset cam4docc --data-root /data/cam4docc --ann-file val.json \\
            --json-out results/robustness_cam4docc_2s.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from configs import get_config, list_configs
from configs.base import DriftConfig
from drift.data.cam4docc_dataset import Cam4DOccDataset, SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.metrics.robustness import DEFAULT_SCENARIOS, DropoutScenario, evaluate_robustness
from drift.models.drift import DRIFT
from drift.utils.seed import seed_everything

_SYNTHETIC_VAL_SEED_OFFSET = 1_000_000  # matches tools/eval.py's val-split convention


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse ``tools/run_robustness.py`` command-line arguments."""
    p = argparse.ArgumentParser(description="Run the DRIFT sensor-dropout robustness study.")
    p.add_argument("--config", type=str, default="tiny", choices=list_configs())
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--random-init", action="store_true",
        help="Skip checkpoint loading; evaluate a freshly-initialized model (smoke test only).",
    )
    p.add_argument("--dataset", type=str, default=None, choices=["synthetic", "cam4docc"])
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ann-file", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--num-batches", type=int, default=None, help="Cap batches evaluated per scenario (fast smoke run).")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--present-index", type=int, default=0)
    p.add_argument(
        "--scenarios", type=str, default=None,
        help=(
            "Comma-separated subset of scenario names to run, e.g. 'clean,camera_only,lidar_only'. "
            f"Default: all of {[s.name for s in DEFAULT_SCENARIOS]}. The first *selected* scenario "
            "is always used as the both-present baseline for % retained, regardless of order given."
        ),
    )
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> DriftConfig:
    """Load the named preset and apply CLI overrides (mirrors ``tools/train.py``/``tools/eval.py``)."""
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
    # The dropout grid below is the sole source of missing-modality frames for this study;
    # `apply_modality_dropout` composes with whatever `lidar_mask`/`cam_mask` the dataset
    # already provides (an AND, same as the model's own SEAM #3), so zero the dataset's own
    # stochastic dropout to keep each scenario's effective drop rate exactly what it names.
    cfg.data.modality_dropout_p = 0.0
    return cfg


def build_val_dataset(cfg: DriftConfig):
    """Build the val-split dataset (identical logic to ``tools/eval.py``'s helper of the same name)."""
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
        return Cam4DOccDataset(data_root=d.data_root, ann_file=d.ann_file, in_channels=d.in_channels)
    raise ValueError(f"Unknown dataset '{d.dataset}'.")


def select_scenarios(names: Optional[str]) -> List[DropoutScenario]:
    """Resolve ``--scenarios`` to a ``DropoutScenario`` list, or the full default grid.

    Args:
        names: Comma-separated scenario names, or ``None`` for the full default grid.

    Returns:
        The selected scenarios, in the order named (the first is the % retained baseline).

    Raises:
        ValueError: If an unknown scenario name is requested.
    """
    if names is None:
        return list(DEFAULT_SCENARIOS)
    by_name = {s.name: s for s in DEFAULT_SCENARIOS}
    selected: List[DropoutScenario] = []
    for raw in names.split(","):
        name = raw.strip()
        if name not in by_name:
            raise ValueError(f"Unknown scenario '{name}'. Available: {sorted(by_name.keys())}")
        selected.append(by_name[name])
    if not selected:
        raise ValueError("--scenarios resolved to an empty list.")
    return selected


class _DeviceLoader:
    """Wraps a ``DataLoader``, moving every yielded batch to ``device``.

    ``evaluate_robustness`` iterates its ``dataloader`` argument once per scenario, so it must
    stay re-iterable (a plain generator *object* would be exhausted after the first scenario);
    wrapping the underlying ``DataLoader`` and defining ``__iter__`` as a fresh generator on
    every call satisfies that.
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.loader:
            yield _move_batch(batch, self.device)

    def __len__(self) -> int:
        return len(self.loader)


def _move(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor / tensor-container in a DRIFT batch dict to ``device``.

    Identical logic to ``tools/train.py``'s ``move_batch_to_device`` (duplicated rather than
    imported since ``tools/`` is not a package and each tool is meant to run standalone).
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to") and callable(v.to):
            out[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            out[k] = [[_move(t, device) for t in sample] for sample in v]
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "to"):
            out[k] = [_move(sample, device) for sample in v]
        else:
            out[k] = v
    return out


def load_checkpoint_into(model: torch.nn.Module, path: str, device: torch.device) -> Dict[str, Any]:
    """Load a ``tools/train.py``-style checkpoint's model weights in place."""
    ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint at {path} has no 'model' key; not a tools/train.py checkpoint?")
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def format_report(table: Dict[str, Dict[str, Any]], cfg: DriftConfig, checkpoint_desc: str) -> str:
    """Render the degradation table returned by ``evaluate_robustness`` as clean text."""
    lines: List[str] = []
    W = 92
    lines.append("=" * W)
    lines.append(f"DRIFT sensor-dropout robustness study -- config={cfg.name}  checkpoint={checkpoint_desc}")
    lines.append(f"T_p={cfg.model.T_p} T_o={cfg.model.T_o} num_classes={cfg.model.num_classes}  "
                 f"cmli_enabled={cfg.model.cmli.enabled}")
    lines.append("=" * W)
    lines.append("")
    header = f"{'scenario':<20}{'lidar_drop':>12}{'cam_drop':>10}{'IoU_c':>10}{'IoU_f':>10}{'% retained':>13}"
    lines.append(header)
    lines.append("-" * W)
    for name, row in table.items():
        pct = row["IoU_f_degradation_pct"]
        retained = 100.0 - pct if not math.isnan(pct) else float("nan")
        lines.append(
            f"{name:<20}{row['lidar_dropout']:>12.2f}{row['cam_dropout']:>10.2f}"
            f"{row['IoU_c']:>10.4f}{row['IoU_f']:>10.4f}{retained:>12.1f}%"
        )
    lines.append("-" * W)
    lines.append("'% retained' = 100 - IoU_f_degradation_pct, relative to the first-listed (baseline) scenario.")
    lines.append("")
    lines.append("Cumulative per-horizon IoU per scenario")
    lines.append("-" * W)
    max_h = max((len(row["per_horizon_IoU"]) for row in table.values()), default=0)
    horizon_header = "".join(f"{'+' + f'{0.5 * (k + 2):.1f}s':>10}" for k in range(max_h))
    lines.append(f"{'scenario':<20}{horizon_header}")
    for name, row in table.items():
        cells = "".join(f"{v:>10.4f}" for v in row["per_horizon_IoU"])
        lines.append(f"{name:<20}{cells}")
    lines.append("=" * W)
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    if not args.checkpoint and not args.random_init:
        raise ValueError(
            "tools/run_robustness.py requires either --checkpoint <path> or --random-init "
            "(for a smoke test with a freshly-initialized model)."
        )

    cfg = build_config(args)
    seed_everything(cfg.train.seed)

    requested_device = cfg.train.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[robustness] CUDA requested but not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    scenarios = select_scenarios(args.scenarios)
    print(f"[robustness] config={cfg.name} device={device} dataset={cfg.data.dataset} "
          f"scenarios={[s.name for s in scenarios]}")

    dataset = build_val_dataset(cfg)
    raw_loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
    )
    loader = _DeviceLoader(raw_loader, device)

    model = DRIFT(cfg.model, cfg.loss).to(device)
    model.eval()

    checkpoint_desc = "random-init (smoke test)"
    if args.checkpoint:
        ckpt = load_checkpoint_into(model, args.checkpoint, device)
        checkpoint_desc = (
            f"{args.checkpoint} (epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}, "
            f"trained_config={ckpt.get('config_name')})"
        )
        print(f"[robustness] loaded checkpoint: {checkpoint_desc}")
    else:
        print("[robustness] --random-init: numbers are meaningless as a quality measure, but "
              "still exercise CMLI's forward path under every scenario's masks.")

    table = evaluate_robustness(
        model=model, dataloader=loader, num_classes=cfg.model.num_classes, num_future=cfg.model.T_o,
        scenarios=scenarios, upsample_size=tuple(cfg.model.occ_size), present_index=args.present_index,
        max_batches=args.num_batches, seed=cfg.train.seed,
    )

    report = format_report(table, cfg, checkpoint_desc)
    print()
    print(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": cfg.name, "checkpoint": checkpoint_desc, "device": str(device), "table": table}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[robustness] wrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
