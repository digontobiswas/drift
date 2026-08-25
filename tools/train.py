#!/usr/bin/env python3
"""DRIFT training loop. See ``docs/DESIGN_SPEC.md`` §2.15, §7.

Supports multi-GPU training via ``torch.nn.parallel.DistributedDataParallel`` (launch with
``torchrun``), automatic mixed precision, checkpointing/resume, and config selection by name
(``--config``, one of ``configs.drift_fusion_nuscenes.list_configs()``).

Examples:
    Single-process CPU smoke run on synthetic data (a couple of iterations)::

        python tools/train.py --config tiny --max-iters 3 --device cpu

    Single-GPU training on the default preset::

        python tools/train.py --config cam4docc_2s --data-root /data/cam4docc \\
            --ann-file train.json --dataset cam4docc

    Multi-GPU (4 ranks) via torchrun::

        torchrun --nproc_per_node=4 tools/train.py --config cam4docc_2s --distributed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from configs import get_config, list_configs
from configs.base import DriftConfig
from drift.data.cam4docc_dataset import Cam4DOccDataset, SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.models.drift import DRIFT
from drift.utils.seed import seed_everything


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DRIFT.")
    p.add_argument("--config", type=str, default="cam4docc_2s", choices=list_configs())
    p.add_argument("--dataset", type=str, default=None, choices=["synthetic", "cam4docc"])
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ann-file", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--max-iters", type=int, default=None, help="Stop after this many optimizer steps (debug/CI).")
    p.add_argument("--distributed", action="store_true", default=False)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> DriftConfig:
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
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.device is not None:
        cfg.train.device = args.device
    if args.amp is not None:
        cfg.train.amp = args.amp
    if args.ckpt_dir is not None:
        cfg.train.ckpt_dir = args.ckpt_dir
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.log_interval is not None:
        cfg.train.log_interval = args.log_interval
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    cfg.train.distributed = args.distributed
    return cfg


def build_dataset(cfg: DriftConfig):
    d = cfg.data
    m = cfg.model
    if d.dataset == "synthetic":
        return SyntheticOccDataset(
            num_samples=d.num_samples, T_p=m.T_p, T_f=m.T_f, T_o=m.T_o, N_cam=m.N_cam,
            H_img=d.H_img, W_img=d.W_img, num_classes=m.num_classes, latent_size=m.latent_size,
            occ_size=m.occ_size, point_cloud_range=m.point_cloud_range, in_channels=d.in_channels,
            num_points_range=d.num_points_range, num_boxes_range=d.num_boxes_range,
            modality_dropout_p=d.modality_dropout_p, seed=d.seed,
        )
    if d.dataset == "cam4docc":
        if not d.data_root or not d.ann_file:
            raise ValueError(
                "dataset='cam4docc' requires --data-root and --ann-file pointing at a "
                "pre-processed Cam4DOcc-protocol archive (see "
                "drift.data.cam4docc_dataset.Cam4DOccDataset's docstring)."
            )
        return Cam4DOccDataset(data_root=d.data_root, ann_file=d.ann_file, in_channels=d.in_channels)
    raise ValueError(f"Unknown dataset '{d.dataset}'.")


def _move(obj: Any, device: torch.device) -> Any:
    """Move one leaf to `device`. Tensors get `non_blocking`; other `.to()`-ables do not."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor / tensor-container in a DRIFT batch dict to `device`."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to") and callable(v.to):  # CameraParams
            out[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            # Nested list: `points` (List[B][T] of Tensor) or `gt_boxes` (List[T_o][B] of BoxSet).
            # Only tensors accept non_blocking, so dispatch per element rather than per key.
            out[k] = [[_move(t, device) for t in sample] for sample in v]
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "to"):
            out[k] = [_move(sample, device) for sample in v]
        else:
            out[k] = v
    return out


def setup_distributed(cfg: DriftConfig) -> "tuple[int, int, int]":
    """Initialize the default process group if launched under torchrun/mpirun-style env vars.

    Returns:
        ``(rank, world_size, local_rank)``. ``(0, 1, 0)`` if not running distributed.
    """
    if not cfg.train.distributed:
        return 0, 1, 0
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise ValueError(
            "--distributed was passed but RANK/WORLD_SIZE are not set; launch this script "
            "with `torchrun --nproc_per_node=N tools/train.py --distributed ...`."
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if cfg.train.device == "cuda" and torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if cfg.train.device == "cuda":
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def save_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, global_step: int, cfg: DriftConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config_name": cfg.name,
        },
        path,
    )


def load_checkpoint(
    path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], device: torch.device
) -> "tuple[int, int]":
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)

    rank, world_size, local_rank = setup_distributed(cfg)
    is_main = rank == 0

    seed_everything(cfg.train.seed + rank)

    requested_device = cfg.train.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        if is_main:
            print("[train] CUDA requested but not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(f"cuda:{local_rank}" if requested_device == "cuda" else "cpu")

    if is_main:
        print(f"[train] config={cfg.name} device={device} world_size={world_size}")

    dataset = build_dataset(cfg)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=cfg.data.num_workers, collate_fn=collate_fn,
        drop_last=world_size > 1,
    )

    model = DRIFT(cfg.model, cfg.loss).to(device)
    if world_size > 1:
        find_unused = not cfg.model.forecaster.use_instance_path or not cfg.model.cmli.enabled
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer '{cfg.train.optimizer}'.")

    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, global_step = 0, 0
    if cfg.train.resume is not None:
        start_epoch, global_step = load_checkpoint(cfg.train.resume, model, optimizer, device)
        if is_main:
            print(f"[train] resumed from {cfg.train.resume} at epoch={start_epoch} step={global_step}")

    ckpt_dir = Path(cfg.train.ckpt_dir)
    stop = False
    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        for it, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                outputs = model(batch)
                losses = raw_model.loss(outputs, batch)
                total_loss = sum(losses.values())

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

            global_step += 1
            if is_main and (it % max(cfg.train.log_interval, 1) == 0):
                loss_str = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                elapsed = time.time() - t0
                print(
                    f"[train] epoch={epoch} it={it} step={global_step} "
                    f"total_loss={total_loss.item():.4f} {loss_str} ({elapsed:.1f}s)"
                )

            if cfg.train.max_iters is not None and global_step >= cfg.train.max_iters:
                stop = True
                break
        # BUGFIX: a --max-iters stop used to `break` here *before* the epoch-end checkpoint
        # save below, so a debug/CI run like `--max-iters 3` (the exact pattern in this
        # script's own module docstring example) silently produced zero checkpoint files --
        # defeating both the documented smoke-test use case and `--resume`. Save on every
        # early stop too (still gated on `is_main`), then break.
        if is_main:
            save_checkpoint(ckpt_dir / f"epoch_{epoch}.pth", model, optimizer, epoch + 1, global_step, cfg)
            save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, epoch + 1, global_step, cfg)
            print(f"[train] epoch {epoch} done in {time.time() - t0:.1f}s, checkpoint saved to {ckpt_dir}")

        if stop:
            break

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        print("[train] finished.")


if __name__ == "__main__":
    main()
