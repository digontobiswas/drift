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
import faulthandler
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Dump a Python traceback on a fatal native signal (SIGSEGV/SIGBUS/SIGFPE/SIGABRT).
# A segfault inside a C extension -- torch, NCCL, PIL's JPEG decoder -- otherwise kills
# the process with no Python-level information at all: torchrun reports only
# "Signal 11 (SIGSEGV) received by PID ...", which says nothing about WHERE. With this
# enabled the stack is written to stderr (the job's .err file) before the process dies.
faulthandler.enable()


# --- graceful shutdown on a walltime kill -----------------------------------
#
# The gpu partition caps a job at 72 hours but a full run needs ~4 days, so every
# run WILL be interrupted at least once. Slurm announces this by sending a signal
# (configured via `#SBATCH --signal=USR1@600`, i.e. 10 minutes before the hard
# kill) and then SIGKILLs the job, which cannot be caught.
#
# Without a handler, everything since the last periodic checkpoint is lost. With
# one, the loop notices the flag at the next step boundary, writes a checkpoint at
# the exact position it reached, and exits cleanly -- so the follow-up job resumes
# from there instead of from up to `ckpt_interval_steps` earlier.
#
# The flag is only *set* here; the actual save happens in the training loop, since
# writing a checkpoint from inside a signal handler (mid-backward, mid-allreduce)
# is exactly how a corrupt checkpoint gets produced.
_STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    if not _STOP_REQUESTED:
        _STOP_REQUESTED = True
        print(
            f"[train] signal {signum} received -- will checkpoint and exit at the next step boundary",
            flush=True,
        )


for _sig in (signal.SIGUSR1, signal.SIGTERM):
    try:
        signal.signal(_sig, _request_stop)
    except (ValueError, OSError):  # not on the main thread, or unsupported platform
        pass

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
    p.add_argument(
        "--ckpt-interval-steps", type=int, default=None,
        help="Save latest.pth every N optimizer steps (0 disables). Default from the config.",
    )
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
    if args.ckpt_interval_steps is not None:
        cfg.train.ckpt_interval_steps = args.ckpt_interval_steps
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
        return Cam4DOccDataset(
            data_root=d.data_root,
            ann_file=d.ann_file,
            in_channels=d.in_channels,
            img_hw=(d.H_img, d.W_img),
            point_dims_on_disk=d.point_dims_on_disk,
        )
    raise ValueError(f"Unknown dataset '{d.dataset}'.")


def epoch_index_order(dataset, sampler, epoch: int, seed: int) -> "list[int]":
    """The exact sequence of dataset indices this rank consumes during `epoch`.

    Materialising the order is what makes a mid-epoch resume cheap. Asking the
    DataLoader to shuffle internally leaves no way to re-enter an epoch part-way
    except to iterate from the start and discard, which pays the full decode cost
    of every skipped sample -- 11 minutes to skip 5000 batches, on real data, and
    growing linearly the deeper into the epoch the crash happened. With the order
    in hand the finished batches are simply dropped off the front.

    The order must be reproducible across processes and across resumes of the same
    epoch, or two ranks would disagree about who trains on what, and a resumed run
    would revisit samples it had already seen while skipping others entirely.
    `DistributedSampler` already guarantees that given `set_epoch`; the
    single-process path seeds its own permutation to get the same guarantee, which
    `shuffle=True` did not provide.
    """
    if sampler is not None:
        sampler.set_epoch(epoch)
        return list(sampler)
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    return torch.randperm(len(dataset), generator=g, dtype=torch.int64).tolist()


def build_epoch_loader(
    cfg: DriftConfig, dataset, order: "list[int]", skip_batches: int, drop_last: bool
) -> DataLoader:
    """A loader over the batches of `order` this epoch has not trained on yet.

    Batch `i` of an epoch consumes `order[i * B : (i + 1) * B]`, so dropping the
    first `skip_batches * B` indices resumes exactly where the checkpoint left off.
    """
    remaining = order[skip_batches * cfg.data.batch_size:]
    return DataLoader(
        dataset, batch_size=cfg.data.batch_size, sampler=remaining,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, drop_last=drop_last,
    )


def in_flight_samples(order: "list[int]", it: int, batch_size: int) -> "list[int]":
    """The dataset indices making up batch `it`, mirroring `build_epoch_loader`'s slicing."""
    lo = it * batch_size
    return order[lo:lo + batch_size]


def record_in_flight(path: Path, epoch: int, it: int, global_step: int, samples: "list[int]") -> None:
    """Overwrite a one-line note naming the batch currently in the forward/backward.

    A SIGSEGV leaves a faulthandler traceback, which says the fault was inside
    `_engine_run_backward` and nothing more. What that cannot say is WHICH sample was
    being trained on -- the one fact that separates a random memory fault, where the
    position is meaningless, from an input the model genuinely cannot process, where
    the same sample kills every attempt. Job 1159744 died about a hundred steps after
    resuming into the same region of epoch 3 that had killed its predecessor, which is
    the pattern that question is worth asking about.

    `epoch_index_order` is deterministic in (seed, epoch), so an index recorded here
    names the same sample on every resume and can be replayed on its own afterwards.

    Never allowed to interrupt training: this is a diagnostic, and a shared-filesystem
    hiccup while writing one must not end a run that is otherwise perfectly healthy.
    """
    try:
        path.write_text(
            f"epoch={epoch} it={it} step={global_step} "
            f"samples={','.join(str(s) for s in samples)}\n"
        )
    except Exception:
        pass


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


def lr_at_step(cfg: DriftConfig, step: int, total_steps: int) -> float:
    """Learning rate for optimizer step ``step`` (0-based) under ``cfg.train.lr_scheduler``.

    Warmup is linear from ``lr * warmup_start_ratio`` up to ``lr`` over the first
    ``warmup_iters`` steps, and applies to every non-constant schedule. After
    warmup:

    - ``"cosine"``: decays to ``lr * min_lr_ratio`` at ``total_steps``.
    - ``"step"``:   multiplied by ``step_gamma`` at each milestone fraction.
    - ``"constant"``: no warmup, no decay -- returns ``lr`` unchanged.

    Args:
        cfg: Full config; reads ``cfg.train``.
        step: 0-based global optimizer step.
        total_steps: Planned total steps for the run, used to place the decay.

    Returns:
        The learning rate to apply at this step.
    """
    t = cfg.train
    base = t.lr
    if t.lr_scheduler == "constant":
        return base

    if t.warmup_iters > 0 and step < t.warmup_iters:
        frac = (step + 1) / float(t.warmup_iters)
        return base * (t.warmup_start_ratio + (1.0 - t.warmup_start_ratio) * frac)

    decay_steps = max(1, total_steps - t.warmup_iters)
    progress = min(1.0, max(0.0, (step - t.warmup_iters) / float(decay_steps)))

    if t.lr_scheduler == "cosine":
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base * (t.min_lr_ratio + (1.0 - t.min_lr_ratio) * cos)
    if t.lr_scheduler == "step":
        factor = 1.0
        for m in t.step_milestones:
            if progress >= m:
                factor *= t.step_gamma
        return base * max(factor, t.min_lr_ratio)
    raise ValueError(
        f"Unknown lr_scheduler '{t.lr_scheduler}'. Expected 'constant', 'cosine', or 'step'."
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Write ``lr`` into every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, global_step: int, cfg: DriftConfig, batch_in_epoch: int = 0,
    world_size: int = 1,
) -> None:
    """Write a resumable checkpoint.

    `epoch` is the epoch to RESUME AT, and `batch_in_epoch` how far into it this
    snapshot is (0 = start of the epoch). An epoch-end save therefore passes
    `epoch + 1` with `batch_in_epoch=0`; a mid-epoch save passes the current
    `epoch` with the batch index, so resume re-enters the same epoch and fast-
    forwards to where it left off instead of silently skipping the remainder.

    The write goes to a temporary file first and is then atomically renamed.
    Without that, a crash or walltime kill landing during `torch.save` leaves a
    truncated `latest.pth` -- and since this is the file `03_ablations.slurm`
    auto-resumes from, a corrupt one turns a recoverable interruption into a
    dead run that fails instantly on every requeue.

    `world_size` is recorded because `batch_in_epoch` counts batches on ONE rank, so
    it means different things at different GPU counts -- resuming a 2-GPU checkpoint
    on 1 GPU without accounting for that lands half way to where it should. Runs here
    switch GPU count often, because the queue hands out single free GPUs far sooner
    than pairs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "batch_in_epoch": batch_in_epoch,
            "world_size": world_size,
            "config_name": cfg.name,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(
    path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
    device: torch.device, world_size: int = 1,
) -> "tuple[int, int, int]":
    """Returns `(epoch, global_step, batch_in_epoch)` to resume at.

    `batch_in_epoch` is 0 for checkpoints written before it was recorded, which
    makes an older checkpoint simply restart its epoch -- the previous behaviour.

    It counts batches on one rank, so a checkpoint written under a different GPU
    count has to be rescaled: 8500 batches per rank on 2 GPUs is 17000 samples of
    the epoch consumed, which is batch 17000 when one rank does all the work. Left
    unscaled, switching 2 GPUs -> 1 silently resumes half as far into the epoch as
    it should. The samples themselves differ either way -- the shuffle partitions
    differently at each GPU count -- so this restores how MUCH of the epoch is done,
    not which samples did it; within an epoch of a shuffled dataset that is the
    property that matters.
    """
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    batch_in_epoch = ckpt.get("batch_in_epoch", 0)
    saved_world_size = ckpt.get("world_size", world_size)
    if batch_in_epoch and saved_world_size != world_size:
        rescaled = batch_in_epoch * saved_world_size // world_size
        print(
            f"[train] checkpoint was written on {saved_world_size} GPU(s), resuming on "
            f"{world_size}: rescaling batch_in_epoch {batch_in_epoch} -> {rescaled}",
            flush=True,
        )
        batch_in_epoch = rescaled
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), batch_in_epoch


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
    # Measures a full epoch, for the LR schedule's total-step count. Training itself
    # iterates a per-epoch loader built by `build_epoch_loader`, which can start
    # part-way in; iterating THIS one would silently retrain batches a resumed run
    # has already seen, so it is deliberately never used for anything but its length.
    full_epoch_loader = DataLoader(
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

    start_epoch, global_step, resume_batch = 0, 0, 0
    if cfg.train.resume is not None:
        start_epoch, global_step, resume_batch = load_checkpoint(
            cfg.train.resume, model, optimizer, device, world_size=world_size
        )
        if is_main:
            print(
                f"[train] resumed from {cfg.train.resume} at epoch={start_epoch} "
                f"step={global_step} batch_in_epoch={resume_batch}"
            )

    ckpt_dir = Path(cfg.train.ckpt_dir)

    # Total planned steps, so the decay schedule lands exactly at the end of training.
    steps_per_epoch = max(1, len(full_epoch_loader))
    total_steps = steps_per_epoch * cfg.train.epochs
    if cfg.train.max_iters is not None:
        total_steps = min(total_steps, cfg.train.max_iters)
    if is_main:
        print(
            f"[train] lr_scheduler={cfg.train.lr_scheduler} base_lr={cfg.train.lr:g} "
            f"warmup={cfg.train.warmup_iters} total_steps={total_steps} "
            f"({steps_per_epoch} it/epoch x {cfg.train.epochs} epochs)"
        )

    stop = False
    interrupted = False  # set when a walltime signal ends the run, vs. a clean finish
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        t0 = time.time()
        # Re-enter a partially-trained epoch by starting the loader at the batch the
        # checkpoint stopped on, rather than iterating from zero and discarding. The
        # skipped batches are never constructed, so resuming costs no decode time at
        # all. `skip_batches` applies only to the epoch we resumed into; later epochs
        # run whole.
        skip_batches = resume_batch if epoch == start_epoch else 0
        order = epoch_index_order(dataset, sampler, epoch, cfg.train.seed)
        epoch_loader = build_epoch_loader(cfg, dataset, order, skip_batches, world_size > 1)
        if skip_batches and is_main:
            print(
                f"[train] resuming epoch {epoch} at batch {skip_batches} "
                f"({len(epoch_loader)} batches left)",
                flush=True,
            )
        # `it` counts batches from the start of the epoch, not from the start of this
        # run, because it is what gets written as `batch_in_epoch`. Pre-set so that
        # `it + 1` still names the resume point if the loader yields nothing at all --
        # which is what an epoch that was already finished looks like.
        it = skip_batches - 1
        for local_it, batch in enumerate(epoch_loader):
            it = skip_batches + local_it
            samples = in_flight_samples(order, it, cfg.data.batch_size)
            if is_main:
                record_in_flight(ckpt_dir / "in_flight.txt", epoch, it, global_step, samples)
            batch = move_batch_to_device(batch, device)

            current_lr = lr_at_step(cfg, global_step, total_steps)
            set_lr(optimizer, current_lr)

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
                    f"sample={','.join(str(s) for s in samples)} lr={current_lr:.3e} "
                    f"total_loss={total_loss.item():.4f} {loss_str} ({elapsed:.1f}s)",
                    flush=True,
                )

            # Periodic mid-epoch checkpoint. On real data an epoch is ~8 hours, so
            # without this any crash between epoch boundaries discards everything since
            # the last one -- exactly what a 3-hour SIGSEGV did on job 1153168, which
            # left no checkpoint at all. Saved with the CURRENT epoch and batch index so
            # resume re-enters this epoch and fast-forwards, rather than skipping ahead.
            ckpt_every = getattr(cfg.train, "ckpt_interval_steps", 0)
            if is_main and ckpt_every and global_step % ckpt_every == 0:
                save_checkpoint(
                    ckpt_dir / "latest.pth", model, optimizer, epoch, global_step, cfg,
                    batch_in_epoch=it + 1, world_size=world_size,
                )
                print(
                    f"[train] checkpoint saved at epoch={epoch} it={it} step={global_step}",
                    flush=True,
                )

            if cfg.train.max_iters is not None and global_step >= cfg.train.max_iters:
                stop = True
                break

            # Walltime kill announced (see _request_stop). Leave the loop now so the
            # save below records the exact position reached; every rank sees the same
            # signal from Slurm, so they break together and DDP does not deadlock.
            if _STOP_REQUESTED:
                stop = True
                interrupted = True
                break
        # An early `break` (--max-iters, or a walltime signal) means the epoch did NOT
        # finish. Recording it as finished -- `epoch + 1, batch_in_epoch=0`, as an
        # epoch-end save does -- would make the follow-up job skip every remaining batch
        # of this epoch and silently train on less data than the config claims. So the
        # two cases save different things:
        #
        #   completed   -> epoch + 1, batch_in_epoch=0   (resume starts the next epoch)
        #   interrupted -> epoch,     batch_in_epoch=it+1 (resume re-enters and fast-forwards)
        #
        # A --max-iters stop still writes a checkpoint either way: an earlier bug had it
        # `break` before any save, so the documented `--max-iters 3` smoke run produced no
        # checkpoint files at all and `--resume` had nothing to load.
        if is_main:
            if interrupted or stop:
                save_checkpoint(
                    ckpt_dir / "latest.pth", model, optimizer, epoch, global_step, cfg,
                    batch_in_epoch=it + 1, world_size=world_size,
                )
                reason = "interrupted by signal" if interrupted else "stopped at max_iters"
                print(
                    f"[train] {reason} at epoch={epoch} it={it} step={global_step}; "
                    f"checkpoint saved to {ckpt_dir}/latest.pth",
                    flush=True,
                )
            else:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch}.pth", model, optimizer, epoch + 1, global_step, cfg,
                    batch_in_epoch=0, world_size=world_size,
                )
                save_checkpoint(
                    ckpt_dir / "latest.pth", model, optimizer, epoch + 1, global_step, cfg,
                    batch_in_epoch=0, world_size=world_size,
                )
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
