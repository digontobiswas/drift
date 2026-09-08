#!/usr/bin/env python
"""Move a checkpoint's resume position forward, to step over a stretch of an epoch
that training cannot get through.

Why this exists
---------------
The SIGSEGVs on this cluster are not uniformly distributed. Epochs 0-2 and most of
epoch 3 ran for eight to twelve hours at a stretch. From around batch 19500 of epoch 3
the crash-free interval collapsed to tens of batches, and the requeue chain has been
crawling forward roughly 50 batches per five-minute job ever since -- a rate that would
need weeks to finish the run.

The fault is not one bad sample: batch 19498 was trained successfully by job 1159744 and
killed job 1160487, so nothing about that sample is reliably fatal on its own. What that
leaves is a stretch of the epoch that is expensive enough, in some way that accumulates
across consecutive batches, to make the crash likely rather than certain. This script
does not explain that. It steps over it.

Skipping is cheap in the only currency that matters here. A few hundred batches is about
one percent of a single epoch out of twelve, and `epoch_index_order` reshuffles every
epoch, so the samples stepped over here are still trained on in the eleven other epochs.
Nothing is permanently dropped from the run.

The previous checkpoint is copied aside before anything is written, so a skip that turns
out to be unnecessary can be undone exactly.

Usage
-----
    python tools/skip_ahead.py <ckpt> --by 300      # 300 batches past where it is now
    python tools/skip_ahead.py <ckpt> --to 19900    # or name the batch to resume at
    python tools/skip_ahead.py <ckpt> --restore     # undo, from the .bak copy

`global_step` is advanced to match, so the LR schedule stays aligned with the batches
actually consumed rather than drifting behind them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="path to latest.pth")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--by", type=int, help="advance batch_in_epoch by this many batches")
    g.add_argument("--to", type=int, help="set batch_in_epoch to exactly this batch")
    g.add_argument("--restore", action="store_true", help="undo a previous skip from the .bak copy")
    p.add_argument(
        "--epoch-batches", type=int, default=22530,
        help="batches in one epoch, used to refuse a skip that would run off the end (default: 22530)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.checkpoint)
    backup = path.with_name(path.name + ".bak")

    if args.restore:
        if not backup.exists():
            print(f"[skip] no backup at {backup} -- nothing to restore", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[skip] restored {path} to epoch={ckpt['epoch']} batch_in_epoch={ckpt['batch_in_epoch']}")
        return 0

    if not path.exists():
        print(f"[skip] no checkpoint at {path}", file=sys.stderr)
        return 1

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    old_batch = ckpt.get("batch_in_epoch", 0)
    old_step = ckpt.get("global_step", 0)
    epoch = ckpt.get("epoch", 0)

    new_batch = old_batch + args.by if args.by is not None else args.to
    if new_batch <= old_batch:
        print(
            f"[skip] refusing to move backwards or stay put: {old_batch} -> {new_batch}. "
            "Resuming earlier than the checkpoint would retrain batches already done.",
            file=sys.stderr,
        )
        return 1
    if new_batch >= args.epoch_batches:
        print(
            f"[skip] refusing: batch {new_batch} is past the end of a {args.epoch_batches}-batch "
            "epoch. To finish this epoch, let it run out normally rather than skipping past it.",
            file=sys.stderr,
        )
        return 1

    # The LR schedule is driven by global_step, so it has to advance with the batches or
    # the learning rate lags behind the position in the run by exactly the skipped amount.
    advanced = new_batch - old_batch
    ckpt["batch_in_epoch"] = new_batch
    ckpt["global_step"] = old_step + advanced

    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

    print(f"[skip] backed up the previous checkpoint to {backup}")
    print(
        f"[skip] epoch={epoch}: batch_in_epoch {old_batch} -> {new_batch} "
        f"(+{advanced}), global_step {old_step} -> {ckpt['global_step']}"
    )
    print("[skip] resubmit training to resume from the new position")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
