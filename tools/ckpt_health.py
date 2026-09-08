#!/usr/bin/env python
"""Check a checkpoint for the damage that survives every restart.

Why this exists
---------------
Training here was crashing roughly once or twice an hour, which was survivable. From
one point onward it began dying within a couple of minutes of every resume, everywhere:
two different stretches of epoch 3, and then epoch 4 from its first batch, with a fresh
shuffle. Position stopped explaining anything.

What does not change across any of that is the checkpoint. Every job resumes from
`latest.pth`, and every job writes the next one from the weights it just loaded, so a
non-finite value that appears once is copied forward for the rest of the run. It cannot
be shuffled away, skipped past, or left behind by moving to the next epoch -- which is
exactly the shape of what is being observed.

This does not fix anything. It answers one question: are the weights and the optimizer
state still finite, and are they still a plausible size? If they are, corruption is ruled
out and the cause is in the code that runs each step. If they are not, no code change
matters until an earlier checkpoint is restored.

Optimizer state is checked as carefully as the weights. AdamW's `exp_avg_sq` is where a
single infinite gradient does its lasting damage: the weights can still look perfectly
finite while the second-moment buffer that divides into every future update is not.

Usage
-----
    python tools/ckpt_health.py <ckpt> [<ckpt> ...]
    python tools/ckpt_health.py drift_runs/cam4docc_gmo/{latest.pth,latest.pth.bak}

Exit status is 0 when every checkpoint is clean, 1 when any is not, so it can gate a
resubmit in a shell one-liner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Weights this far from zero are not yet NaN but are on the way there: a network whose
# parameters have grown past this has diverged, and the run it came from is not worth
# resuming even though every value in it is still technically finite.
HUGE = 1e4


def _is_counter(name: str) -> bool:
    """AdamW stores its per-parameter step COUNT alongside the moment buffers, as a
    float tensor. It is supposed to grow without bound -- it equals the number of
    optimizer steps taken -- so the divergence heuristic below must not be pointed at
    it. Left in, it reports every healthy checkpoint past ten thousand steps as damaged,
    which is not a harmless false alarm: the whole point of this script is to decide
    whether to throw a run away and restart from an earlier checkpoint.
    """
    return name.endswith(".step")


def _scan(tensors) -> "tuple[int, int, float, list[str]]":
    """Count non-finite entries across named tensors; return the worst offenders too."""
    n_nan = n_inf = 0
    largest = 0.0
    offenders: list[str] = []
    for name, t in tensors:
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        t = t.detach()
        nan = int(torch.isnan(t).sum())
        inf = int(torch.isinf(t).sum())
        finite = t[torch.isfinite(t)]
        peak = float(finite.abs().max()) if finite.numel() else 0.0
        n_nan += nan
        n_inf += inf
        if nan or inf:
            offenders.append(f"{name}: {nan} NaN, {inf} Inf")
            continue
        # Counters are checked for NaN/Inf like everything else, but their magnitude
        # carries no information about the health of the run.
        if _is_counter(name):
            continue
        largest = max(largest, peak)
        if peak > HUGE:
            offenders.append(f"{name}: finite but huge (max |x| = {peak:.3g})")
    return n_nan, n_inf, largest, offenders


def _optimizer_tensors(opt_state: dict):
    """AdamW keeps `exp_avg` / `exp_avg_sq` per parameter, under integer keys."""
    for pid, entry in (opt_state.get("state") or {}).items():
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            yield f"param[{pid}].{key}", value


def check(path: Path) -> bool:
    if not path.exists():
        print(f"[health] {path}: MISSING", file=sys.stderr)
        return False

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("global_step", "?")
    batch = ckpt.get("batch_in_epoch", "?")
    print(f"[health] {path}")
    print(f"         epoch={epoch} global_step={step} batch_in_epoch={batch}")

    ok = True
    for label, tensors in (
        ("model", list((ckpt.get("model") or {}).items())),
        ("optimizer", list(_optimizer_tensors(ckpt.get("optimizer") or {}))),
    ):
        if not tensors:
            print(f"         {label}: (empty)")
            continue
        n_nan, n_inf, largest, offenders = _scan(tensors)
        status = "clean" if not (n_nan or n_inf or largest > HUGE) else "DAMAGED"
        if status != "clean":
            ok = False
        print(
            f"         {label}: {status} -- {len(tensors)} tensors, "
            f"{n_nan} NaN, {n_inf} Inf, max |x| = {largest:.3g}"
        )
        for line in offenders[:10]:
            print(f"           - {line}")
        if len(offenders) > 10:
            print(f"           ... and {len(offenders) - 10} more")

    print(f"         verdict: {'usable' if ok else 'DO NOT RESUME FROM THIS'}")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="checkpoint files to check")
    args = p.parse_args(argv)
    return 0 if all([check(Path(c)) for c in args.checkpoints]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
