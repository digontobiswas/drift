"""Print `epoch global_step` from a checkpoint, for shell scripts to branch on.

Used by the Slurm scripts' self-requeue logic: after a run ends they need to know
whether training actually finished (epoch >= total) and whether it made any
progress at all (step advanced), without hand-rolling `torch.load` in bash.

    python tools/ckpt_info.py path/to/latest.pth
    -> "7 65432"

A missing, unreadable, or truncated checkpoint prints "-1 -1" and exits 0 rather
than raising: the caller is a `set -euo pipefail` shell script, and a checkpoint
that cannot be read is a normal state (first run, or a crash mid-write) that
should be handled as "no progress", not as a scripting error.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) != 2:
        print("-1 -1")
        return
    try:
        import torch

        ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
        print(f"{int(ckpt.get('epoch', -1))} {int(ckpt.get('global_step', -1))}")
    except Exception:
        print("-1 -1")


if __name__ == "__main__":
    main()
