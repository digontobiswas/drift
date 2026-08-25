"""Determinism helper. See docs/DESIGN_SPEC.md SS7.

A single entry point for seeding every RNG DRIFT touches (Python's `random`,
`numpy`, and `torch`, CPU and CUDA), used by `tools/train.py`,
`tools/eval.py`, and the test suite so runs are reproducible given the same
seed and hardware.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices) RNGs.

    Args:
        seed: Seed value shared by every RNG.
        deterministic: If True, additionally request cuDNN-deterministic
            algorithms (`torch.backends.cudnn.deterministic = True`,
            `benchmark = False`) and set `PYTHONHASHSEED`. This can slow down
            training measurably on GPU; default False leaves cuDNN free to
            pick the fastest (non-deterministic) algorithms.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
