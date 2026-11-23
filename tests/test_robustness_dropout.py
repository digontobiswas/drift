"""The sensor-dropout study must run on the GPU.

`evaluate_robustness` seeds a CPU `torch.Generator` per scenario, while the batch masks live
on the GPU. `apply_modality_dropout` used to draw with `device=mask.device`, which torch
rejects for a CPU generator -- the robustness job (1232624) finished its clean scenario and
then died on the first dropout one, so the CMLI robustness claim had no numbers behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from drift.metrics.robustness import apply_modality_dropout


def _batch(device: str = "cpu"):
    return {"lidar_mask": torch.ones(4, 3, device=device), "cam_mask": torch.ones(4, 3, device=device),
            "imgs": "untouched"}


def test_drops_roughly_the_requested_fraction_and_leaves_the_rest_alone() -> None:
    torch.manual_seed(0)
    big = {"lidar_mask": torch.ones(200, 3), "cam_mask": torch.ones(200, 3)}
    out = apply_modality_dropout(big, 0.0, 0.5, torch.Generator().manual_seed(1))
    assert torch.equal(out["lidar_mask"], big["lidar_mask"])
    assert 0.4 < 1 - out["cam_mask"].mean().item() < 0.6


def test_a_seeded_generator_gives_the_same_frames_every_time() -> None:
    a = apply_modality_dropout(_batch(), 0.5, 0.5, torch.Generator().manual_seed(7))
    b = apply_modality_dropout(_batch(), 0.5, 0.5, torch.Generator().manual_seed(7))
    assert torch.equal(a["lidar_mask"], b["lidar_mask"]) and torch.equal(a["cam_mask"], b["cam_mask"])


def test_other_batch_entries_pass_through() -> None:
    assert apply_modality_dropout(_batch(), 1.0, 1.0)["imgs"] == "untouched"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to reproduce")
def test_cpu_generator_with_gpu_masks() -> None:
    """The exact failing combination, and the same frames as on the CPU."""
    gpu = apply_modality_dropout(_batch("cuda"), 0.5, 0.5, torch.Generator().manual_seed(3))
    cpu = apply_modality_dropout(_batch("cpu"), 0.5, 0.5, torch.Generator().manual_seed(3))
    assert gpu["cam_mask"].device.type == "cuda"
    assert torch.equal(gpu["cam_mask"].cpu(), cpu["cam_mask"])
