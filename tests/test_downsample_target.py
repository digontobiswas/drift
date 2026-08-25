"""`downsample_target` must survive two memory optimisations without changing a label.

Two things were changed for memory reasons and neither may alter results:

  1. The per-block class histogram was `F.one_hot(flat, num_classes).sum(dim=1)`,
     which allocates an `(M, K, num_classes)` int64 intermediate -- ~1.5 GB for a
     single real Cam4DOcc batch element, on its own enough to OOM a 16 GB V100. It
     is now a short loop of `(flat == c).sum(dim=1)` comparisons.
  2. The full-resolution target is no longer widened to int64 internally, and now
     arrives from the dataset as `uint8` (~500 MB -> ~63 MB per batch element).

`_reference_impl` below is the ORIGINAL one-hot formulation, kept verbatim as the
oracle. If the optimised path ever diverges from it, these tests fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from drift.losses.occupancy import _IGNORE_INDEX, downsample_target


def _reference_impl(target: torch.Tensor, ratio: int) -> torch.Tensor:
    """The original one-hot implementation, used only as a correctness oracle."""
    *lead, X, Y, Z = target.shape
    Xl, Yl, Zl = X // ratio, Y // ratio, Z // ratio
    num_classes = int(target.max().item()) + 1 if target.numel() > 0 else 1
    num_classes = max(num_classes, 1)

    blocks = target.reshape(*lead, Xl, ratio, Yl, ratio, Zl, ratio)
    n_lead = len(lead)
    perm = list(range(n_lead)) + [
        n_lead + 0, n_lead + 2, n_lead + 4, n_lead + 1, n_lead + 3, n_lead + 5,
    ]
    blocks = blocks.permute(*perm).contiguous()
    K = ratio ** 3
    flat = blocks.reshape(-1, K).long()

    counts = F.one_hot(flat, num_classes=num_classes).sum(dim=1)
    nonempty_count = K - counts[:, 0]
    counts_nonzero = counts.clone()
    counts_nonzero[:, 0] = -1
    mode_count, mode_class = counts_nonzero.max(dim=1)

    all_empty = nonempty_count == 0
    has_majority = (mode_count * 2 > nonempty_count) & (~all_empty)

    out = torch.full_like(mode_class, fill_value=_IGNORE_INDEX)
    out = torch.where(all_empty, torch.zeros_like(out), out)
    out = torch.where(has_majority, mode_class, out)
    return out.reshape(*lead, Xl, Yl, Zl).long()


class TestDownsampleTargetEquivalence:
    def test_matches_reference_across_random_grids(self) -> None:
        torch.manual_seed(0)
        for num_classes in (3, 17):
            for shape in ((2, 3, 8, 8, 4), (1, 16, 16, 8), (4, 4, 4)):
                target = torch.randint(0, num_classes, shape, dtype=torch.long)
                got = downsample_target(target, ratio=4)
                want = _reference_impl(target, ratio=4)
                assert torch.equal(got, want), (
                    f"mismatch for num_classes={num_classes} shape={shape}: "
                    f"{(got != want).sum().item()} of {want.numel()} blocks differ"
                )

    def test_uint8_input_matches_int64_input(self) -> None:
        """The dataset now hands over uint8; that must not change a single label."""
        torch.manual_seed(1)
        target_long = torch.randint(0, 3, (2, 3, 8, 8, 4), dtype=torch.long)
        target_u8 = target_long.to(torch.uint8)
        assert torch.equal(
            downsample_target(target_u8, ratio=4), downsample_target(target_long, ratio=4)
        )

    def test_returns_long_for_cross_entropy(self) -> None:
        """F.cross_entropy rejects a non-Long target, so the output must be widened
        regardless of how compact the input was."""
        target = torch.randint(0, 3, (1, 8, 8, 4), dtype=torch.uint8)
        assert downsample_target(target, ratio=4).dtype == torch.long

    def test_ignore_index_survives_a_uint8_round_trip(self) -> None:
        """255 is the ignore index and also uint8's maximum -- a block with no strict
        majority must come back as 255, not silently wrap."""
        # One 4x4x4 block, evenly split 32/32 between classes 1 and 2: no majority.
        block = torch.cat(
            [torch.ones(32, dtype=torch.uint8), torch.full((32,), 2, dtype=torch.uint8)]
        ).reshape(4, 4, 4)
        out = downsample_target(block, ratio=4)
        assert out.item() == _IGNORE_INDEX

    def test_all_empty_block_is_free_not_ignored(self) -> None:
        block = torch.zeros((4, 4, 4), dtype=torch.uint8)
        assert downsample_target(block, ratio=4).item() == 0
