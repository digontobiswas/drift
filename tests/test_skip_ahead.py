"""`skip_ahead` edits the one file the whole run depends on, so its guards are the test.

Training on this cluster keeps hitting stretches of an epoch it cannot get through: every
resume dies within a few dozen batches and banks nothing, and the requeue chain grinds in
place until it gives up. `tools/skip_ahead.py` steps the resume position over such a
stretch. That makes it the only thing in the repo that writes a checkpoint by hand, on the
exact file every job auto-resumes from -- a wrong write here does not fail loudly, it
quietly changes what the run trains on.

The destructive direction is skipping too much. `--next-epoch` on a checkpoint that sits
at the start of an epoch would discard a whole untrained epoch while printing the same
calm message as abandoning a few hundred batches; that was real, and is pinned below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.skip_ahead import main

EPOCH_BATCHES = 22530


def _ckpt(tmp_path, *, epoch=3, step=89800, batch=22210) -> Path:
    path = tmp_path / "latest.pth"
    torch.save(
        {
            "model": {}, "optimizer": {}, "epoch": epoch, "global_step": step,
            "batch_in_epoch": batch, "world_size": 1, "config_name": "cam4docc_gmo",
        },
        path,
    )
    return path


def _read(path: Path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    return c["epoch"], c["batch_in_epoch"], c["global_step"]


class TestSkipWithinAnEpoch:
    def test_by_advances_batch_and_step_together(self, tmp_path) -> None:
        """`global_step` drives the LR schedule, so skipping batches without advancing it
        would leave the learning rate lagging the run by exactly the skipped amount --
        silently, and for the rest of training."""
        path = _ckpt(tmp_path)
        assert main([str(path), "--by", "300"]) == 0
        assert _read(path) == (3, 22510, 90100)

    def test_to_sets_an_exact_position(self, tmp_path) -> None:
        path = _ckpt(tmp_path, batch=19560, step=87150)
        assert main([str(path), "--to", "19860"]) == 0
        assert _read(path) == (3, 19860, 87450)

    def test_refuses_to_move_backwards(self, tmp_path) -> None:
        """Resuming earlier than the checkpoint retrains batches already done and rewinds
        the step counter, which corrupts the schedule rather than merely wasting time."""
        path = _ckpt(tmp_path)
        before = _read(path)
        assert main([str(path), "--to", "100"]) == 1
        assert _read(path) == before, "a refused skip must leave the checkpoint untouched"

    def test_refuses_a_position_past_the_end_of_the_epoch(self, tmp_path) -> None:
        path = _ckpt(tmp_path)
        before = _read(path)
        assert main([str(path), "--to", str(EPOCH_BATCHES + 10)]) == 1
        assert _read(path) == before


class TestNextEpoch:
    def test_rolls_to_the_state_an_epoch_end_save_would_have_written(self, tmp_path) -> None:
        """The point of rolling forward rather than clamping to the last batch is that
        nothing downstream should be able to tell this from an epoch that finished on its
        own: `(epoch + 1, batch_in_epoch=0)` is exactly what `save_checkpoint` writes at an
        epoch boundary."""
        path = _ckpt(tmp_path, epoch=3, step=89800, batch=22210)
        assert main([str(path), "--next-epoch"]) == 0
        assert _read(path) == (4, 0, 89800 + (EPOCH_BATCHES - 22210))

    def test_refuses_to_discard_an_epoch_that_has_barely_started(self, tmp_path) -> None:
        """The bug this pins: at `batch_in_epoch=0` the roll-forward discarded all 22530
        batches of an untrained epoch and reported it in the same tone as abandoning a few
        hundred. Nothing downstream would flag it -- the epoch counter simply moves on and
        the loss keeps looking normal -- so the refusal has to live here.

        `batch_in_epoch=0` is also the state an epoch-end save leaves behind, i.e. the one
        most likely to be sitting in `latest.pth` when someone reaches for this flag.
        """
        path = _ckpt(tmp_path, epoch=4, step=90120, batch=0)
        before = _read(path)
        assert main([str(path), "--next-epoch"]) == 1
        assert _read(path) == before, "an entire epoch was discarded"

    def test_force_still_allows_it_when_that_is_genuinely_wanted(self, tmp_path) -> None:
        path = _ckpt(tmp_path, epoch=4, step=90120, batch=0)
        assert main([str(path), "--next-epoch", "--force"]) == 0
        assert _read(path) == (5, 0, 90120 + EPOCH_BATCHES)

    def test_a_tail_end_is_allowed_without_force(self, tmp_path) -> None:
        """Right at the 5% boundary, so the threshold itself is exercised rather than only
        the comfortable cases either side of it."""
        limit = int(EPOCH_BATCHES * 0.05)
        path = _ckpt(tmp_path, epoch=2, step=50000, batch=EPOCH_BATCHES - limit)
        assert main([str(path), "--next-epoch"]) == 0
        assert _read(path) == (3, 0, 50000 + limit)


class TestBackupAndRestore:
    def test_restore_undoes_a_skip_exactly(self, tmp_path) -> None:
        path = _ckpt(tmp_path)
        before = _read(path)
        assert main([str(path), "--by", "300"]) == 0
        assert _read(path) != before
        assert main([str(path), "--restore"]) == 0
        assert _read(path) == before

    def test_a_skip_leaves_no_partial_file_behind(self, tmp_path) -> None:
        """The checkpoint is written via a temporary file and renamed, because this is the
        file every requeued job resumes from: a half-written one turns a recoverable
        interruption into a run that fails instantly on every retry."""
        path = _ckpt(tmp_path)
        assert main([str(path), "--by", "300"]) == 0
        assert not list(tmp_path.glob("*.tmp"))
        assert (tmp_path / "latest.pth.bak").exists()

    def test_restore_without_a_backup_fails_instead_of_pretending(self, tmp_path) -> None:
        path = _ckpt(tmp_path)
        assert main([str(path), "--restore"]) == 1
