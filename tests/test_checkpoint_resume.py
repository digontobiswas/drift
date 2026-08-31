"""Mid-epoch checkpointing must survive a crash without losing an epoch of work.

Job 1153168 segfaulted three hours into epoch 0 and left NO checkpoint at all,
because saving only happened at epoch boundaries and a real epoch is ~8 hours.
These tests pin the fix:

  * `save_checkpoint` records `batch_in_epoch`, so a mid-epoch snapshot knows how
    far into the epoch it is;
  * `load_checkpoint` returns it, defaulting to 0 for older checkpoints written
    before the field existed (those simply restart their epoch, as before);
  * the write is atomic, so a crash during `torch.save` cannot leave a truncated
    `latest.pth` -- which would be worse than no checkpoint, since the Slurm
    scripts auto-resume from that exact file on every requeue.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs import get_config
from tools.train import load_checkpoint, save_checkpoint


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _roundtrip(tmp_path, epoch: int, step: int, batch_in_epoch: int):
    cfg = get_config("tiny")
    model, model2 = _Tiny(), _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    # Give the optimizer real state so the round-trip is not vacuous.
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

    path = tmp_path / "latest.pth"
    save_checkpoint(path, model, opt, epoch, step, cfg, batch_in_epoch=batch_in_epoch)
    return path, model, model2, opt2


class TestCheckpointResume:
    def test_batch_in_epoch_survives_roundtrip(self, tmp_path) -> None:
        path, model, model2, opt2 = _roundtrip(tmp_path, epoch=2, step=1234, batch_in_epoch=987)
        got_epoch, got_step, got_batch = load_checkpoint(
            str(path), model2, opt2, torch.device("cpu")
        )
        assert (got_epoch, got_step, got_batch) == (2, 1234, 987)
        for a, b in zip(model.parameters(), model2.parameters()):
            assert torch.equal(a, b), "weights did not survive the checkpoint round-trip"

    def test_missing_batch_in_epoch_defaults_to_zero(self, tmp_path) -> None:
        """An older checkpoint has no `batch_in_epoch`; it must restart its epoch
        rather than raising or resuming at a garbage offset."""
        path, _, model2, opt2 = _roundtrip(tmp_path, epoch=1, step=50, batch_in_epoch=7)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["batch_in_epoch"]
        torch.save(ckpt, path)

        _, _, got_batch = load_checkpoint(str(path), model2, opt2, torch.device("cpu"))
        assert got_batch == 0

    def test_write_is_atomic_leaving_no_partial_file(self, tmp_path) -> None:
        """The checkpoint must be renamed into place, never written in-place, so an
        interrupted save cannot corrupt the file the Slurm scripts resume from."""
        path, _, model2, opt2 = _roundtrip(tmp_path, epoch=0, step=10, batch_in_epoch=3)
        assert path.exists()
        leftovers = list(tmp_path.glob("*.tmp"))
        assert not leftovers, f"temporary checkpoint files left behind: {leftovers}"
        # And the result is loadable, i.e. the rename published a complete file.
        assert load_checkpoint(str(path), model2, opt2, torch.device("cpu"))[2] == 3

    def test_early_stop_is_not_recorded_as_a_finished_epoch(self, tmp_path) -> None:
        """An interrupted epoch must save `(epoch, batch_in_epoch=n)`, never
        `(epoch+1, 0)`.

        Recording an interrupted epoch as finished is silently destructive: the
        follow-up job would skip every remaining batch of that epoch, so a run
        interrupted once per epoch would train on a fraction of the data while the
        logs still claimed the configured number of epochs. This drives the real
        entry point rather than the helper, because the bug lived in the training
        loop's save-on-break path, not in `save_checkpoint` itself.
        """
        import subprocess

        repo = Path(__file__).resolve().parents[1]
        ckpt_dir = tmp_path / "ck"
        proc = subprocess.run(
            [
                sys.executable, str(repo / "tools" / "train.py"),
                "--config", "tiny", "--max-iters", "4", "--batch-size", "1",
                "--num-workers", "0", "--log-interval", "100",
                "--ckpt-interval-steps", "0", "--device", "cpu",
                "--ckpt-dir", str(ckpt_dir),
            ],
            cwd=str(repo), capture_output=True, text=True, timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        ckpt = torch.load(ckpt_dir / "latest.pth", map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 0, (
            f"early stop recorded epoch={ckpt['epoch']}; resume would skip the rest of epoch 0"
        )
        assert ckpt["batch_in_epoch"] == 4, ckpt

    def test_overwrite_keeps_previous_checkpoint_loadable(self, tmp_path) -> None:
        """Re-saving over an existing latest.pth must not corrupt it -- periodic saves
        overwrite the same path hundreds of times per epoch."""
        path, _, model2, opt2 = _roundtrip(tmp_path, epoch=0, step=10, batch_in_epoch=3)
        cfg = get_config("tiny")
        m, o = _Tiny(), None
        o = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for step in (20, 30, 40):
            save_checkpoint(path, m, o, 0, step, cfg, batch_in_epoch=step)
            assert load_checkpoint(str(path), model2, opt2, torch.device("cpu"))[1] == step
