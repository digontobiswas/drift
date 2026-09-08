"""`ckpt_health` decides whether a run gets thrown away, so its false alarms cost as
much as its misses.

The script exists to answer one question during a debugging session: is the checkpoint
every job resumes from still sound, or is the damage being copied forward? A "DAMAGED"
verdict sends someone back to an older checkpoint and discards the training since. It
gave exactly that verdict on three healthy checkpoints on its first real use, because the
divergence heuristic was pointed at AdamW's step counter -- a number that grows by one per
optimizer step and had reached 89,700. That is pinned first below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.ckpt_health import check, main


def _ckpt(path: Path, *, weight=None, exp_avg_sq=None, step=22530.0) -> Path:
    torch.save(
        {
            "epoch": 4, "global_step": 90400, "batch_in_epoch": 280,
            "model": {"enc.weight": torch.randn(4) if weight is None else weight},
            "optimizer": {
                "state": {
                    0: {
                        "exp_avg": torch.zeros(4),
                        "exp_avg_sq": torch.rand(4) if exp_avg_sq is None else exp_avg_sq,
                        "step": torch.tensor(step),
                    }
                }
            },
        },
        path,
    )
    return path


class TestHealthyCheckpointsPass:
    def test_a_large_step_counter_is_not_damage(self, tmp_path) -> None:
        """The false positive this script shipped with. AdamW's `step` is a count, not a
        value: at 89,700 steps it is doing exactly what it should. Reporting that as
        divergence condemned three checkpoints that were provably fine -- one of them the
        checkpoint a successful evaluation had already been run from.
        """
        assert check(_ckpt(tmp_path / "c.pth", step=89_700.0)) is True

    def test_ordinary_checkpoint_is_usable(self, tmp_path) -> None:
        assert check(_ckpt(tmp_path / "c.pth")) is True

    def test_weights_of_normal_size_are_not_flagged(self, tmp_path) -> None:
        """Real weights here reach a few hundred; the threshold is for divergence, orders
        of magnitude above that, not for merely large-looking values."""
        assert check(_ckpt(tmp_path / "c.pth", weight=torch.tensor([386.0, -255.0]))) is True


class TestDamageIsCaught:
    def test_inf_in_the_second_moment_buffer(self, tmp_path) -> None:
        """The realistic failure: one infinite gradient poisons `exp_avg_sq`, which then
        divides into every future update. The weights still look perfectly finite, so
        checking them alone would miss it entirely."""
        bad = torch.tensor([1.0, float("inf"), 2.0, 3.0])
        assert check(_ckpt(tmp_path / "c.pth", exp_avg_sq=bad)) is False

    def test_nan_in_the_weights(self, tmp_path) -> None:
        bad = torch.tensor([1.0, float("nan"), 2.0, 3.0])
        assert check(_ckpt(tmp_path / "c.pth", weight=bad)) is False

    def test_finite_but_diverged_weights(self, tmp_path) -> None:
        """A run can be unusable before any value is literally non-finite."""
        assert check(_ckpt(tmp_path / "c.pth", weight=torch.tensor([1e6, 1.0]))) is False

    def test_a_nan_counter_is_still_caught(self, tmp_path) -> None:
        """Exempting the counter from the magnitude check must not exempt it from the
        finiteness check -- a NaN step count means the optimizer state is nonsense."""
        assert check(_ckpt(tmp_path / "c.pth", step=float("nan"))) is False


class TestExitStatus:
    def test_zero_only_when_every_checkpoint_is_clean(self, tmp_path) -> None:
        good = _ckpt(tmp_path / "good.pth", step=89_700.0)
        bad = _ckpt(tmp_path / "bad.pth", weight=torch.tensor([float("nan"), 1.0]))
        assert main([str(good)]) == 0
        assert main([str(good), str(bad)]) == 1

    def test_a_missing_file_fails_rather_than_passing_silently(self, tmp_path) -> None:
        assert main([str(tmp_path / "nope.pth")]) == 1
