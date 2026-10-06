"""A checkpoint is the only thing a two-week training run leaves behind.

This pins one rule: a non-finite model is never written to disk. The run it comes from is
already lost, but the checkpoint before it is not -- unless the save overwrites it.

That is exactly what happened here. A twelve-epoch run finished, and every one of its 587
weight tensors was NaN. Because an epoch boundary writes `epoch_N.pth` and `latest.pth` from
the same weights, both copies of the last good state went in one breath, and the only
survivor was an unrelated backup four epochs earlier. The paper was written from that backup,
on a quarter of the training that had actually been done.

Nothing detected it at the time: `torch.save` writes NaN as happily as anything else, the job
exited zero, and Slurm reported COMPLETED.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from configs import get_config
from tools.train import _first_non_finite, save_checkpoint


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _setup():
    model = _Model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    return model, opt, get_config("tiny")


class TestFindingTheBadTensor:
    def test_a_clean_state_dict_reports_nothing(self) -> None:
        model, _, _ = _setup()
        assert _first_non_finite(model.state_dict(), "model") is None

    def test_a_nan_weight_is_found_and_named(self) -> None:
        """The name matters: with 587 tensors, "something is NaN" is not a usable report."""
        model, _, _ = _setup()
        state = model.state_dict()
        state["lin.weight"][0, 0] = float("nan")
        assert _first_non_finite(state, "model") == "model.lin.weight"

    def test_an_inf_is_caught_too(self) -> None:
        model, _, _ = _setup()
        state = model.state_dict()
        state["lin.bias"][0] = float("inf")
        assert _first_non_finite(state, "model") == "model.lin.bias"

    def test_optimizer_moment_buffers_are_reached(self) -> None:
        """AdamW nests its buffers under integer parameter ids. A walk that stops at the
        top level would call a poisoned `exp_avg_sq` clean -- and that buffer divides into
        every future update, so the damage is permanent and invisible in the weights."""
        model, opt, _ = _setup()
        state = opt.state_dict()
        state["state"][0]["exp_avg_sq"][0, 0] = float("nan")
        assert _first_non_finite(state, "optimizer") == "optimizer.state.0.exp_avg_sq"

    def test_integer_tensors_are_not_misread(self) -> None:
        """`torch.isfinite` rejects integer tensors outright, so the walk must skip them
        rather than raise on an ordinary checkpoint."""
        assert _first_non_finite({"step": torch.tensor([3, 4])}, "model") is None


class TestRefusingToWrite:
    def test_a_healthy_checkpoint_is_written(self, tmp_path) -> None:
        model, opt, cfg = _setup()
        path = tmp_path / "latest.pth"
        assert save_checkpoint(path, model, opt, 1, 100, cfg) is True
        assert torch.load(path, map_location="cpu", weights_only=False)["global_step"] == 100

    def test_a_nan_model_is_refused_and_the_old_file_survives(self, tmp_path) -> None:
        """The whole point. The good checkpoint must still be on disk afterwards."""
        model, opt, cfg = _setup()
        path = tmp_path / "latest.pth"
        assert save_checkpoint(path, model, opt, 1, 100, cfg) is True

        with torch.no_grad():
            model.lin.weight[0, 0] = float("nan")
        assert save_checkpoint(path, model, opt, 2, 200, cfg) is False

        kept = torch.load(path, map_location="cpu", weights_only=False)
        assert kept["global_step"] == 100, "the NaN save overwrote the good checkpoint"
        assert torch.isfinite(kept["model"]["lin.weight"]).all()

    def test_a_nan_optimizer_is_refused_even_with_finite_weights(self, tmp_path) -> None:
        """Weights can look perfectly healthy while the second-moment buffer is already
        poisoned; resuming from that produces NaN weights within a few steps."""
        model, opt, cfg = _setup()
        opt.state_dict()["state"][0]["exp_avg_sq"][0, 0] = float("inf")
        assert save_checkpoint(tmp_path / "latest.pth", model, opt, 1, 100, cfg) is False

    def test_a_refused_save_leaves_no_temporary_file(self, tmp_path) -> None:
        """The write is staged through `.tmp`; a refusal must not leave one lying around
        for a later run to trip over."""
        model, opt, cfg = _setup()
        with torch.no_grad():
            model.lin.bias[0] = float("nan")
        save_checkpoint(tmp_path / "latest.pth", model, opt, 1, 100, cfg)
        assert not list(tmp_path.glob("*.tmp"))
        assert not (tmp_path / "latest.pth").exists()

    def test_the_refusal_names_the_tensor_on_stderr(self, tmp_path, capsys) -> None:
        """Whoever reads the log at 3am needs to know which tensor went, not just that
        something did."""
        model, opt, cfg = _setup()
        with torch.no_grad():
            model.lin.weight[1, 1] = float("nan")
        save_checkpoint(tmp_path / "latest.pth", model, opt, 1, 100, cfg)
        err = capsys.readouterr().err
        assert "REFUSING" in err and "lin.weight" in err


class TestLossGuardsOnARealRun:
    """The cheap guards, exercised through `tools/train.py` itself on the tiny preset.

    A non-finite loss reaching `backward()` is how one bad batch destroys a two-week run:
    without AMP there is no GradScaler to veto the step, `clip_grad_norm_` scales the
    gradients by a NaN norm, and `optimizer.step()` writes NaN into every parameter.
    """

    def _run(self, tmp_path, extra=()):
        import subprocess

        return subprocess.run(
            [sys.executable, "tools/train.py", "--config", "tiny", "--dataset", "synthetic",
             "--device", "cpu", "--no-amp", "--epochs", "1", "--max-iters", "3",
             "--batch-size", "1", "--num-workers", "0", "--log-interval", "1",
             "--ckpt-dir", str(tmp_path), *extra],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=900,
        )

    def test_a_short_run_reports_its_skip_counters(self, tmp_path) -> None:
        """Zeros are reported too: a log that says nothing cannot distinguish a clean run
        from one that never counted."""
        result = self._run(tmp_path)
        assert result.returncode == 0, result.stderr[-2000:]
        assert "non-finite batches skipped: 0" in result.stdout
        assert "weight rollbacks: 0" in result.stdout

    def test_the_run_leaves_a_finite_checkpoint(self, tmp_path) -> None:
        self._run(tmp_path)
        ckpt = torch.load(tmp_path / "latest.pth", map_location="cpu", weights_only=False)
        for name, tensor in ckpt["model"].items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                assert torch.isfinite(tensor).all(), f"{name} is not finite"


class TestRecoveryPolicy:
    """`--nan-policy` decides whether a poisoned run dies or rolls back.

    Rolling back is only safe because `save_checkpoint` refuses non-finite writes: the file
    it reloads is guaranteed to be a model that was finite when written. The bound on
    retries matters as much as the retry -- a run reliably producing NaN would otherwise
    hold a scarce GPU for days, recovering and re-poisoning itself.
    """

    def test_both_policies_are_accepted(self) -> None:
        from tools.train import parse_args

        assert parse_args(["--nan-policy", "abort"]).nan_policy == "abort"
        assert parse_args([]).nan_policy == "recover", "recovery should be the default"
        assert parse_args([]).max_nan_recoveries == 3

    def test_an_unknown_policy_is_rejected(self) -> None:
        from tools.train import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--nan-policy", "ignore"])


class TestRollbackRestoresAUsableModel:
    """The round-trip recovery depends on: refuse the bad write, reload the good file.

    Recovery is only sound because the file being reloaded is guaranteed finite -- nothing
    else has ever been written. This pins that the reload actually restores the earlier
    weights rather than leaving the poisoned ones in place.
    """

    def test_reloading_after_a_refused_save_undoes_the_damage(self, tmp_path) -> None:
        from tools.train import load_checkpoint

        model, opt, cfg = _setup()
        path = tmp_path / "latest.pth"
        assert save_checkpoint(path, model, opt, 1, 100, cfg) is True
        good = model.lin.weight.detach().clone()

        with torch.no_grad():
            model.lin.weight.fill_(float("nan"))
        assert save_checkpoint(path, model, opt, 1, 150, cfg) is False
        assert not torch.isfinite(model.lin.weight).all(), "precondition: weights are poisoned"

        load_checkpoint(str(path), model, opt, torch.device("cpu"), world_size=1)
        assert torch.isfinite(model.lin.weight).all()
        assert torch.allclose(model.lin.weight, good)

    def test_the_optimizer_is_restored_too(self, tmp_path) -> None:
        """Reloading weights but keeping a poisoned optimizer state would re-poison them
        within a few steps, which would look like the recovery simply not working."""
        from tools.train import load_checkpoint

        model, opt, cfg = _setup()
        path = tmp_path / "latest.pth"
        save_checkpoint(path, model, opt, 1, 100, cfg)

        opt.state_dict()["state"][0]["exp_avg"].fill_(float("nan"))
        load_checkpoint(str(path), model, opt, torch.device("cpu"), world_size=1)
        assert torch.isfinite(opt.state_dict()["state"][0]["exp_avg"]).all()


class TestTheGuardIsVisiblyRunning:
    """A guard that only speaks when it fails cannot be distinguished from one that was
    accidentally removed. Before committing two weeks of GPU time, "no warning appeared" is
    not evidence -- so the successful path reports what it checked, and the whole cycle can
    be provoked on demand with `--inject-nan-at-step`.
    """

    def _run(self, tmp_path, extra):
        import subprocess

        return subprocess.run(
            [sys.executable, "tools/train.py", "--config", "tiny", "--dataset", "synthetic",
             "--device", "cpu", "--no-amp", "--epochs", "1", "--batch-size", "1",
             "--num-workers", "0", "--log-interval", "50", "--ckpt-dir", str(tmp_path), *extra],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=900,
        )

    def test_a_passing_check_announces_itself(self, tmp_path) -> None:
        out = self._run(tmp_path, ["--max-iters", "2", "--ckpt-interval-steps", "1"]).stdout
        assert "finiteness check passed" in out
        assert "tensors, max |w| =" in out, "the report should show what was measured"

    def test_injected_nan_is_refused_and_recovered_from(self, tmp_path) -> None:
        """The whole cycle on a real model: poison, refuse, roll back, carry on."""
        result = self._run(tmp_path, [
            "--max-iters", "4", "--ckpt-interval-steps", "1", "--inject-nan-at-step", "3",
        ])
        assert result.returncode == 0, result.stderr[-2000:]
        assert "REFUSING to write" in result.stderr
        assert "recovered from non-finite weights" in result.stdout
        assert "weight rollbacks: 1" in result.stdout

    def test_the_surviving_checkpoint_is_finite_and_newer_than_the_poisoning(self, tmp_path) -> None:
        """Recovery is only worth anything if the run goes on to write good checkpoints."""
        self._run(tmp_path, [
            "--max-iters", "4", "--ckpt-interval-steps", "1", "--inject-nan-at-step", "3",
        ])
        ckpt = torch.load(tmp_path / "latest.pth", map_location="cpu", weights_only=False)
        assert ckpt["global_step"] > 3, "training did not continue past the rollback"
        for name, t in ckpt["model"].items():
            if torch.is_tensor(t) and t.is_floating_point():
                assert torch.isfinite(t).all(), f"{name} is not finite"

    def test_injection_is_off_unless_asked_for(self) -> None:
        from tools.train import parse_args

        assert parse_args([]).inject_nan_at_step is None
