"""The Slurm scripts only ever run on a cluster, so their bugs surface hours after submission.

Nothing else in this repository exercises them. A mistake here is not caught by a failing
import or a red test -- it is caught by six array tasks dying on a compute node a day later,
after the queue wait, with one line of error in a log file nobody is watching.

That is exactly how the ablation grid was lost: every task failed inside a minute with

    line 60: RESUME_ARG[@]: unbound variable

The compute nodes run RHEL7-era bash 4.2, where expanding an EMPTY array as "${ARR[@]}" under
`set -u` is an error rather than an expansion to nothing; bash 4.4 changed that. The scripts
are written and read on bash 5, where the pattern is harmless, so it looks correct right up
until it runs where it matters. Both checks below are about that gap: what the login shell
tolerates is not what the compute node will.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SLURM_DIR = Path(__file__).resolve().parents[1] / "slurm"
SCRIPTS = sorted(SLURM_DIR.glob("*.slurm"))


def test_there_are_scripts_to_check() -> None:
    """A glob that silently matches nothing turns every test below into a free pass."""
    assert SCRIPTS, f"no .slurm files found in {SLURM_DIR}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_valid_bash(script: Path) -> None:
    """A syntax error costs a queue wait to discover. `bash -n` costs milliseconds."""
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"{script.name}: {result.stderr.strip()}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_empty_array_is_expanded(script: Path) -> None:
    """The bug that killed the ablation grid, pinned as a rule rather than a fix.

    An array declared empty (`ARR=()`) and later expanded as "${ARR[@]}" is fine on bash 4.4+
    and fatal on the 4.2 the compute nodes run. Since the failure depends on a shell version
    this test suite will never run under, the pattern itself is banned.

    The safe shapes: build one always-non-empty array and append to it conditionally, or use a
    plain string with `${VAR:+--flag "$VAR"}` as slurm/02_train.slurm does.
    """
    text = script.read_text()
    declared_empty = set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=\(\s*\)\s*$", text, re.M))
    offenders = [
        name for name in declared_empty
        if re.search(r'"\$\{' + re.escape(name) + r'\[@\]\}"', text)
    ]
    assert not offenders, (
        f"{script.name} expands possibly-empty array(s) {offenders} as \"${{NAME[@]}}\". "
        "On the cluster's bash 4.2 that is an 'unbound variable' error under set -u. "
        "Append to one non-empty array instead, or use ${VAR:+--flag \"$VAR\"}."
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_strict_mode_is_on(script: Path) -> None:
    """`set -euo pipefail` is what makes a broken step stop the job instead of letting it
    carry on and write a half-finished result that looks real."""
    assert re.search(r"^set -euo pipefail\s*$", script.read_text(), re.M), (
        f"{script.name} does not enable strict mode"
    )


def _sbatch_directives(script: Path) -> dict:
    return dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)", script.read_text(), re.M))


def test_ablations_run_under_the_main_runs_exact_budget() -> None:
    """The main 02_train run is the full-model row of the ablation table, so every ablation
    must train under identical conditions. When 03 asked for 2 GPUs against 02's 1, DDP
    halved the steps per epoch and the table showed every ablation beating the full model."""
    main = _sbatch_directives(SLURM_DIR / "02_train.slurm")
    abl = _sbatch_directives(SLURM_DIR / "03_ablations.slurm")
    for key in ("gres", "cpus-per-task", "mem", "time", "partition"):
        assert abl[key] == main[key], f"--{key}: ablations {abl[key]!r} vs main run {main[key]!r}"


def test_ablation_grid_does_not_retrain_the_full_model_by_default() -> None:
    """Task 0 would write into the main run's checkpoint directory."""
    assert _sbatch_directives(SLURM_DIR / "03_ablations.slurm")["array"] == "1-6"
    assert "ABL_INCLUDE_FULL" in (SLURM_DIR / "03_ablations.slurm").read_text()


@pytest.mark.parametrize("name", ["02_train.slurm", "03_ablations.slurm"])
def test_training_chains_tolerate_stalled_runs(name: str) -> None:
    """Crashes here are stochastic; a run that dies before its first checkpoint is usually
    unlucky, not doomed. A chain that stops on the first one dies silently overnight."""
    assert "DRIFT_MAX_STALLS" in (SLURM_DIR / name).read_text()
