"""The CSVs feed a paper table, so a wrong number here is worse than a missing one.

`tools/results_to_csv.py` is the only thing standing between the evaluation JSONs and a
results table. Three of its jobs can fail silently -- produce a plausible-looking CSV that is
wrong -- and those are what the tests below pin:

- **Scale.** Published Cam4DOcc IoU is a percentage (31.30); `tools/eval.py` reports a fraction
  (0.1357). A comparison table that forgets to convert looks finished and is off by 100x.
- **Shape drift.** `tools/run_robustness.py` writes `payload["table"]` and
  `tools/benchmark_latency.py` writes nested `latency`/`memory`/`params`/`flops`. Reading a key
  that no longer exists yields an empty column, not an error, which is how
  `tools/collect_results.py` came to render dashes for two whole sections.
- **Meaning.** Class index 2 is GMO only in the 3-class `cam4docc_gmo` preset. Labelling index 2
  as GMO in a 17-class run would put "vegetation" in a table column headed "movable objects".
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.published_baselines import as_percent, for_protocol
from tools.results_to_csv import (
    build_baseline_comparison_rows,
    build_calibration_rows,
    build_latency_rows,
    build_per_class_rows,
    build_per_horizon_rows,
    build_robustness_rows,
    build_summary_rows,
    main,
)

CONFIGS = [("cam4docc_gmo", "Full model"), ("no_cmli", "- CMLI")]


def _eval_json(rdir: Path, config="cam4docc_gmo", *, num_classes=3, iou_c=0.1357, iou_f=0.1189):
    """One eval_<config>.json in the exact shape tools/eval.py writes."""
    rdir.mkdir(parents=True, exist_ok=True)
    present = [0.9, 0.30, 0.20][:num_classes] + [0.1] * max(0, num_classes - 3)
    future = [0.88, 0.25, 0.15][:num_classes] + [0.05] * max(0, num_classes - 3)
    payload = {
        "config": config,
        "checkpoint": f"/scratch/runs/{config}/latest.pth (epoch=12, step=270360)",
        "device": "cuda",
        "iou": {
            "IoU_c": iou_c, "IoU_f": iou_f, "IOU_mean": 0.25,
            "per_class_present": present, "per_class_future": future,
            "per_horizon_IoU": [0.21, 0.18, 0.16, 0.14, 0.12],
        },
        "flow": {"epe": 0.42, "angular_error": 0.31, "magnitude_error": 0.27, "n_valid": 12345},
        "ece": {
            "ece": 0.061, "n_valid": 999,
            "bin_confidence": [0.1, 0.5, 0.9], "bin_accuracy": [0.2, 0.45, 0.8],
            "bin_count": [10.0, 50.0, 40.0],
        },
        "class_names": [f"class_{i}" for i in range(num_classes)],
        "num_batches": 100, "num_samples": 100, "elapsed_s": 60.0,
    }
    (rdir / f"eval_{config}.json").write_text(json.dumps(payload))
    return rdir


def _read(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


class TestSummary:
    def test_headline_numbers_survive_the_round_trip(self, tmp_path) -> None:
        rows = build_summary_rows(_eval_json(tmp_path), CONFIGS)
        assert len(rows) == 1
        row = rows[0]
        assert row["IoU_c"] == 0.1357 and row["IoU_f"] == 0.1189
        assert row["flow_epe_m"] == 0.42 and row["ece"] == 0.061

    def test_gmo_columns_are_filled_for_the_three_class_preset(self, tmp_path) -> None:
        """Class 2 of `cam4docc_gmo` is the unified movable-object class -- the one quantity
        the Cam4DOcc benchmark reports -- so it gets its own column rather than being left
        for the reader to dig out of the per-class file."""
        row = build_summary_rows(_eval_json(tmp_path), CONFIGS)[0]
        assert row["gmo_iou_present"] == 0.20
        assert row["gmo_iou_future"] == 0.15

    def test_gmo_columns_stay_empty_when_index_two_is_not_gmo(self, tmp_path) -> None:
        """The failure this prevents: in a 17-class run, class 2 is `bicycle`. Filling a
        column headed "GMO" with it would put the wrong quantity in a paper table with
        nothing to signal the substitution."""
        row = build_summary_rows(_eval_json(tmp_path, num_classes=17), CONFIGS)[0]
        assert row["num_classes"] == 17
        assert row["gmo_iou_present"] is None and row["gmo_iou_future"] is None

    def test_a_config_that_has_not_been_evaluated_is_absent_not_blank(self, tmp_path) -> None:
        """A blank row reads as a measured zero to anything that plots it; an absent row
        cannot be mistaken for a result."""
        rows = build_summary_rows(_eval_json(tmp_path), CONFIGS)
        assert [r["config"] for r in rows] == ["cam4docc_gmo"]


class TestLongFormatTables:
    def test_per_class_rows_cover_every_class_and_flag_free(self, tmp_path) -> None:
        rows = build_per_class_rows(_eval_json(tmp_path), CONFIGS)
        assert len(rows) == 3
        assert [r["class_index"] for r in rows] == [0, 1, 2]
        assert rows[0]["is_free_class"] is True and rows[2]["is_free_class"] is False

    def test_horizon_seconds_match_the_printed_report(self, tmp_path) -> None:
        """`tools/eval.py` prints bucket k at 0.5*(k+1) seconds, k starting at 1. If the CSV
        used a different formula the same curve would appear at two different x-axes in the
        report and in the paper figure."""
        rows = build_per_horizon_rows(_eval_json(tmp_path), CONFIGS)
        assert [r["horizon_s"] for r in rows] == [1.0, 1.5, 2.0, 2.5, 3.0]
        assert rows[0]["iou_cumulative"] == 0.21

    def test_calibration_gap_is_accuracy_minus_confidence(self, tmp_path) -> None:
        """Sign convention matters: the reliability diagram reads a positive gap as
        underconfidence. Flipping it would invert the paper's calibration claim."""
        rows = build_calibration_rows(_eval_json(tmp_path), CONFIGS)
        assert len(rows) == 3
        assert rows[0]["gap_accuracy_minus_confidence"] == 0.2 - 0.1
        assert rows[2]["gap_accuracy_minus_confidence"] == 0.8 - 0.9
        assert abs(sum(r["voxel_fraction"] for r in rows) - 1.0) < 1e-9


class TestReadsTheShapesTheToolsActuallyWrite:
    def test_robustness_reads_the_table_dict(self, tmp_path) -> None:
        """`tools/run_robustness.py` writes `payload["table"]`, a dict keyed by scenario name.
        `tools/collect_results.py` looks for `payload["scenarios"]`, a list that is never
        written -- so its robustness section silently renders as nothing at all."""
        (tmp_path / "robustness_cam4docc_gmo.json").write_text(json.dumps({
            "config": "cam4docc_gmo",
            "table": {
                "clean": {"lidar_dropout": 0.0, "cam_dropout": 0.0, "IoU_c": 0.20,
                          "IoU_f": 0.18, "IoU_f_degradation_pct": 0.0},
                "severe_dropout": {"lidar_dropout": 0.7, "cam_dropout": 0.7, "IoU_c": 0.12,
                                   "IoU_f": 0.09, "IoU_f_degradation_pct": 50.0},
            },
        }))
        rows = build_robustness_rows(tmp_path)
        assert [r["scenario"] for r in rows] == ["clean", "severe_dropout"]
        assert rows[0]["IoU_f_retained_pct"] == 100.0
        assert rows[1]["IoU_f_retained_pct"] == 50.0

    def test_a_nan_degradation_does_not_become_a_retained_number(self, tmp_path) -> None:
        """`IoU_f_degradation_pct` is NaN when the baseline scenario scored zero. `100 - NaN`
        is NaN, which a CSV reader will happily plot; leaving the cell empty says "not
        measurable" instead."""
        (tmp_path / "robustness_x.json").write_text(json.dumps({
            "config": "x",
            "table": {"clean": {"IoU_c": 0.0, "IoU_f": 0.0,
                                "IoU_f_degradation_pct": float("nan")}},
        }))
        assert build_robustness_rows(tmp_path)[0]["IoU_f_retained_pct"] is None

    def test_latency_reads_the_nested_sub_dicts(self, tmp_path) -> None:
        """`tools/benchmark_latency.py` nests its numbers under `latency`/`memory`/`params`/
        `flops`. The flat keys `tools/collect_results.py` reads (`latency_ms`,
        `peak_memory_gb`, `num_parameters`) do not exist in the file it reads."""
        (tmp_path / "latency_cam4docc_gmo.json").write_text(json.dumps({
            "config": "cam4docc_gmo", "device": "cuda", "hardware": "Tesla V100",
            "batch_size": 1, "T_p": 3, "T_o": 6,
            "latency": {"mean_ms": 812.4, "std_ms": 11.2, "p95_ms": 830.0},
            "memory": {"peak_mb": 10240.0, "method": "torch.cuda.max_memory_allocated"},
            "params": {"total_m": 214.7, "trainable_m": 214.7},
            "flops": {"backend": "fvcore", "gflops": 3120.5},
        }))
        row = build_latency_rows(tmp_path)[0]
        assert row["latency_mean_ms"] == 812.4
        assert row["peak_memory_gb"] == 10.0
        assert row["params_total_m"] == 214.7
        assert row["gflops"] == 3120.5


class TestBaselineComparison:
    def test_drift_numbers_are_converted_to_the_published_scale(self, tmp_path) -> None:
        """The scale trap. DRIFT's GMO IoU of 0.20 belongs in a table of percentages as 20.0,
        beside OCFNet's 27.86. Left as 0.20 the model looks 100x worse than it is, and the
        table still renders without complaint."""
        rows = build_baseline_comparison_rows(_eval_json(tmp_path), CONFIGS)
        drift = [r for r in rows if r["measured_here"]]
        assert len(drift) == 1
        assert drift[0]["IoU_c_pct"] == 20.0
        assert drift[0]["IoU_f_pct"] == 15.0

    def test_published_rows_keep_their_citation_and_are_marked_as_not_re_run(self, tmp_path) -> None:
        rows = build_baseline_comparison_rows(_eval_json(tmp_path), CONFIGS)
        ocfnet = [r for r in rows if r["method"] == "OCFNet"][0]
        assert ocfnet["IoU_c_pct"] == 27.86 and ocfnet["IoU_f_pct"] == 23.89
        assert ocfnet["measured_here"] is False
        assert "2311.17663" in ocfnet["source"]

    def test_the_unverified_dagger_row_says_so(self, tmp_path) -> None:
        """A transcribed row whose table caption was never read must not sit in a comparison
        looking exactly as solid as the rows that were checked."""
        rows = build_baseline_comparison_rows(_eval_json(tmp_path), CONFIGS)
        dagger = [r for r in rows if "dagger" in r["method"]]
        assert dagger and all(r["needs_verification"] for r in dagger)

    def test_no_drift_row_when_the_class_count_is_not_the_gmo_preset(self, tmp_path) -> None:
        """A 17-class run has no column meaning the same thing as the benchmark's GMO IoU.
        Emitting one anyway -- by reusing IOU_mean, say -- would be a comparison between two
        different quantities presented as a like-for-like table."""
        rows = build_baseline_comparison_rows(_eval_json(tmp_path, num_classes=17), CONFIGS)
        assert not [r for r in rows if r["measured_here"]]
        assert [r for r in rows if not r["measured_here"]], "baselines should still be listed"

    def test_percent_conversion_is_exact_at_the_ends(self) -> None:
        assert as_percent(0.0) == 0.0
        assert as_percent(1.0) == 100.0
        assert as_percent(None) is None

    def test_an_unknown_protocol_raises_rather_than_returning_no_baselines(self) -> None:
        """Returning [] would render a comparison table containing only DRIFT, which reads as
        "no baseline exists" rather than "you asked for a protocol that does not exist"."""
        import pytest
        with pytest.raises(ValueError, match="inflated_gmo"):
            for_protocol("gmo")


class TestEndToEnd:
    def test_main_writes_the_expected_files(self, tmp_path) -> None:
        rdir = _eval_json(tmp_path / "results")
        out = tmp_path / "csv"
        assert main(["--results-dir", str(rdir), "--out-dir", str(out)]) == 0
        for name in ["summary.csv", "per_class_iou.csv", "per_horizon_iou.csv",
                     "calibration.csv", "baseline_comparison.csv", "model_complexity.csv",
                     "README_CSV.md"]:
            assert (out / name).exists(), f"{name} was not written"
        assert _read(out / "summary.csv")[0]["config"] == "cam4docc_gmo"

    def test_empty_tables_are_not_written_as_header_only_files(self, tmp_path) -> None:
        """A robustness.csv containing nothing but a header line looks like a study that ran
        and found nothing, rather than one that has not been run."""
        rdir = _eval_json(tmp_path / "results")
        out = tmp_path / "csv"
        main(["--results-dir", str(rdir), "--out-dir", str(out)])
        assert not (out / "robustness.csv").exists()
        assert not (out / "latency.csv").exists()

    def test_a_truncated_json_does_not_take_the_other_configs_with_it(self, tmp_path) -> None:
        """A job killed mid-write leaves half a JSON. One unreadable file must not cost the
        six configs that finished."""
        rdir = _eval_json(tmp_path / "results")
        _eval_json(rdir, config="no_cmli")
        (rdir / "eval_camera_only.json").write_text('{"iou": {"IoU_c":')
        out = tmp_path / "csv"
        assert main(["--results-dir", str(rdir), "--out-dir", str(out)]) == 0
        assert len(_read(out / "summary.csv")) == 2

    def test_a_missing_results_dir_fails_loudly(self, tmp_path) -> None:
        assert main(["--results-dir", str(tmp_path / "nope")]) == 1
