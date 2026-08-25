"""Metrics correctness tests. See docs/DESIGN_SPEC.md §8.

`fast_hist`/pooled-IoU against a hand-computed confusion matrix; flow EPE masking.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from drift.metrics.calibration import expected_calibration_error
from drift.metrics.flow_epe import flow_epe
from drift.metrics.iou import OccupancyIoUMetric, cm_to_ious, fast_hist


class TestFastHist:
    def test_matches_hand_computed_confusion_matrix(self) -> None:
        # 3 classes. GT: [0,0,1,1,2,2]; Pred: [0,1,1,2,2,2].
        gt = torch.tensor([0, 0, 1, 1, 2, 2])
        pred = torch.tensor([0, 1, 1, 2, 2, 2])
        hist = fast_hist(pred, gt, num_cls=3)
        # hand-built: hist[gt,pred] += 1
        expected = torch.zeros(3, 3, dtype=torch.int64)
        for g, p in zip(gt.tolist(), pred.tolist()):
            expected[g, p] += 1
        assert torch.equal(hist, expected)
        assert hist.sum().item() == 6

    def test_ignores_out_of_range_labels(self) -> None:
        gt = torch.tensor([0, 255, 1])
        pred = torch.tensor([0, 1, 1])
        hist = fast_hist(pred, gt, num_cls=2)
        assert hist.sum().item() == 2  # the 255 entry is dropped
        assert hist[0, 0].item() == 1
        assert hist[1, 1].item() == 1


class TestCmToIous:
    def test_perfect_prediction_gives_iou_one(self) -> None:
        hist = torch.tensor([[5, 0], [0, 7]])
        ious = cm_to_ious(hist)
        assert torch.allclose(ious, torch.tensor([1.0, 1.0]), atol=1e-4)

    def test_hand_computed_iou(self) -> None:
        # class 0: TP=3, FP=1 (col sum row1->col0), FN=2 (row0 mispredicted as col1)
        # hist[gt,pred]:
        #        pred0 pred1
        # gt0      3     2
        # gt1      1     4
        hist = torch.tensor([[3, 2], [1, 4]])
        ious = cm_to_ious(hist)
        # IoU_0 = 3 / (3+2 + 3+1 - 3) = 3/6 = 0.5
        # IoU_1 = 4 / (1+4 + 2+4 - 4) = 4/7
        assert abs(ious[0].item() - 0.5) < 1e-4
        assert abs(ious[1].item() - 4.0 / 7.0) < 1e-4


class TestOccupancyIoUMetric:
    def test_pooled_over_multiple_updates_matches_hand_computation(self) -> None:
        num_classes = 2
        # latent logits (B=1,T_o=2,C=2,x=1,y=1,z=1); upsample_size small (2,2,2) for speed.
        metric = OccupancyIoUMetric(num_classes=num_classes, num_future=2, upsample_size=(2, 2, 2), present_index=0)

        # Frame 0 (present): make logits strongly prefer class 1 everywhere; GT all class 1 ->
        # a perfect match, contributes to hist_present only.
        logits_t0 = torch.zeros(1, 1, num_classes, 1, 1, 1)
        logits_t0[:, :, 1] = 10.0
        gt_t0 = torch.ones(1, 1, 2, 2, 2, dtype=torch.long)

        # Frame 1 (future): logits strongly prefer class 0 everywhere; GT all class 1 ->
        # total mismatch, contributes to hist_future_total and the (only) per-horizon bucket.
        logits_t1 = torch.zeros(1, 1, num_classes, 1, 1, 1)
        logits_t1[:, :, 0] = 10.0
        gt_t1 = torch.ones(1, 1, 2, 2, 2, dtype=torch.long)

        logits = torch.cat([logits_t0, logits_t1], dim=1)
        gt = torch.cat([gt_t0, gt_t1], dim=1)
        metric.update(logits, gt)

        result = metric.compute()
        # Present frame: perfect class-1 prediction -> IoU_c (mean over classes[1:]) == 1.0
        assert abs(result["IoU_c"] - 1.0) < 1e-4
        # Future frame: class 1 never predicted -> IoU for class 1 == 0
        assert abs(result["IoU_f"] - 0.0) < 1e-4
        assert len(result["per_horizon_IoU"]) == 1
        assert abs(result["per_horizon_IoU"][0] - 0.0) < 1e-4

    def test_reset_clears_accumulators(self) -> None:
        metric = OccupancyIoUMetric(num_classes=2, num_future=1, upsample_size=(2, 2, 2))
        logits = torch.zeros(1, 1, 2, 1, 1, 1)
        gt = torch.zeros(1, 1, 2, 2, 2, dtype=torch.long)
        metric.update(logits, gt)
        metric.reset()
        assert metric._hist_present.sum().item() == 0
        assert metric._hist_future_total.sum().item() == 0


class TestFlowEPE:
    def test_masking_excludes_ignored_voxels(self) -> None:
        # 2 voxels: voxel 0 valid with a known error, voxel 1 fully ignored (255).
        pred = torch.zeros(1, 3, 1, 1, 2)
        target = torch.zeros(1, 3, 1, 1, 2)
        # voxel 0: pred=(1,0,0), target=(0,0,0) -> L2 error = 1.0 voxel = 0.8 m (default voxel_size)
        pred[0, 0, 0, 0, 0] = 1.0
        target[0, 0, 0, 0, 0] = 0.0
        # voxel 1: ignored
        target[0, :, 0, 0, 1] = 255.0

        out = flow_epe(pred, target, voxel_size=0.8)
        assert int(out["n_valid"].item()) == 1
        assert abs(out["epe"].item() - 0.8) < 1e-4

    def test_all_ignored_returns_zero_not_nan(self) -> None:
        pred = torch.randn(1, 3, 1, 1, 1)
        target = torch.full((1, 3, 1, 1, 1), 255.0)
        out = flow_epe(pred, target)
        assert int(out["n_valid"].item()) == 0
        assert torch.isfinite(out["epe"])
        assert out["epe"].item() == 0.0


class TestCalibration:
    def test_perfectly_calibrated_gives_zero_ece(self) -> None:
        # Confidence 1.0 predictions that are always correct -> ECE 0.
        probs = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        labels = torch.tensor([1, 1, 0])
        out = expected_calibration_error(probs, labels, num_bins=5)
        assert out["ece"].item() < 1e-6

    def test_ignore_index_excluded(self) -> None:
        probs = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        labels = torch.tensor([1, 255])
        out = expected_calibration_error(probs, labels, num_bins=5)
        assert out["bin_count"].sum().item() == 1
