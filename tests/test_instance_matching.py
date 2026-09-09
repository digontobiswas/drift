"""Hungarian matching must survive a non-finite prediction instead of ending the run.

This is the failure that finally showed itself after days of segfaults. Training kept
dying mid-epoch, and with gradient checkpointing on it surfaced as SIGSEGV inside the
autograd engine, which pointed nowhere. With checkpointing off the same fault arrived as
what it actually was:

    File "drift/losses/instance.py", in _linear_sum_assignment_with_fallback
        row, col = linear_sum_assignment(cost_np)
    ValueError: matrix contains invalid numeric entries

Under AMP the instance head runs in fp16, which saturates at 65504. Once the weights grew
enough, an occasional predicted centre or size overflowed, `cdist` spread the Inf across a
whole row of the cost matrix, and scipy refused it. Crash frequency rose with training
because the overflow got likelier as the weights got larger -- which is why it looked at
first like a bad data region, and why moving to a fresh epoch did not help.

Two things are pinned here: the matcher tolerates a non-finite cost matrix, and the cost
matrix is built in fp32 so an fp16 input cannot make it non-finite in the first place.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from drift.data.cam4docc_dataset import BoxSet
from drift.losses.instance import (
    _box_cost_matrix,
    _cls_cost_matrix,
    _linear_sum_assignment_with_fallback,
    instance_loss,
)
from drift.models.instance_path import InstanceState


class TestMatcherToleratesNonFiniteCosts:
    def test_a_nan_entry_no_longer_ends_the_run(self) -> None:
        """The literal crash. Before the fix this raised ValueError and took the job with
        it, discarding every batch since the last checkpoint."""
        cost = torch.tensor([[0.0, 5.0], [float("nan"), 1.0]])
        rows, cols = _linear_sum_assignment_with_fallback(cost)
        assert len(rows) == 2 and len(cols) == 2
        assert sorted(cols) == [0, 1], "every ground-truth box must still be assigned"

    def test_an_infinite_entry_is_treated_as_unmatchable(self) -> None:
        """A broken prediction should lose any match it could otherwise have won: query 0
        would be the cheapest pairing for gt 0 if its Inf were read as a number, so the
        matcher must give that box to query 1 instead."""
        cost = torch.tensor([[float("inf"), 9.0], [1.0, 9.0]])
        rows, cols = _linear_sum_assignment_with_fallback(cost)
        pairing = dict(zip(rows, cols))
        assert pairing[1] == 0, f"the finite query should have taken gt 0, got {pairing}"

    def test_negative_infinity_does_not_become_irresistible(self) -> None:
        """-Inf is just as broken as +Inf, but reads as infinitely attractive. Treating it
        by sign rather than by finiteness would let a garbage prediction capture the
        match it least deserves."""
        cost = torch.tensor([[float("-inf"), 9.0], [1.0, 9.0]])
        rows, cols = _linear_sum_assignment_with_fallback(cost)
        pairing = dict(zip(rows, cols))
        assert pairing[1] == 0, f"the finite query should have taken gt 0, got {pairing}"

    def test_an_entirely_non_finite_matrix_still_returns_a_valid_assignment(self) -> None:
        """Nothing is comparable, so no assignment is better than another -- but the run
        must continue, and the result must still be a legal matching."""
        cost = torch.full((3, 2), float("nan"))
        rows, cols = _linear_sum_assignment_with_fallback(cost)
        assert len(rows) == 2 and sorted(cols) == [0, 1]
        assert len(set(rows)) == len(rows), "a query was matched twice"

    def test_finite_matrices_are_matched_exactly_as_before(self) -> None:
        """The sanitising path must not perturb the ordinary case, which is almost every
        batch: same optimal assignment as plain scipy."""
        torch.manual_seed(0)
        cost = torch.rand(6, 4)
        rows, cols = _linear_sum_assignment_with_fallback(cost)

        from scipy.optimize import linear_sum_assignment

        exp_r, exp_c = linear_sum_assignment(cost.numpy())
        assert rows == exp_r.tolist() and cols == exp_c.tolist()

    def test_empty_inputs_are_still_handled(self) -> None:
        assert _linear_sum_assignment_with_fallback(torch.zeros(0, 3)) == ([], [])
        assert _linear_sum_assignment_with_fallback(torch.zeros(3, 0)) == ([], [])


def _state(Q=3, num_classes=2, poison=None) -> InstanceState:
    """One horizon of predictions; `poison` sets center[0, 0, 0] to that value."""
    center = torch.zeros(1, Q, 3)
    if poison is not None:
        center[0, 0, 0] = poison
    return InstanceState(
        center=center,
        size=torch.ones(1, Q, 3),
        yaw=torch.zeros(1, Q, 1),
        velocity=torch.zeros(1, Q, 3),
        logits=torch.zeros(1, Q, num_classes),
        embed=torch.zeros(1, Q, 4),
        score=torch.full((1, Q, 1), 0.5),
    )


def _gt(n=2) -> BoxSet:
    return BoxSet(
        center=torch.zeros(n, 3),
        size=torch.ones(n, 3),
        yaw=torch.zeros(n, 1),
        velocity=torch.zeros(n, 3),
        label=torch.zeros(n, dtype=torch.long),
        track_id=torch.arange(n, dtype=torch.long),
    )


class TestTheLossItselfSurvives:
    def test_a_non_finite_prediction_does_not_raise(self) -> None:
        """End to end through `instance_loss`, which is where the traceback came from.
        One query's centre is Inf -- what an fp16 overflow in the instance head actually
        produces -- and the run has to carry on."""
        states = [_state(poison=float("inf")), _state()]
        losses = instance_loss(states, [[_gt(), _gt()]])
        assert set(losses) == {
            "loss_instance_cls", "loss_instance_box", "loss_instance_traj", "loss_instance_score"
        }

    def test_a_clean_batch_still_produces_finite_losses(self) -> None:
        """The guard must not quietly change the ordinary path."""
        losses = instance_loss([_state(), _state()], [[_gt(), _gt()]])
        for name, value in losses.items():
            assert torch.isfinite(value), f"{name} is not finite: {value}"


class TestCostMatrixIsBuiltInFp32:
    def test_half_precision_boxes_do_not_overflow_the_cost_matrix(self) -> None:
        """A predicted centre of 5e4 is representable in fp16 but squaring or summing a few
        of them is not. Computed in fp16 the cost comes out non-finite; in fp32 it is just
        a large number, and a large number is something the matcher can rank."""
        big = 5e4
        pred_center = torch.full((2, 3), big, dtype=torch.float16)
        pred_size = torch.full((2, 3), big, dtype=torch.float16)
        pred_yaw = torch.zeros(2, 1, dtype=torch.float16)
        gt_center = torch.zeros(1, 3, dtype=torch.float16)
        gt_size = torch.ones(1, 3, dtype=torch.float16)
        gt_yaw = torch.zeros(1, 1, dtype=torch.float16)

        cost = _box_cost_matrix(pred_center, pred_size, pred_yaw, gt_center, gt_size, gt_yaw)
        assert cost.dtype == torch.float32
        assert torch.isfinite(cost).all(), f"cost went non-finite: {cost}"

    def test_class_cost_is_fp32_and_bounded(self) -> None:
        logits = torch.tensor([[20.0, -20.0], [0.0, 0.0]], dtype=torch.float16)
        cost = _cls_cost_matrix(logits, torch.tensor([0, 1]))
        assert cost.dtype == torch.float32
        assert torch.isfinite(cost).all()
        assert (cost <= 0).all() and (cost >= -1).all(), "a probability cost must stay in [-1, 0]"
