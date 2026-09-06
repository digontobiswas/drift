"""Gradient checkpointing must be a pure memory/compute trade -- never a change in results.

`ModelConfig.grad_checkpoint` exists solely so the model fits a 16 GB V100 (PARAM
Shakti's only GPU). It re-runs each wrapped submodule's forward during backward
instead of keeping activations alive. That is only an acceptable thing to enable
for a paper's ablation grid if it provably does not perturb the numbers, so this
module asserts bit-comparable outputs AND bit-comparable gradients between a
checkpointed and a non-checkpointed model sharing identical weights.

The two hazards this guards against:
  * a norm layer with running statistics being updated twice by the second forward
    (the model uses GroupNorm everywhere, which has none -- this test would catch a
    regression that introduced BatchNorm);
  * RNG-dependent layers (`ModalityDropout`) drawing a different mask on recompute,
    which both checkpoint implementations prevent by preserving RNG state. Dropout is
    left ON in the fixture below precisely so this is exercised rather than asserted.

Everything here runs against BOTH implementations, because `grad_checkpoint_reentrant`
lets the cluster switch between them: every SIGSEGV on PARAM Shakti landed inside the
autograd engine, on one GPU and on two, and the reentrant path was worth trying as the
older and more heavily exercised of the two. Reentrant mode has a failure mode of its
own, though -- it needs an input of each checkpointed block to require grad, or
gradients stop at the boundary and the parameters upstream silently stop training,
with the loss still falling. Comparing both against an uncheckpointed forward is what
makes that switch safe to make.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from configs import get_config
from drift.data.cam4docc_dataset import SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.models.drift import DRIFT

BOTH_IMPLEMENTATIONS = pytest.mark.parametrize(
    "reentrant", [True, False], ids=["reentrant", "non_reentrant"]
)


def _build(grad_checkpoint: bool, reentrant: bool = True) -> tuple:
    cfg = get_config("tiny")
    cfg.model.grad_checkpoint = grad_checkpoint
    cfg.model.grad_checkpoint_reentrant = reentrant
    model = DRIFT(cfg.model, cfg.loss)
    model.train()

    ds = SyntheticOccDataset(
        num_samples=1, T_p=cfg.model.T_p, T_f=cfg.model.T_f, T_o=cfg.model.T_o,
        N_cam=cfg.model.N_cam, H_img=cfg.data.H_img, W_img=cfg.data.W_img,
        num_classes=cfg.model.num_classes, latent_size=cfg.model.latent_size,
        occ_size=cfg.model.occ_size, point_cloud_range=cfg.model.point_cloud_range,
        num_points_range=cfg.data.num_points_range, num_boxes_range=(2, 4),
        # Non-zero on purpose: with dropout off, an implementation that failed to
        # restore RNG state before recomputing would still pass every assertion here.
        modality_dropout_p=0.5, seed=0,
    )
    return model, collate_fn([ds[0]])


class TestGradCheckpointEquivalence:
    def test_flag_reaches_every_submodule_that_declares_it(self) -> None:
        """DRIFT propagates the flag to every submodule with its own `grad_checkpoint`.

        A half-applied flag is the failure mode this guards: the model still trains,
        just with a silently larger memory peak, which presents as "checkpointing
        didn't help" rather than as an error. Both the forecaster (largest volumes)
        and every EfficientAggregation4D (whose per-stage checkpointing is what keeps
        the *backward* recompute peak down) must be switched on.
        """
        on, _ = _build(grad_checkpoint=True)
        off, _ = _build(grad_checkpoint=False)
        assert on.grad_checkpoint is True
        assert off.grad_checkpoint is False

        on_subs = [m for m in on.modules() if m is not on and hasattr(m, "grad_checkpoint")]
        off_subs = [m for m in off.modules() if m is not off and hasattr(m, "grad_checkpoint")]
        assert on_subs, "no submodule declares grad_checkpoint -- propagation is untested"
        assert all(m.grad_checkpoint for m in on_subs), [
            type(m).__name__ for m in on_subs if not m.grad_checkpoint
        ]
        assert not any(m.grad_checkpoint for m in off_subs)

        from drift.models.e4a import EfficientAggregation4D

        e4as = [m for m in on.modules() if isinstance(m, EfficientAggregation4D)]
        assert e4as, "expected at least one EfficientAggregation4D in the model"
        assert all(m.grad_checkpoint for m in e4as)

    @BOTH_IMPLEMENTATIONS
    def test_reentrant_choice_reaches_every_submodule(self, reentrant: bool) -> None:
        """The implementation choice has to propagate as far as the flag itself.

        If it reached only the top-level `_ckpt`, the E4A stages -- the ones this whole
        mechanism exists for, and the ones running when the segfaults happened -- would
        keep using the other implementation while the equivalence test below passed.
        """
        model, _ = _build(grad_checkpoint=True, reentrant=reentrant)
        subs = [m for m in model.modules() if m is not model and hasattr(m, "grad_checkpoint")]
        assert subs, "no submodule declares grad_checkpoint -- propagation is untested"
        for m in subs:
            assert m.grad_checkpoint_reentrant is reentrant, (
                f"{type(m).__name__} kept reentrant={m.grad_checkpoint_reentrant}, "
                f"not the configured {reentrant}"
            )

    def test_reentrant_mode_would_silently_freeze_the_encoders(self) -> None:
        """Pin WHY `grad_checkpoint_reentrant` ships False, so nobody flips it back.

        Reentrant checkpointing propagates gradient only when an input to the
        checkpointed block requires grad. Here the outer blocks are entered with raw
        batch tensors that do not, so everything upstream of the first checkpoint --
        the whole camera backbone and the LiDAR encoder -- receives no gradient and
        stops training, while the loss keeps falling on the parameters that still do.
        There is no error and no log line; the only symptom is a model that never
        learns to see.

        Asserted rather than merely commented, so that if a future PyTorch makes
        reentrant mode safe here, this fails and the choice gets revisited on purpose.
        """
        plain, batch = _build(grad_checkpoint=False)
        reentrant, _ = _build(grad_checkpoint=True, reentrant=True)
        reentrant.load_state_dict(copy.deepcopy(plain.state_dict()))

        for model in (plain, reentrant):
            torch.manual_seed(1234)
            model.zero_grad(set_to_none=True)
            sum(model.loss(model(batch), batch).values()).backward()

        def trained(model) -> set:
            return {n for n, p in model.named_parameters() if p.grad is not None}

        lost = trained(plain) - trained(reentrant)
        assert lost, (
            "reentrant checkpointing no longer drops gradients -- the constraint that "
            "forced grad_checkpoint_reentrant=False may have been lifted; re-run the "
            "equivalence test with reentrant=True and reconsider the default."
        )
        assert any(n.startswith("camera_encoder.") for n in lost), (
            f"expected the camera encoder among the frozen parameters, got {sorted(lost)[:5]}"
        )

    def test_outputs_and_grads_match_uncheckpointed(self) -> None:
        reentrant = False  # the only implementation this model can use; see above
        plain, batch = _build(grad_checkpoint=False)
        ckpt, _ = _build(grad_checkpoint=True, reentrant=reentrant)
        # Identical weights: copy rather than re-seed, so this compares only the
        # checkpointing behaviour and not two independent inits.
        ckpt.load_state_dict(copy.deepcopy(plain.state_dict()))

        def run(model):
            torch.manual_seed(1234)  # same RNG draw for ModalityDropout in both runs
            out = model(batch)
            # `loss()` returns the individual `loss_*` terms, not a pre-summed total;
            # sum them so a single backward exercises every branch at once.
            losses = model.loss(out, batch)
            total = sum(losses.values())
            model.zero_grad(set_to_none=True)
            total.backward()
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            return out["occ_logits"].detach().clone(), total.detach().clone(), grads

        plain_logits, plain_loss, plain_grads = run(plain)
        ckpt_logits, ckpt_loss, ckpt_grads = run(ckpt)

        torch.testing.assert_close(ckpt_logits, plain_logits, rtol=0, atol=0)
        torch.testing.assert_close(ckpt_loss, plain_loss, rtol=0, atol=0)

        assert set(ckpt_grads) == set(plain_grads), (
            "checkpointing changed which parameters receive gradient: "
            f"only-plain={sorted(set(plain_grads) - set(ckpt_grads))[:5]} "
            f"only-ckpt={sorted(set(ckpt_grads) - set(plain_grads))[:5]}"
        )
        for name in plain_grads:
            # Recomputation reorders floating-point accumulation, so allow a tight
            # tolerance here rather than the exact match asserted for the forward.
            torch.testing.assert_close(
                ckpt_grads[name], plain_grads[name], rtol=1e-4, atol=1e-6,
                msg=lambda m, n=name: f"gradient mismatch for {n}:\n{m}",
            )
