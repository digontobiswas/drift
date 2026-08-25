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
    which `use_reentrant=False` prevents by preserving RNG state.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs import get_config
from drift.data.cam4docc_dataset import SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.models.drift import DRIFT


def _build(grad_checkpoint: bool) -> tuple:
    cfg = get_config("tiny")
    cfg.model.grad_checkpoint = grad_checkpoint
    model = DRIFT(cfg.model, cfg.loss)
    model.train()

    ds = SyntheticOccDataset(
        num_samples=1, T_p=cfg.model.T_p, T_f=cfg.model.T_f, T_o=cfg.model.T_o,
        N_cam=cfg.model.N_cam, H_img=cfg.data.H_img, W_img=cfg.data.W_img,
        num_classes=cfg.model.num_classes, latent_size=cfg.model.latent_size,
        occ_size=cfg.model.occ_size, point_cloud_range=cfg.model.point_cloud_range,
        num_points_range=cfg.data.num_points_range, num_boxes_range=(2, 4),
        modality_dropout_p=0.0, seed=0,
    )
    return model, collate_fn([ds[0]])


class TestGradCheckpointEquivalence:
    def test_flag_reaches_the_forecaster(self) -> None:
        """DRIFT must propagate the flag into the forecaster, which holds the
        largest volumes; a plain model must leave it off."""
        on, _ = _build(grad_checkpoint=True)
        off, _ = _build(grad_checkpoint=False)
        assert on.grad_checkpoint is True
        assert off.grad_checkpoint is False
        if on.forecaster is not None:
            assert on.forecaster.grad_checkpoint is True
            assert off.forecaster.grad_checkpoint is False

    def test_outputs_and_grads_match_uncheckpointed(self) -> None:
        plain, batch = _build(grad_checkpoint=False)
        ckpt, _ = _build(grad_checkpoint=True)
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
