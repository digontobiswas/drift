"""Full DRIFT forward + loss() + backward() test. See docs/DESIGN_SPEC.md §8.

Asserts every registered parameter that requires grad receives a non-None gradient --
this is the test that catches a silently dead branch, most importantly the ★NOVEL
instance-query dynamic path: `InstanceQueryExtractor` -> `MotionForecaster` ->
`InstanceSplatter` must receive gradient purely from the dense occupancy/flow
reconstruction loss (via the differentiable soft-splat), not only from the (GT-count-
dependent) auxiliary instance-detection loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs import get_config
from drift.data.cam4docc_dataset import SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.models.drift import DRIFT


def _build_tiny_batch_and_model(num_boxes_range=(2, 4)):
    cfg = get_config("tiny")
    # Force CMLI's stochastic full-drop off so the imagination generators' gradient path
    # is deterministic (they still receive gradient via the consistency loss regardless --
    # see drift.models.drift module docstring seam #1/#3 -- but a *dropped* frame is masked
    # out of that loss's supervision, so forcing drop_rate=0 makes every frame supervised
    # and removes randomness from this specific "no dead parameters" assertion).
    cfg.model.cmli.lidar_drop_rate = 0.0
    cfg.model.cmli.cam_drop_rate = 0.0

    model = DRIFT(cfg.model, cfg.loss)
    model.train()

    ds = SyntheticOccDataset(
        num_samples=1, T_p=cfg.model.T_p, T_f=cfg.model.T_f, T_o=cfg.model.T_o,
        N_cam=cfg.model.N_cam, H_img=cfg.data.H_img, W_img=cfg.data.W_img,
        num_classes=cfg.model.num_classes, latent_size=cfg.model.latent_size,
        occ_size=cfg.model.occ_size, point_cloud_range=cfg.model.point_cloud_range,
        num_points_range=cfg.data.num_points_range,
        # >=2 dynamic-object tracks guarantees `flow_loss` / `instance_loss`'s box+traj
        # terms have real supervision this run, in addition to the dense-loss path that
        # already guarantees gradient reaches every instance-path parameter regardless.
        num_boxes_range=num_boxes_range,
        modality_dropout_p=0.0, seed=0,
    )
    batch = collate_fn([ds[0]])
    return model, batch


class TestForwardBackward:
    def test_full_gradient_flow_no_dead_parameters(self) -> None:
        torch.manual_seed(0)
        model, batch = _build_tiny_batch_and_model()

        outputs = model(batch)
        losses = model.loss(outputs, batch)

        assert len(losses) > 0
        assert all(k.startswith("loss_") for k in losses), losses.keys()
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

        total = sum(losses.values())
        assert total.requires_grad

        model.zero_grad()
        total.backward()

        missing = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing, (
            f"{len(missing)} parameter(s) received NO gradient (dead branch): {missing}"
        )

    def test_instance_path_receives_gradient_through_splatter(self) -> None:
        """Targeted check for the paper's core contribution (spec §8): every stage of the
        instance-query path -- extractor, per-step motion heads, and the splatter's own
        feature projection -- must receive a *non-zero* gradient, sourced through the
        differentiable soft-splat into the dense occupancy/flow loss.
        """
        torch.manual_seed(0)
        model, batch = _build_tiny_batch_and_model()

        outputs = model(batch)
        losses = model.loss(outputs, batch)
        total = sum(losses.values())
        model.zero_grad()
        total.backward()

        params = dict(model.named_parameters())
        instance_param_names = [
            "forecaster.instance_extractor.center_head.weight",
            "forecaster.instance_extractor.size_head.weight",
            "forecaster.instance_extractor.yaw_head.weight",
            "forecaster.instance_extractor.score_head.weight",
            "forecaster.instance_extractor.query_embed",
            "forecaster.motion_forecaster.delta_center.weight",
            "forecaster.motion_forecaster.velocity_head.weight",
            "forecaster.motion_forecaster.step_embed",
            "forecaster.instance_splatter.feat_proj.weight",
        ]
        for name in instance_param_names:
            assert name in params, f"expected parameter '{name}' not found in model"
            grad = params[name].grad
            assert grad is not None, f"{name} received no gradient at all"
            assert torch.any(grad != 0), f"{name}'s gradient is identically zero"

    def test_no_gt_boxes_still_gives_instance_path_gradient(self) -> None:
        """Even with zero ground-truth dynamic objects in the batch (so the auxiliary
        `instance_loss` box/trajectory terms are inactive), the instance path must still
        receive gradient through the dense occupancy/flow reconstruction loss alone --
        this is the specific "dead branch" failure mode the splatter design is meant to
        avoid (see `drift/models/instance_path.py` module docstring).
        """
        torch.manual_seed(0)
        model, batch = _build_tiny_batch_and_model(num_boxes_range=(0, 0))

        outputs = model(batch)
        losses = model.loss(outputs, batch)
        total = sum(losses.values())
        model.zero_grad()
        total.backward()

        params = dict(model.named_parameters())
        grad = params["forecaster.instance_splatter.feat_proj.weight"].grad
        assert grad is not None
        assert torch.any(grad != 0)
        grad2 = params["forecaster.instance_extractor.center_head.weight"].grad
        assert grad2 is not None
        assert torch.any(grad2 != 0)
