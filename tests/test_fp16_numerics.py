"""fp16 numerics: the root cause of the NaN that stalled every full training run.

Training runs under fp16 autocast on V100s. The symptom was always the same: a checkpoint
that evaluated cleanly (evaluation runs in fp32) produced a non-finite loss on essentially
every training batch, so a run either wrote NaN weights (before the checkpoint guard) or
skipped every batch and made no progress (after it). The tests below pin the mechanisms,
each of which is finite in fp32 and non-finite in fp16 for the *same* values:

* `_score_to_logit` on an fp16 objectness score. In fp16 `1 - 1e-4` rounds to exactly 1.0
  and a sigmoid above 8.32 also rounds to 1.0, so a confident query gives log(1/0) = +Inf
  and BCE-with-logits returns Inf (unmatched) or NaN (matched). This one needs no weight
  blow-up at all -- only a model confident enough to say p > 0.99976 about something --
  which is why it surfaces several epochs into training and then fires on nearly every
  batch.
* `mask * real + (1 - mask) * imagined` in CMLI with every dropout rate at 0: `1 - mask` is
  exactly 0, and `0 * Inf = NaN` replaced the real latent with NaN.
* `x.sum() * 0.0` as a "graph-connected zero": NaN whenever `x` holds one Inf.
* the camera un-projection's intermediate `u * depth`, up to 1598 * 72.4 = 115,695.

CPU autocast uses bf16, which has fp32's range, so these tests reproduce fp16 by feeding
fp16 tensors directly rather than by enabling autocast.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn.functional as F

from drift.data.cam4docc_dataset import BoxSet
from drift.losses.cmli import cmli_consistency_loss
from drift.losses.instance import _score_to_logit, instance_loss
from drift.models.cmli import CrossModalLatentImagination
from drift.models.encoders.camera_encoder import CameraEncoder
from drift.models.encoders.camera_params import CameraParams
from drift.models.instance_path import InstanceSplatter, InstanceState
from drift.utils.activation_monitor import ActivationMonitor
from drift.utils.precision import graph_zero, select

FP16_MAX = 65504.0
REPO = Path(__file__).resolve().parents[1]


class TestHelpers:
    def test_select_ignores_a_non_finite_unselected_branch(self) -> None:
        a = torch.tensor([1.0, 2.0], dtype=torch.half)
        b = torch.tensor([float("inf"), 3.0], dtype=torch.half)
        m = torch.tensor([1.0, 0.0])
        assert torch.isnan(m * a + (1 - m) * b).any(), "precondition: the blend form NaNs"
        assert torch.equal(select(m, a, b), torch.tensor([1.0, 3.0], dtype=torch.half))

    def test_select_routes_gradient_like_the_blend(self) -> None:
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([5.0, 3.0], requires_grad=True)
        select(torch.tensor([1.0, 0.0]), a, b).sum().backward()
        assert a.grad.tolist() == [1.0, 0.0] and b.grad.tolist() == [0.0, 1.0]

    def test_graph_zero_is_zero_even_over_inf_and_nan(self) -> None:
        x = torch.tensor([float("inf"), float("nan"), 1.0], requires_grad=True)
        assert torch.isnan(x.sum() * 0.0), "precondition: the old idiom is NaN"
        z = graph_zero(x)
        assert z.item() == 0.0 and z.requires_grad
        z.backward()
        assert torch.isfinite(x.grad).all()


class TestObjectnessScore:
    """The mechanism that best fits the evidence: finite in fp32, non-finite in fp16, and
    triggered by ordinary training progress rather than by divergence."""

    def test_fp16_sigmoid_saturates_to_exactly_one(self) -> None:
        assert torch.sigmoid(torch.tensor(8.4, dtype=torch.half)).item() == 1.0
        assert torch.tensor(1.0 - 1e-4, dtype=torch.half).item() == 1.0

    def test_saturated_fp16_score_gives_a_finite_logit(self) -> None:
        score = torch.tensor([[1.0], [0.0], [0.5]], dtype=torch.half)
        logit = _score_to_logit(score)
        assert torch.isfinite(logit).all(), logit
        assert logit[0].item() == pytest.approx(math.log(0.9999 / 0.0001), rel=1e-3)

    def test_bce_on_the_recovered_logit_is_finite_for_both_targets(self) -> None:
        logit = _score_to_logit(torch.tensor([[1.0]], dtype=torch.half))
        for target in (0.0, 1.0):
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor([[target]]))
            assert torch.isfinite(loss), f"target {target}: {loss}"

    def test_instance_loss_is_finite_with_a_confident_fp16_query(self) -> None:
        B, Q, C = 1, 4, 2
        def state():
            return InstanceState(
                center=torch.tensor([[[1.0, 2.0, 0.0], [10.0, -3.0, 0.5], [-20.0, 5.0, 0.0], [0.0, 0.0, 0.0]]]).half(),
                size=torch.full((B, Q, 3), 2.0).half(), yaw=torch.zeros(B, Q, 1).half(),
                velocity=torch.zeros(B, Q, 3).half(), logits=torch.zeros(B, Q, C).half(),
                embed=torch.zeros(B, Q, 8).half(),
                score=torch.tensor([[[1.0], [0.2], [0.999], [0.0]]]).half(),  # query 0 saturated
            )
        gt = BoxSet(
            center=torch.tensor([[1.0, 2.0, 0.0]]), size=torch.full((1, 3), 2.0),
            yaw=torch.zeros(1, 1), velocity=torch.zeros(1, 3),
            label=torch.zeros(1, dtype=torch.long), track_id=torch.ones(1, dtype=torch.long),
        )
        losses = instance_loss([state(), state()], [[gt, gt]])
        for name, value in losses.items():
            assert torch.isfinite(value), f"{name} = {value}"


class TestCMLI:
    def _cmli(self) -> CrossModalLatentImagination:
        return CrossModalLatentImagination(channels=4, latent_size=(4, 4, 2), hidden=4)

    def test_real_latent_survives_an_overflowed_generator(self) -> None:
        """All masks 1 (the default config): the output must be the real latent, exactly,
        however broken the never-used imagined branch is."""
        cmli = self._cmli()
        for gen in (cmli.lidar_generator, cmli.cam_generator):
            gen.forward = lambda x: torch.full_like(x[:, :4], float("inf"))
        lidar = torch.randn(1, 2, 4, 4, 4, 2).half()
        cam = torch.randn(1, 2, 4, 4, 4, 2).half()
        ones = torch.ones(1, 2)
        lidar_out, cam_out, aux = cmli(lidar, cam, ones, ones)
        assert torch.equal(lidar_out, lidar) and torch.equal(cam_out, cam)
        assert torch.isfinite(aux["disagreement"]).all()

    def test_a_dropped_frame_still_takes_the_imagined_latent(self) -> None:
        cmli = self._cmli()
        lidar = torch.randn(1, 2, 4, 4, 4, 2)
        cam = torch.randn(1, 2, 4, 4, 4, 2)
        lidar_mask = torch.tensor([[1.0, 0.0]])
        lidar_out, _, aux = cmli(lidar, cam, lidar_mask, torch.ones(1, 2))
        assert torch.equal(lidar_out[:, 0], lidar[:, 0])
        assert torch.equal(lidar_out[:, 1], aux["imagined_lidar"][:, 1])

    def test_unused_cmli_occ_term_is_zero_not_nan(self) -> None:
        """`loss_cmli_occ` is this graph-connected zero on every step of the default
        config, so one Inf voxel in the imagined latent made the total loss NaN."""
        imagined = torch.zeros(1, 2, 4, 4, 4, 2)
        imagined[0, 0, 0, 0, 0, 0] = float("inf")
        aux = {"imagined_lidar": imagined, "imagined_cam": torch.zeros_like(imagined)}
        masks = {"lidar_mask": torch.ones(1, 2), "cam_mask": torch.ones(1, 2)}
        out = cmli_consistency_loss(aux, masks)
        assert out["loss_cmli_occ"].item() == 0.0

    def test_consistency_l2_does_not_overflow_on_large_fp16_latents(self) -> None:
        real = torch.full((1, 1, 4, 2, 2, 2), 60000.0).half()
        imagined = torch.full_like(real, -60000.0)
        assert torch.isinf(imagined - real).all(), "precondition: the fp16 difference overflows"
        aux = {"imagined_lidar": imagined, "imagined_cam": imagined,
               "real_lidar": real, "real_cam": real}
        out = cmli_consistency_loss(aux, {"lidar_mask": torch.ones(1, 1), "cam_mask": torch.ones(1, 1)})
        assert torch.isfinite(out["loss_cmli_lidar"])


class TestCameraLift:
    def _encoder(self) -> CameraEncoder:
        return CameraEncoder(backbone="resnet18", out_channels=8, latent_size=(128, 128, 10),
                             depth_bins=112, pretrained=False)

    def _nuscenes_params(self, dtype=torch.float32) -> CameraParams:
        """One front camera at nuScenes scale: 1600 px wide, cropped/resized to 704."""
        intr = torch.tensor([[1266.4, 0.0, 816.3], [0.0, 1266.4, 491.5], [0.0, 0.0, 1.0]])
        rot = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # cam -> ego
        scale = 704.0 / 1600.0
        post_rot = torch.diag(torch.tensor([scale, scale, 1.0]))
        post_trans = torch.tensor([0.0, -140.0, 0.0])
        v = lambda t: t.to(dtype).view(1, 1, 1, *t.shape)
        return CameraParams(rots=v(rot), trans=v(torch.tensor([1.7, 0.0, 1.5])), intrins=v(intr),
                            post_rots=v(post_rot), post_trans=v(post_trans))

    def test_the_fp16_intermediate_really_overflows(self) -> None:
        enc = self._encoder()
        frustum = enc._frustum(32, 88, 256, 704, torch.device("cpu"))
        u_orig = frustum[..., 0].max() / (704.0 / 1600.0)
        assert u_orig * enc.d_max > FP16_MAX, "precondition for this whole test class"

    def test_lift_is_fp32_and_finite_even_from_fp16_inputs(self) -> None:
        enc = self._encoder()
        frustum = enc._frustum(32, 88, 256, 704, torch.device("cpu")).half()
        pts = enc._lift_points(frustum, self._nuscenes_params(torch.half))
        assert pts.dtype == torch.float32
        assert torch.isfinite(pts).all()

    def test_lift_matches_a_float64_reference(self) -> None:
        enc = self._encoder()
        frustum = enc._frustum(32, 88, 256, 704, torch.device("cpu"))
        cp = self._nuscenes_params()
        got = enc._lift_points(frustum, cp).double()
        f = frustum.double()
        pr_inv = torch.inverse(cp.post_rots[0, 0, 0].double())
        p = (pr_inv @ (f - cp.post_trans[0, 0, 0].double()).unsqueeze(-1)).squeeze(-1)
        p = torch.cat([p[..., :2] * p[..., 2:3], p[..., 2:3]], dim=-1)
        comb = cp.rots[0, 0, 0].double() @ torch.inverse(cp.intrins[0, 0, 0].double())
        ref = (comb @ p.unsqueeze(-1)).squeeze(-1) + cp.trans[0, 0, 0].double()
        assert torch.allclose(got[0, 0, 0], ref, atol=1e-3)


class TestSplatter:
    def test_far_out_fp16_queries_do_not_poison_the_grid(self) -> None:
        sp = InstanceSplatter(embed_dims=8, out_channels=4, latent_size=(16, 16, 4),
                              point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        B, Q = 1, 3
        st = InstanceState(
            center=torch.tensor([[[0.0, 0.0, 0.0], [60000.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]]).half(),
            size=torch.full((B, Q, 3), 2.0).half(), yaw=torch.zeros(B, Q, 1).half(),
            velocity=torch.zeros(B, Q, 3).half(), logits=torch.zeros(B, Q, 2).half(),
            embed=torch.randn(B, Q, 8).half(), score=torch.full((B, Q, 1), 0.9).half(),
        )
        feats, occ = sp([st])
        assert feats.dtype == torch.half
        assert torch.isfinite(feats).all() and torch.isfinite(occ).all()
        assert occ.sum() > 0, "the in-range query must still be splatted"


class TestActivationMonitor:
    def test_names_the_module_where_the_overflow_starts(self) -> None:
        class Boom(torch.nn.Module):
            def forward(self, x):
                return x * 1e5
        model = torch.nn.Sequential(torch.nn.Identity(), Boom(), torch.nn.ReLU())
        mon = ActivationMonitor(model)
        mon.start_batch()
        model(torch.ones(2, 2).half())
        rec = mon.first_non_finite()
        assert rec is not None and rec.name == "1"
        assert "inputs were finite" in mon.describe_first_non_finite()
        mon.remove()

    def test_reports_headroom_without_any_overflow(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        mon = ActivationMonitor(model)
        mon.start_batch()
        model(torch.ones(1, 2))
        assert mon.first_non_finite() is None
        assert "[headroom]" in mon.describe_headroom()
        mon.remove()


class TestDiagnosticRun:
    def test_max_batches_and_detect_nonfinite_run_end_to_end(self, tmp_path) -> None:
        result = subprocess.run(
            [sys.executable, "tools/train.py", "--config", "tiny", "--dataset", "synthetic",
             "--device", "cpu", "--no-amp", "--epochs", "1", "--max-batches", "2",
             "--batch-size", "1", "--num-workers", "0", "--log-interval", "1",
             "--detect-nonfinite", "--ckpt-dir", str(tmp_path)],
            cwd=str(REPO), capture_output=True, text=True, timeout=900,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "--detect-nonfinite: watching" in result.stdout
        assert "[headroom] largest activations seen" in result.stdout
        assert result.stdout.count("[train] epoch=0 it=") == 2


class TestNormFedConv:
    """The fix for the overflow that actually stopped both full runs.

    `cmli.lidar_generator.net.3` -- a conv feeding a GroupNorm -- reached 99.2% of fp16's
    limit by epoch 5 and then overflowed on every batch (diagnostic job 1225732). A conv
    before a GroupNorm is scale-invariant, so nothing stops its weights growing. The fix
    must (1) make overflow impossible, (2) change neither the function nor the gradients,
    (3) be an exact no-op at initialisation, and (4) leave checkpoints loadable.
    """

    @staticmethod
    def _pair(scale: float, conv_cls=torch.nn.Conv3d):
        torch.manual_seed(0)
        conv = conv_cls(8, 8, 3, padding=1)
        with torch.no_grad():
            conv.weight.mul_(scale)
            conv.bias.mul_(scale)
        return conv, torch.nn.GroupNorm(2, 8)

    def test_plain_fp16_conv_overflows_where_the_fixed_one_does_not(self) -> None:
        from drift.utils.precision import NormFedConv3d
        conv, gn = self._pair(scale=2000.0)
        x = torch.rand(1, 8, 6, 6, 4) * 140  # the input range seen on the cluster
        plain = conv.half()(x.half())
        assert torch.isinf(plain).any(), "precondition: the plain fp16 conv must overflow"
        conv.__class__ = NormFedConv3d
        fixed = conv(x.half())
        assert torch.isfinite(fixed).all()
        assert torch.isfinite(gn(fixed.float())).all()

    def test_output_is_provably_bounded(self) -> None:
        from drift.utils.precision import NORM_FED_TARGET, NormFedConv3d
        conv, _ = self._pair(scale=1e4)
        conv.__class__ = NormFedConv3d
        for _ in range(5):
            x = torch.randn(1, 8, 5, 5, 3) * 1e3
            assert conv(x).abs().max().item() <= NORM_FED_TARGET * (1 + 1e-5)

    def test_function_and_gradients_match_the_plain_conv(self) -> None:
        from drift.utils.precision import NormFedConv3d
        x = torch.randn(1, 8, 6, 6, 4) * 50
        results = []
        for cls in (torch.nn.Conv3d, NormFedConv3d):
            conv, gn = self._pair(scale=300.0)
            conv.__class__ = cls
            out = gn(conv(x))
            (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().backward()
            results.append((out.detach(), conv.weight.grad.clone(), conv.bias.grad.clone()))
        (o1, gw1, gb1), (o2, gw2, gb2) = results
        assert torch.allclose(o1, o2, rtol=1e-4, atol=1e-5)
        assert torch.allclose(gw1, gw2, rtol=1e-3, atol=1e-7)
        assert torch.allclose(gb1, gb2, rtol=1e-3, atol=1e-7)

    def test_exact_no_op_at_default_initialisation(self) -> None:
        """A fresh run must start exactly where it always did."""
        from drift.utils.precision import NormFedConv3d, _norm_fed_scale
        torch.manual_seed(0)
        plain = torch.nn.Conv3d(64, 64, 3, padding=1)
        fixed = torch.nn.Conv3d(64, 64, 3, padding=1)
        fixed.load_state_dict(plain.state_dict())
        fixed.__class__ = NormFedConv3d
        x = torch.randn(1, 64, 6, 6, 4)  # roughly what a GroupNorm emits at init
        assert _norm_fed_scale(fixed, x).item() == 1.0
        assert torch.equal(plain(x), fixed(x))

    def test_only_conv_directly_before_groupnorm_is_converted(self) -> None:
        from drift.utils.precision import NormFedConv2d, protect_norm_fed_convs
        nn = torch.nn
        model = nn.ModuleDict({
            "a": nn.Sequential(nn.Conv2d(3, 8, 3), nn.GroupNorm(2, 8), nn.ReLU()),
            "b": nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.GroupNorm(2, 8)),
            "c": nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)),
            "d": nn.Sequential(nn.Conv2d(3, 8, 3), nn.Conv2d(8, 8, 3), nn.GroupNorm(2, 8)),
        })
        assert sorted(protect_norm_fed_convs(model)) == ["a.0", "d.1"]
        assert isinstance(model["a"][0], NormFedConv2d)
        assert type(model["b"][0]) is nn.Conv2d
        assert type(model["c"][0]) is nn.Conv2d, "BatchNorm in eval mode is not scale-invariant"
        assert type(model["d"][0]) is nn.Conv2d

    def test_real_model_converts_the_layers_that_overflowed(self) -> None:
        from configs import get_config
        from drift.models.drift import DRIFT
        from drift.utils.precision import NormFedConv3d
        cfg = get_config("tiny")
        model = DRIFT(cfg.model, cfg.loss)
        names = set(model.norm_fed_convs)
        assert {"cmli.lidar_generator.net.3", "cmli.cam_generator.net.3"} <= names
        e4a_convs = [n for n, m in model.named_modules()
                     if isinstance(m, NormFedConv3d) and ".e4a." in n and n.endswith(".conv")]
        assert e4a_convs, "E4A's _ConvBnAct convs must be converted too"

    def test_checkpoint_keys_are_unchanged(self) -> None:
        """Conversion swaps the class, not the parameters: old checkpoints load as-is."""
        from configs import get_config
        from drift.models.drift import DRIFT
        from drift.utils.precision import protect_norm_fed_convs
        cfg = get_config("tiny")
        model = DRIFT(cfg.model, cfg.loss)
        keys = list(model.state_dict().keys())
        protect_norm_fed_convs(model)  # idempotent: a second pass finds nothing new to change
        assert list(model.state_dict().keys()) == keys
        fresh = DRIFT(cfg.model, cfg.loss)
        fresh.load_state_dict(model.state_dict(), strict=True)
