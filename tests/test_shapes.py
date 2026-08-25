"""Shape tests: every module's output shape matches ``docs/DESIGN_SPEC.md``, at the tiny config.

See spec §7 ("Smoke-testability is mandatory") and §8.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs import get_config
from configs.base import ModelConfig
from drift.data.cam4docc_dataset import SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.models.cmli import CrossModalLatentImagination, ModalityDropout
from drift.models.condition import ConditionalForecaster
from drift.models.drift import DRIFT
from drift.models.e4a import EfficientAggregation4D
from drift.models.encoders.camera_encoder import CameraEncoder
from drift.models.encoders.cross_modal_fusion import CrossModalFusion
from drift.models.encoders.lidar_encoder import LidarEncoder
from drift.models.encoders.voxel_query_generator import CoarseVoxelQueryGenerator
from drift.models.instance_path import InstanceQueryExtractor, InstanceSplatter, MotionForecaster
from drift.models.observer import Observer
from drift.models.predictor import FlowHead, OccupancyHead, UncertaintyHead
from drift.models.static_path import StaticForecastPath
from drift.models.taf import TriplingAttentionFusion


def _tiny_cfg() -> ModelConfig:
    return get_config("tiny").model


def _make_batch(cfg: ModelConfig, batch_size: int = 1):
    ds = SyntheticOccDataset(
        num_samples=batch_size, T_p=cfg.T_p, T_f=cfg.T_f, T_o=cfg.T_o, N_cam=cfg.N_cam,
        H_img=32, W_img=48, num_classes=cfg.num_classes, latent_size=cfg.latent_size,
        occ_size=cfg.occ_size, point_cloud_range=cfg.point_cloud_range,
        num_points_range=(50, 120), num_boxes_range=(0, 3),
    )
    samples = [ds[i] for i in range(batch_size)]
    return collate_fn(samples)


class TestEncoders:
    def test_camera_encoder_shape(self) -> None:
        cfg = _tiny_cfg()
        batch = _make_batch(cfg)
        enc = CameraEncoder(
            backbone="resnet18", out_channels=cfg.camera.out_channels, latent_size=cfg.latent_size,
            point_cloud_range=cfg.point_cloud_range, depth_bins=cfg.camera.depth_bins, pretrained=False,
        )
        vol, depth = enc(batch["imgs"], batch["cam_params"])
        X, Y, Z = cfg.latent_size
        assert vol.shape == (1, cfg.T_p, cfg.camera.out_channels, X, Y, Z)
        # Spec §2.1: depth_pred is (B*T, N_cam, depth_bins, H_feat, W_feat).
        assert depth.shape[0] == 1 * cfg.T_p
        assert depth.shape[1] == cfg.N_cam
        assert depth.shape[2] == cfg.camera.depth_bins

    def test_lidar_encoder_shape_and_z_variation(self) -> None:
        cfg = _tiny_cfg()
        batch = _make_batch(cfg)
        enc = LidarEncoder(
            in_channels=4, out_channels=cfg.lidar.out_channels, latent_size=cfg.latent_size,
            point_cloud_range=cfg.point_cloud_range, backbone="pillar3d",
        )
        vol = enc(batch["points"])
        X, Y, Z = cfg.latent_size
        assert vol.shape == (1, cfg.T_p, cfg.lidar.out_channels, X, Y, Z)
        # spec §2.2: output must have genuinely different features at different z (no
        # height-broadcast repeat).
        per_z = [vol[0, 0, :, :, :, z] for z in range(Z)]
        assert not all(torch.allclose(per_z[0], per_z[z]) for z in range(1, Z))


class TestCVQGAndFusion:
    def test_cvqg_shape(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        lidar_vol = torch.randn(1, cfg.T_p, 8, X, Y, Z)
        cam_vol = torch.randn(1, cfg.T_p, 8, X, Y, Z)
        cvqg = CoarseVoxelQueryGenerator(
            embed_dims=16, latent_size=cfg.latent_size, point_cloud_range=cfg.point_cloud_range,
            fusion="gate", lidar_channels=8, cam_channels=8,
        )
        q = cvqg(lidar_vol, cam_vol)
        assert q.shape == (1, cfg.T_p, 16, X, Y, Z)

    def test_cross_modal_fusion_shape(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        a = torch.randn(1, cfg.T_p, 16, X, Y, Z)
        b = torch.randn(1, cfg.T_p, 16, X, Y, Z)
        fuse = CrossModalFusion(channels=16, two_sided=True, aux_heads=True)
        out, aux = fuse(a, b)
        assert out.shape == (1, cfg.T_p, 16, X, Y, Z)
        assert aux["occ_mask_logits"].shape == (1, cfg.T_p, 1, X, Y, Z)


class TestCMLI:
    def test_cmli_shapes(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        lidar_vol = torch.randn(1, cfg.T_p, 8, X, Y, Z)
        cam_vol = torch.randn(1, cfg.T_p, 8, X, Y, Z)
        lidar_mask = torch.ones(1, cfg.T_p)
        cam_mask = torch.ones(1, cfg.T_p)
        cmli = CrossModalLatentImagination(channels=8, latent_size=cfg.latent_size)
        l_out, c_out, aux = cmli(lidar_vol, cam_vol, lidar_mask, cam_mask)
        assert l_out.shape == lidar_vol.shape
        assert c_out.shape == cam_vol.shape
        assert aux["disagreement"].shape == (1, cfg.T_p, 1, X, Y, Z)

    def test_modality_dropout_train_vs_eval(self) -> None:
        cfg = _tiny_cfg()
        batch = _make_batch(cfg)
        dropout = ModalityDropout(lidar_drop_rate=1.0, cam_drop_rate=1.0)  # always drop
        dropout.eval()
        pts, imgs, lm, cm = dropout(batch["points"], batch["imgs"])
        assert torch.all(lm == 1.0) and torch.all(cm == 1.0)  # no-op in eval

        dropout.train()
        pts, imgs, lm, cm = dropout(batch["points"], batch["imgs"])
        assert torch.all(lm == 0.0) and torch.all(cm == 0.0)  # drop_rate=1.0 always drops


class TestTemporalCore:
    def test_taf_shape(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        x = torch.randn(1, cfg.T_p, 16, X, Y, Z)
        taf = TriplingAttentionFusion(embed_dims=16, num_heads=8, window_size=4, timesteps=cfg.T_p)
        out = taf(x)
        assert out.shape == x.shape

    def test_e4a_shape(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        x = torch.randn(1, cfg.T_p, 16, X, Y, Z)
        e4a = EfficientAggregation4D(in_channels=16, embed_dims=16, downsample_layers=2, timesteps=cfg.T_p)
        out = e4a(x)
        assert out.shape == (1, cfg.T_p, 16, X, Y, Z)

    def test_observer_shape(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        fused = torch.randn(1, cfg.T_p, 16, X, Y, Z)
        ego = torch.eye(4).reshape(1, 1, 4, 4).expand(1, cfg.T_p, 4, 4).clone()
        obs = Observer(in_channels=16, embed_dims=16, timesteps=cfg.T_p, downsample_layers=2)
        out = obs(fused, ego)
        assert out.shape == (1, cfg.T_p, 16, X, Y, Z)


class TestStaticAndInstancePaths:
    def test_static_path_shape(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        obs = torch.randn(1, 16, X, Y, Z)
        future_ego = torch.eye(4).reshape(1, 1, 4, 4).expand(1, cfg.T_o, 4, 4).clone()
        path = StaticForecastPath(latent_size=cfg.latent_size, point_cloud_range=cfg.point_cloud_range, channels=16)
        out = path(obs, future_ego)
        assert out.shape == (1, cfg.T_o, 16, X, Y, Z)

    def test_instance_path_shapes(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        obs = torch.randn(1, 16, X, Y, Z)
        extractor = InstanceQueryExtractor(in_channels=16, embed_dims=16, num_queries=8, num_classes=4, num_decoder_layers=1)
        state = extractor(obs)
        assert state.center.shape == (1, 8, 3)
        assert state.logits.shape == (1, 8, 4)

        motion = MotionForecaster(embed_dims=16, num_future=cfg.T_o, mode="gru")
        states = motion(state)
        assert len(states) == cfg.T_o
        assert states[0].center.shape == (1, 8, 3)

        splatter = InstanceSplatter(embed_dims=16, out_channels=16, latent_size=cfg.latent_size,
                                     point_cloud_range=cfg.point_cloud_range, window_voxels=(5, 5, 3))
        dyn_feats, dyn_occ = splatter(states)
        assert dyn_feats.shape == (1, cfg.T_o, 16, X, Y, Z)
        assert dyn_occ.shape == (1, cfg.T_o, 1, X, Y, Z)

    def test_condition_forecaster_shape(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        x = torch.randn(1, cfg.T_p, 16, X, Y, Z)
        cond = ConditionalForecaster(in_timesteps=cfg.T_p, out_timesteps=cfg.T_o, in_channels=16, kernel_size=1)
        out = cond(x)
        assert out.shape == (1, cfg.T_o, 16, X, Y, Z)


class TestHeads:
    def test_head_shapes(self) -> None:
        cfg = _tiny_cfg()
        X, Y, Z = cfg.latent_size
        feats = torch.randn(1, cfg.T_o, 16, X, Y, Z)
        occ_head = OccupancyHead(in_channels=16, num_classes=cfg.num_classes)
        flow_head = FlowHead(in_channels=16)
        unc_head = UncertaintyHead(in_channels=16, num_future=cfg.T_o, use_disagreement=True)

        occ = occ_head(feats)
        flow = flow_head(feats)
        disagreement = torch.randn(1, cfg.T_o, 1, X, Y, Z)
        log_var = unc_head(feats, disagreement)

        assert occ.shape == (1, cfg.T_o, cfg.num_classes, X, Y, Z)
        assert flow.shape == (1, cfg.T_o, 3, X, Y, Z)
        assert log_var.shape == (1, cfg.T_o, 1, X, Y, Z)


class TestDRIFTEndToEnd:
    def test_output_shapes_and_to_indexing_convention(self) -> None:
        """Full DRIFT forward shape check, and the T_o-indexing agreement (seam #5,
        see drift.models.drift module docstring): output index k lines up directly,
        with no shift, against batch['gt_occ'][:, k] / batch['gt_flow'][:, k] --
        i.e. every T_o axis in every returned/consumed tensor has the same length and
        the same meaning throughout the pipeline.
        """
        cfg = _tiny_cfg()
        model = DRIFT(cfg, get_config("tiny").loss)
        model.eval()
        batch = _make_batch(cfg)
        with torch.no_grad():
            out = model(batch)

        X, Y, Z = cfg.latent_size
        assert out["occ_logits"].shape == (1, cfg.T_o, cfg.num_classes, X, Y, Z)
        assert out["flow_pred"].shape == (1, cfg.T_o, 3, X, Y, Z)
        assert out["log_var"].shape == (1, cfg.T_o, 1, X, Y, Z)

        # T_o-indexing agreement: model output T_o must equal the dataset's own gt_occ/gt_flow/
        # gt_boxes T_o (both derived from the same cfg.T_o, with no offset applied anywhere).
        assert batch["gt_occ"].shape[1] == cfg.T_o
        assert batch["gt_flow"].shape[1] == cfg.T_o
        assert len(batch["gt_boxes"][0]) == cfg.T_o
        assert len(out["instance_states"]) == cfg.T_o
        # future_ego (used directly, unshifted, as DecoupledForecaster/StaticForecastPath's
        # `future_ego` argument) also has exactly T_o entries.
        assert batch["future_ego"].shape[1] == cfg.T_o

    def test_no_instance_path_fallback_shapes(self) -> None:
        cfg = get_config("no_instance_path").model
        cfg.latent_size = (16, 16, 4)
        cfg.occ_size = (64, 64, 16)
        cfg.N_cam = 2
        cfg.num_classes = 4
        cfg.camera.backbone = "resnet18"
        cfg.camera.out_channels = 8
        cfg.camera.depth_bins = 12
        cfg.lidar.out_channels = 8
        cfg.fusion.embed_dims = 16
        cfg.observer.downsample_layers = 2
        cfg.refiner.downsample_layers = 2
        model = DRIFT(cfg, get_config("no_instance_path").loss)
        model.eval()
        batch = _make_batch(cfg)
        with torch.no_grad():
            out = model(batch)
        assert out["instance_states"] is None
        X, Y, Z = cfg.latent_size
        assert out["occ_logits"].shape == (1, cfg.T_o, cfg.num_classes, X, Y, Z)
