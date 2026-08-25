"""Default DRIFT config and named presets/ablations. See ``docs/DESIGN_SPEC.md`` §1, §6.

Every preset below starts from :func:`cam4docc_2s` (the spec's default) and overrides only
what it needs to, so every preset shares the same base hyperparameters except the one axis
it is ablating/varying -- this is what makes the resulting ablation table (see README)
apples-to-apples.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List

from configs.base import DriftConfig

__all__ = [
    "get_config",
    "list_configs",
    "cam4docc_2s",
    "extended_3s",
    "no_cmli",
    "no_instance_path",
    "no_uncertainty",
    "camera_only",
    "lidar_only",
    "fusion_sum",
    "tiny",
]


def cam4docc_2s() -> DriftConfig:
    """Default preset (spec §1 table): T_p=3, T_f=4, T_o=6, +2.0s horizon.

    Cam4DOcc / OccProphet comparable protocol.
    """
    cfg = DriftConfig(name="cam4docc_2s")
    cfg.model.T_p, cfg.model.T_f, cfg.model.T_o = 3, 4, 6
    return cfg


def extended_3s() -> DriftConfig:
    """T_p=3, T_f=6, T_o=8, +3.0s horizon. NOT directly comparable to Cam4DOcc numbers."""
    cfg = cam4docc_2s()
    cfg.name = "extended_3s"
    cfg.model.T_p, cfg.model.T_f, cfg.model.T_o = 3, 6, 8
    return cfg


def no_cmli() -> DriftConfig:
    """Ablation: disable CMLI entirely -- a missing modality is zeroed, not imagined."""
    cfg = cam4docc_2s()
    cfg.name = "no_cmli"
    cfg.model.cmli.enabled = False
    return cfg


def no_instance_path() -> DriftConfig:
    """Ablation: replace the ★NOVEL instance-query dynamic path with a dense-flow-field
    baseline (`drift.models.drift._DenseFlowDynamicPath`), DFIT-OccWorld-style."""
    cfg = cam4docc_2s()
    cfg.name = "no_instance_path"
    cfg.model.forecaster.use_instance_path = False
    return cfg


def no_uncertainty() -> DriftConfig:
    """Ablation: disable the ★NOVEL per-voxel uncertainty head."""
    cfg = cam4docc_2s()
    cfg.name = "no_uncertainty"
    cfg.model.uncertainty.enabled = False
    return cfg


def camera_only() -> DriftConfig:
    """Ablation: no LiDAR encoder at all -- pure camera perception (CMLI, if enabled,
    imagines the missing LiDAR latent from camera + temporal context every frame)."""
    cfg = cam4docc_2s()
    cfg.name = "camera_only"
    cfg.model.use_lidar = False
    return cfg


def lidar_only() -> DriftConfig:
    """Ablation: no camera encoder at all -- pure LiDAR perception."""
    cfg = cam4docc_2s()
    cfg.name = "lidar_only"
    cfg.model.use_camera = False
    return cfg


def fusion_sum() -> DriftConfig:
    """Ablation: CoarseVoxelQueryGenerator uses plain summation (`Q = Q_l + Q_c`) instead
    of the gated fusion default (spec §2.3)."""
    cfg = cam4docc_2s()
    cfg.name = "fusion_sum"
    cfg.model.fusion.cvqg_fusion = "sum"
    return cfg


def tiny() -> DriftConfig:
    """Tiny config for CI / smoke tests (spec §7): latent_size=(16,16,4), num_queries=8,
    B=1 (set by the caller's dataloader), T_p=2, T_o=3. Not one of the paper's named
    presets/ablations -- provided so `tools/*.py` and `tests/*.py` have a fast, CPU-only,
    no-network default without duplicating these numbers everywhere.
    """
    cfg = DriftConfig(name="tiny")
    m = cfg.model
    m.T_p, m.T_f, m.T_o = 2, 2, 3
    m.N_cam = 2
    m.num_classes = 4
    m.latent_size = (16, 16, 4)
    m.occ_size = (64, 64, 16)
    m.camera.backbone = "resnet18"
    m.camera.out_channels = 8
    m.camera.depth_bins = 12
    m.camera.pretrained = False
    m.lidar.out_channels = 8
    m.fusion.embed_dims = 16
    m.observer.downsample_layers = 2
    m.refiner.downsample_layers = 2
    m.forecaster.instance_extractor.embed_dims = 16
    m.forecaster.instance_extractor.num_queries = 8
    m.forecaster.instance_extractor.num_decoder_layers = 1
    m.forecaster.splatter.window_voxels = (5, 5, 3)

    cfg.data.dataset = "synthetic"
    cfg.data.batch_size = 1
    cfg.data.num_samples = 8
    cfg.data.H_img = 32
    cfg.data.W_img = 48
    cfg.data.num_points_range = (50, 120)
    cfg.data.num_boxes_range = (0, 3)

    cfg.train.epochs = 1
    cfg.train.device = "cpu"
    cfg.train.amp = False
    return cfg


_REGISTRY: Dict[str, Callable[[], DriftConfig]] = {
    "cam4docc_2s": cam4docc_2s,
    "extended_3s": extended_3s,
    "no_cmli": no_cmli,
    "no_instance_path": no_instance_path,
    "no_uncertainty": no_uncertainty,
    "camera_only": camera_only,
    "lidar_only": lidar_only,
    "fusion_sum": fusion_sum,
    "tiny": tiny,
}


def list_configs() -> List[str]:
    """Names of every registered preset, e.g. for a CLI's ``--config`` help text."""
    return sorted(_REGISTRY.keys())


def get_config(name: str = "cam4docc_2s") -> DriftConfig:
    """Look up a registered preset by name.

    Args:
        name: One of :func:`list_configs`.

    Returns:
        A fresh ``DriftConfig`` instance -- mutating the result never affects future calls.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown config '{name}'. Available: {list_configs()}")
    return copy.deepcopy(_REGISTRY[name]())
