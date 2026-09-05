"""DRIFT configuration dataclasses. See ``docs/DESIGN_SPEC.md`` §6.

Plain Python dataclasses, no mmcv/mmdet3d dependency. ``DriftConfig`` nests
``ModelConfig``, ``DataConfig``, ``TrainConfig``, ``LossConfig``.
``configs/drift_fusion_nuscenes.py`` instantiates the default and the named
presets/ablations on top of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "POINT_CLOUD_RANGE",
    "OCC_SIZE",
    "LATENT_SIZE",
    "LATENT_DOWNSAMPLE",
    "EMPTY_IDX",
    "CameraEncoderConfig",
    "LidarEncoderConfig",
    "CMLIConfig",
    "FusionConfig",
    "ObserverConfig",
    "StaticPathConfig",
    "InstanceExtractorConfig",
    "MotionForecasterConfig",
    "SplatterConfig",
    "ConditionConfig",
    "ForecasterConfig",
    "RefinerConfig",
    "UncertaintyConfig",
    "HeadConfig",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "LossConfig",
    "DriftConfig",
]

# Global constants, spec §1.
POINT_CLOUD_RANGE: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
OCC_SIZE: Tuple[int, int, int] = (512, 512, 40)
LATENT_DOWNSAMPLE: int = 4
LATENT_SIZE: Tuple[int, int, int] = (128, 128, 10)
EMPTY_IDX: int = 0


@dataclass
class CameraEncoderConfig:
    """See ``drift.models.encoders.camera_encoder.CameraEncoder``."""

    backbone: str = "resnet50"
    out_channels: int = 64
    depth_bins: int = 112
    # Default False: avoids a network call to fetch ImageNet weights (CameraEncoder falls
    # back to random init automatically if unavailable, but tests/CI should not depend on
    # network access at all). Set True for a real training run with internet access.
    pretrained: bool = False


@dataclass
class LidarEncoderConfig:
    """See ``drift.models.encoders.lidar_encoder.LidarEncoder``."""

    in_channels: int = 4
    out_channels: int = 64
    backbone: str = "pillar3d"  # "pillar3d" | "voxelnet"


@dataclass
class CMLIConfig:
    """See ``drift.models.cmli.CrossModalLatentImagination`` / ``ModalityDropout``."""

    enabled: bool = True
    hidden: int = 64
    use_temporal_context: bool = True
    # Sensor-dropout / degradation augmentation disabled (all rates 0.0): ModalityDropout
    # becomes a full no-op, so training sees clean, both-sensors-present frames with no
    # dropped modalities, no LiDAR point subsampling, and no image darkening. The CMLI
    # module still builds (enabled=True) but is simply never exercised. Restore the
    # original 0.1 / 0.1 / 0.3 values to re-enable robustness training.
    lidar_drop_rate: float = 0.0
    cam_drop_rate: float = 0.0
    degrade_prob: float = 0.0
    point_keep_range: Tuple[float, float] = (0.1, 0.5)
    darken_range: Tuple[float, float] = (0.2, 0.6)


@dataclass
class FusionConfig:
    """``CoarseVoxelQueryGenerator`` + ``CrossModalFusion``. Spec §2.3/§2.5."""

    embed_dims: int = 128
    cvqg_fusion: str = "gate"  # "sum" | "concat" | "gate"
    two_sided: bool = True
    aux_heads: bool = True


@dataclass
class ObserverConfig:
    """See ``drift.models.observer.Observer``."""

    downsample_layers: int = 3


@dataclass
class StaticPathConfig:
    """See ``drift.models.static_path.StaticForecastPath``."""

    learned_residual: bool = True


@dataclass
class InstanceExtractorConfig:
    """See ``drift.models.instance_path.InstanceQueryExtractor``."""

    embed_dims: int = 256
    num_queries: int = 300
    num_decoder_layers: int = 3


@dataclass
class MotionForecasterConfig:
    """See ``drift.models.instance_path.MotionForecaster``."""

    mode: str = "gru"  # "gru" | "transformer"


@dataclass
class SplatterConfig:
    """See ``drift.models.instance_path.InstanceSplatter``."""

    soft: bool = True
    sigma: float = 1.0
    window_voxels: Tuple[int, int, int] = (7, 7, 5)


@dataclass
class ConditionConfig:
    """See ``drift.models.condition.ConditionalForecaster``."""

    enabled: bool = True
    kernel_size: int = 1
    norm_and_act: bool = True


@dataclass
class ForecasterConfig:
    """See ``drift.models.forecaster.DecoupledForecaster``."""

    merge: str = "gate"  # "gate" | "sum"
    # False -> the `no_instance_path` ablation: a dense-flow-field dynamic path
    # (`drift.models.drift._DenseFlowDynamicPath`) replaces the novel instance-query path.
    use_instance_path: bool = True
    static: StaticPathConfig = field(default_factory=StaticPathConfig)
    instance_extractor: InstanceExtractorConfig = field(default_factory=InstanceExtractorConfig)
    motion: MotionForecasterConfig = field(default_factory=MotionForecasterConfig)
    splatter: SplatterConfig = field(default_factory=SplatterConfig)
    condition: ConditionConfig = field(default_factory=ConditionConfig)


@dataclass
class RefinerConfig:
    """See ``drift.models.refiner.Refiner``."""

    downsample_layers: int = 2


@dataclass
class UncertaintyConfig:
    """See ``drift.models.predictor.UncertaintyHead``."""

    enabled: bool = True
    mode: str = "variance"  # "variance" | "evidential"
    use_disagreement: bool = True


@dataclass
class HeadConfig:
    """Shared config for ``OccupancyHead`` / ``FlowHead``."""

    hidden_channels: Optional[int] = None


@dataclass
class ModelConfig:
    """Everything ``drift.models.drift.DRIFT`` needs to build itself.

    Attributes:
        T_p: Number of past+present input frames.
        T_f: Number of future frames the protocol evaluates.
        T_o: Number of output frames the model emits (``T_o >= T_f + 1``).
        N_cam: Number of camera views.
        num_classes: Occupancy semantic class count, including class 0 (free/empty).
        latent_size: ``(X, Y, Z)`` internal latent voxel grid.
        point_cloud_range: ``[x_min,y_min,z_min,x_max,y_max,z_max]`` metres.
        occ_size: Full-resolution ground-truth grid; must be an axis-uniform integer
            multiple of ``latent_size``.
        use_camera: If False, no ``CameraEncoder`` is built (``lidar_only`` ablation).
        use_lidar: If False, no ``LidarEncoder`` is built (``camera_only`` ablation).
    """

    T_p: int = 3
    T_f: int = 4
    T_o: int = 6
    N_cam: int = 6
    num_classes: int = 17
    latent_size: Tuple[int, int, int] = LATENT_SIZE
    point_cloud_range: List[float] = field(default_factory=lambda: list(POINT_CLOUD_RANGE))
    occ_size: Tuple[int, int, int] = OCC_SIZE
    use_camera: bool = True
    use_lidar: bool = True
    # Trade compute for memory: re-run each wrapped submodule's forward during the
    # backward pass instead of keeping its activations alive. Costs roughly 30% more
    # step time and changes NOTHING about the model or its outputs -- the arithmetic
    # is identical, only the order in which it happens differs. Needed to fit the
    # dense 5D latent volumes on a 16 GB V100 (PARAM Shakti's only GPU); harmless to
    # leave off on a larger card. Safe here because every norm in the model is
    # GroupNorm, which keeps no running statistics for the second forward to corrupt.
    grad_checkpoint: bool = False
    camera: CameraEncoderConfig = field(default_factory=CameraEncoderConfig)
    lidar: LidarEncoderConfig = field(default_factory=LidarEncoderConfig)
    cmli: CMLIConfig = field(default_factory=CMLIConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    forecaster: ForecasterConfig = field(default_factory=ForecasterConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    occ_head: HeadConfig = field(default_factory=HeadConfig)
    flow_head: HeadConfig = field(default_factory=HeadConfig)

    def __post_init__(self) -> None:
        if len(self.point_cloud_range) != 6:
            raise ValueError(f"point_cloud_range must have 6 entries, got {self.point_cloud_range!r}")
        if self.T_o < self.T_f + 1:
            raise ValueError(f"T_o ({self.T_o}) must be >= T_f+1 ({self.T_f + 1})")
        for axis, (o, l) in enumerate(zip(self.occ_size, self.latent_size)):
            if l <= 0 or o % l != 0:
                raise ValueError(
                    f"occ_size axis {axis}={o} must be a positive integer multiple of "
                    f"latent_size axis {axis}={l}"
                )
        if not self.use_camera and not self.use_lidar:
            raise ValueError("ModelConfig: at least one of use_camera/use_lidar must be True")


@dataclass
class DataConfig:
    """Dataset + dataloader configuration. See ``docs/DESIGN_SPEC.md`` §4.

    ``data_root``/``ann_file`` default to the ``DRIFT_DATA_ROOT`` /
    ``DRIFT_ANN_FILE`` environment variables so that a Slurm script can point a
    whole ablation grid at one preprocessed dataset without editing any config.
    Both are still overridable per-run with ``--data-root`` / ``--ann-file``.
    """

    dataset: str = "synthetic"  # "synthetic" | "cam4docc"
    data_root: str = ""
    ann_file: str = ""
    batch_size: int = 2
    num_workers: int = 0
    # cam4docc-only: channels stored per point in the raw .bin. nuScenes LiDAR
    # sweeps store 5 (x, y, z, intensity, ring); the first `in_channels` are kept.
    point_dims_on_disk: int = 5
    # SyntheticOccDataset-only knobs (ignored for "cam4docc").
    num_samples: int = 64
    H_img: int = 256
    W_img: int = 448
    in_channels: int = 4
    num_points_range: Tuple[int, int] = (300, 900)
    num_boxes_range: Tuple[int, int] = (0, 6)
    modality_dropout_p: float = 0.0
    seed: int = 0


@dataclass
class TrainConfig:
    """Optimization / training-loop configuration. See ``tools/train.py``."""

    epochs: int = 24
    lr: float = 2e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # "adamw" | "sgd"
    amp: bool = True
    grad_clip: float = 35.0
    log_interval: int = 10
    ckpt_dir: str = "work_dirs/drift"
    # Save `latest.pth` every this many optimizer steps, in ADDITION to the
    # epoch-end save. On real data an epoch is ~8 hours, so epoch-end-only saving
    # means any mid-epoch crash (a segfault, a node failure, a walltime kill)
    # throws away every hour of work since the epoch began. Set 0 to disable.
    #
    # At ~2.6 s/step, 200 steps caps that loss at roughly 9 minutes. The interval
    # has to be short relative to the crash-free interval, not merely cheap: the
    # Slurm scripts refuse to requeue a run that ended without passing a checkpoint,
    # so that a job failing on startup cannot spawn an endless chain of failures.
    # Runs on this cluster segfault every ~1500-3500 steps, and at 500 that guard
    # kept tripping on runs that died before banking anything, stalling the chain
    # until it was resubmitted by hand. A 507 MB save every 200 steps costs ~3%.
    ckpt_interval_steps: int = 200
    resume: Optional[str] = None
    seed: int = 0
    device: str = "cuda"
    distributed: bool = False
    max_iters: Optional[int] = None  # if set, stop after this many optimizer steps (debug/CI)
    # Learning-rate schedule, applied per optimizer step (not per epoch).
    #   "constant" -- fixed `lr` throughout (the original behaviour)
    #   "cosine"   -- linear warmup over `warmup_iters` steps, then cosine decay
    #                 from `lr` down to `lr * min_lr_ratio` at the final step
    #   "step"     -- linear warmup, then multiply by `step_gamma` at each of
    #                 `step_milestones` (fractions of total training)
    lr_scheduler: str = "cosine"
    warmup_iters: int = 500
    warmup_start_ratio: float = 0.001  # LR at step 0, as a fraction of `lr`
    min_lr_ratio: float = 0.01
    step_gamma: float = 0.1
    step_milestones: List[float] = field(default_factory=lambda: [0.7, 0.9])


@dataclass
class LossConfig:
    """Loss-term weights. See ``docs/DESIGN_SPEC.md`` §3."""

    class_weights: Optional[List[float]] = None  # None -> occupancy_loss's own default (1.0 / 5.0)
    occ_weights: Dict[str, float] = field(
        default_factory=lambda: {"ce": 1.0, "lovasz": 1.0, "geo_scal": 1.0, "sem_scal": 1.0}
    )
    flow_weight: float = 1.0
    instance_weight: float = 1.0
    cmli_weight: float = 1.0
    uncertainty_weight: float = 1.0
    # Present-frame detection auxiliary on the query extractor: shapes the shared representation
    # and keeps the extractor's objectness/velocity heads supervised.
    detection_weight: float = 0.5
    # Binary occupied/free auxiliary on the fused latent (Doracamom Sec. III-F).
    aux_occ_weight: float = 0.5


@dataclass
class DriftConfig:
    """Top-level config: everything needed to build the model, data, and trainer."""

    name: str = "drift_fusion_nuscenes"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
