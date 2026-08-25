"""Encoder and fusion modules for DRIFT (see ``docs/DESIGN_SPEC.md`` §2.1-2.5)."""

from drift.models.encoders.camera_encoder import CameraEncoder
from drift.models.encoders.camera_params import CameraParams
from drift.models.encoders.cross_modal_fusion import CrossModalFusion
from drift.models.encoders.lidar_encoder import LidarEncoder
from drift.models.encoders.voxel_query_generator import CoarseVoxelQueryGenerator

__all__ = [
    "CameraParams",
    "CameraEncoder",
    "LidarEncoder",
    "CoarseVoxelQueryGenerator",
    "CrossModalFusion",
]
