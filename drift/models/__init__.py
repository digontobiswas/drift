"""DRIFT model modules. See docs/DESIGN_SPEC.md SS2."""

from drift.models.cmli import CrossModalLatentImagination, ModalityDropout
from drift.models.condition import ConditionalForecaster
from drift.models.drift import DRIFT
from drift.models.e4a import EfficientAggregation4D
from drift.models.encoders import (
    CameraEncoder,
    CameraParams,
    CoarseVoxelQueryGenerator,
    CrossModalFusion,
    LidarEncoder,
)
from drift.models.forecaster import DecoupledForecaster
from drift.models.instance_path import (
    InstanceQueryExtractor,
    InstanceSplatter,
    InstanceState,
    MotionForecaster,
)
from drift.models.observer import Observer
from drift.models.predictor import FlowHead, OccupancyHead, UncertaintyHead
from drift.models.refiner import Refiner
from drift.models.static_path import StaticForecastPath
from drift.models.taf import TriplingAttentionFusion

__all__ = [
    "DRIFT",
    "CameraEncoder",
    "CameraParams",
    "LidarEncoder",
    "CoarseVoxelQueryGenerator",
    "CrossModalFusion",
    "CrossModalLatentImagination",
    "ModalityDropout",
    "TriplingAttentionFusion",
    "EfficientAggregation4D",
    "Observer",
    "StaticForecastPath",
    "InstanceState",
    "InstanceQueryExtractor",
    "MotionForecaster",
    "InstanceSplatter",
    "ConditionalForecaster",
    "DecoupledForecaster",
    "Refiner",
    "OccupancyHead",
    "FlowHead",
    "UncertaintyHead",
]
