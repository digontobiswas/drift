"""DRIFT loss functions. See docs/DESIGN_SPEC.md SS3."""

from drift.losses.cmli import cmli_consistency_loss
from drift.losses.flow import flow_loss
from drift.losses.instance import instance_loss
from drift.losses.occupancy import (
    CE_ssc_loss,
    downsample_target,
    geo_scal_loss,
    lovasz_softmax,
    occupancy_loss,
    sem_scal_loss,
)
from drift.losses.uncertainty import uncertainty_nll

__all__ = [
    "occupancy_loss",
    "downsample_target",
    "lovasz_softmax",
    "geo_scal_loss",
    "sem_scal_loss",
    "CE_ssc_loss",
    "flow_loss",
    "instance_loss",
    "cmli_consistency_loss",
    "uncertainty_nll",
]
