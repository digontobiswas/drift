"""DRIFT: Decoupled Robust Instance-Flow Temporal occupancy world model.

LiDAR-Camera fusion world model for 4D semantic occupancy forecasting. See
``docs/DESIGN_SPEC.md`` for the full design & interface specification and the
top-level README for an overview, provenance/novelty map, and usage.
"""

from drift.models.drift import DRIFT

__all__ = ["DRIFT"]

__version__ = "0.1.0"
