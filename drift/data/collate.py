"""Batch collation for DRIFT samples.

`torch.utils.data.default_collate` cannot handle this dataset's ragged fields
(`points`: a variable-length point cloud per frame; `gt_boxes`: a variable
object count per frame) or the `CameraParams`/`BoxSet` dataclass containers,
so a dedicated `collate_fn` is required. Samples are expected to come from a
`Dataset` whose `__getitem__` returns the per-sample (no leading `B`) fields
documented in docs/DESIGN_SPEC.md SS4 -- see
`drift.data.cam4docc_dataset.Cam4DOccDataset` / `SyntheticOccDataset`.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor

from drift.data.cam4docc_dataset import CAM_PARAM_FIELDS

try:
    from drift.models.encoders.camera_params import CameraParams
except ImportError:  # pragma: no cover - fallback if the models package is mid-edit / unavailable
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class CameraParams:  # type: ignore[no-redef]
        """Fallback, structurally identical to drift.models.encoders.camera_params.CameraParams.

        Used only if that module cannot be imported (e.g. during concurrent
        development); field names/shapes match the real class exactly, so
        downstream duck-typed consumers behave identically either way.
        """

        rots: Tensor
        trans: Tensor
        intrins: Tensor
        post_rots: Tensor
        post_trans: Tensor


__all__ = ["collate_fn"]

# Fields collated by plain torch.stack (fixed shape, one B dim added).
_STACK_FIELDS = ("imgs", "ego_motion", "future_ego", "gt_occ", "gt_flow", "gt_instance", "lidar_mask", "cam_mask")
# Fields kept as a ragged List[B] of per-sample values (no stacking).
_RAGGED_FIELDS = ("points", "gt_boxes")
# Raw-dict-of-tensors field assembled into a real `CameraParams` at collate time
# (see drift.data.cam4docc_dataset.CAM_PARAM_FIELDS for why it isn't already one).
_CAMERA_PARAM_FIELDS = ("cam_params",)


def _stack_checked(name: str, values: List[Tensor]) -> Tensor:
    """`torch.stack` with a clear error message on shape mismatch across the batch."""
    shape0 = values[0].shape
    for i, v in enumerate(values):
        if v.shape != shape0:
            raise ValueError(
                f"collate_fn: field '{name}' has inconsistent shape across the batch "
                f"(sample 0: {tuple(shape0)}, sample {i}: {tuple(v.shape)}). All non-ragged "
                "fields must share a fixed shape driven by dataset config."
            )
    return torch.stack(values, dim=0)


def _collate_camera_params(name: str, values: List[Any]) -> "CameraParams":
    """Stack a list of per-sample raw camera-param dicts into one batched `CameraParams`.

    Each sample's `cam_params` is a plain `dict` of unbatched `(T, N, ...)`
    tensors (not already a `CameraParams`, since that class validates a
    *batched* `(B, T, N, ...)` shape on construction -- see
    `drift.data.cam4docc_dataset.CAM_PARAM_FIELDS`). This stacks each field
    across the batch and constructs the real `CameraParams` once, at the
    correct shape.
    """
    kwargs = {}
    for field_name in CAM_PARAM_FIELDS:
        per_sample = []
        for v in values:
            if isinstance(v, dict):
                if field_name not in v:
                    raise ValueError(f"collate_fn: '{name}' dict is missing field '{field_name}'")
                per_sample.append(v[field_name])
            elif is_dataclass(v):
                per_sample.append(getattr(v, field_name))
            else:
                raise ValueError(
                    f"collate_fn: field '{name}' must be a dict of tensors or a CameraParams-like "
                    f"dataclass, got {type(v)}"
                )
        kwargs[field_name] = _stack_checked(f"{name}.{field_name}", per_sample)
    return CameraParams(**kwargs)


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of per-sample dicts into the DRIFT batch dict (spec SS4).

    Args:
        batch: List of length `B`, each element the dict returned by a
            DRIFT-protocol `Dataset.__getitem__` (unbatched fields).

    Returns:
        Batch dict with every field carrying a leading `B` dimension, except
        `points` (`List[B][T_p]` ragged tensors) and `gt_boxes`
        (`List[B][T_o]` ragged `BoxSet`s), which stay as nested Python lists.

    Raises:
        ValueError: If `batch` is empty, samples disagree on which keys are
            present, or a nominally fixed-shape field varies in shape across
            the batch.
    """
    if len(batch) == 0:
        raise ValueError("collate_fn received an empty batch")

    keys = set(batch[0].keys())
    for i, sample in enumerate(batch):
        if set(sample.keys()) != keys:
            raise ValueError(
                f"collate_fn: sample {i} has keys {sorted(sample.keys())}, expected {sorted(keys)}"
            )

    out: Dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if key in _RAGGED_FIELDS:
            out[key] = values
        elif key in _CAMERA_PARAM_FIELDS:
            out[key] = _collate_camera_params(key, values)
        elif key in _STACK_FIELDS:
            out[key] = _stack_checked(key, values)
        else:
            # Unknown field (e.g. dataset-specific extras): stack tensors, pass
            # through everything else as a plain list rather than silently dropping it.
            if torch.is_tensor(values[0]):
                out[key] = _stack_checked(key, values)
            else:
                out[key] = values
    return out
