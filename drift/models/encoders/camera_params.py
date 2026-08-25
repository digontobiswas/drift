"""Camera calibration/augmentation container used by the LSS-style camera encoder.

REUSED (structure) — the field layout follows the standard LSS / BEVDet camera-parameter
convention (Philion & Fischer, "Lift, Splat, Shoot", ECCV 2020; Huang et al., "BEVDet",
2021), reimplemented clean-room for DRIFT. See ``docs/DESIGN_SPEC.md`` §2.1.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Union

import torch
from torch import Tensor

__all__ = ["CameraParams"]

# Names of all tensor fields, in a fixed order used for validation and iteration.
_FIELD_NAMES = ("rots", "trans", "intrins", "post_rots", "post_trans")


@dataclass
class CameraParams:
    """Per-camera calibration and image-augmentation parameters.

    All fields share a common ``(B, T, N)`` leading shape (batch, time, camera).

    Attributes:
        rots: Camera-to-ego rotation matrices, ``(B, T, N, 3, 3)``.
        trans: Camera-to-ego translation vectors, ``(B, T, N, 3)``.
        intrins: Camera intrinsic matrices, ``(B, T, N, 3, 3)``.
        post_rots: Image-space augmentation rotation/scale (crop, resize, flip) applied
            after the raw intrinsic projection, ``(B, T, N, 3, 3)``.
        post_trans: Image-space augmentation translation, ``(B, T, N, 3)``.
    """

    rots: Tensor
    trans: Tensor
    intrins: Tensor
    post_rots: Tensor
    post_trans: Tensor

    def __post_init__(self) -> None:
        for name in _FIELD_NAMES:
            value = getattr(self, name)
            if not torch.is_tensor(value):
                raise ValueError(
                    f"CameraParams.{name} must be a torch.Tensor, got {type(value)!r}."
                )
        expected_ndim = {
            "rots": 5,
            "trans": 4,
            "intrins": 5,
            "post_rots": 5,
            "post_trans": 4,
        }
        for name, ndim in expected_ndim.items():
            value = getattr(self, name)
            if value.dim() != ndim:
                raise ValueError(
                    f"CameraParams.{name} must have {ndim} dims (shape "
                    f"{'(B,T,N,3,3)' if ndim == 5 else '(B,T,N,3)'}), got shape "
                    f"{tuple(value.shape)}."
                )
        b, t, n = self.rots.shape[:3]
        for name in _FIELD_NAMES:
            value = getattr(self, name)
            if tuple(value.shape[:3]) != (b, t, n):
                raise ValueError(
                    "CameraParams fields must share a common (B, T, N) leading shape; "
                    f"'rots' has {(b, t, n)} but '{name}' has {tuple(value.shape[:3])}."
                )

    def to(self, device: Union[str, torch.device]) -> "CameraParams":
        """Return a new ``CameraParams`` with every tensor moved to ``device``.

        Args:
            device: Target device (e.g. ``"cuda:0"`` or a ``torch.device``).

        Returns:
            A new ``CameraParams`` instance; the original is left unmodified.
        """
        return CameraParams(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def __getitem__(self, key: Union[int, slice]) -> "CameraParams":
        """Slice along the time dimension (``dim=1``).

        Args:
            key: An ``int`` selecting a single timestep (the returned fields keep a
                length-1 time dimension so downstream shapes stay ``(B, 1, N, ...)``),
                or a ``slice`` selecting a contiguous range of timesteps.

        Returns:
            A new ``CameraParams`` restricted to the requested timestep(s).

        Raises:
            ValueError: If ``key`` is not an ``int`` or ``slice``, or an int key is out
                of range.
        """
        if isinstance(key, bool):
            raise ValueError("CameraParams time index must be an int or slice, not bool.")
        if isinstance(key, int):
            t = self.rots.shape[1]
            if key < -t or key >= t:
                raise ValueError(f"Time index {key} out of range for T={t}.")
            key = key % t
            sl = slice(key, key + 1)
        elif isinstance(key, slice):
            sl = key
        else:
            raise ValueError(
                f"CameraParams.__getitem__ expects an int or slice, got {type(key)!r}."
            )
        return CameraParams(**{f.name: getattr(self, f.name)[:, sl] for f in fields(self)})

    def __len__(self) -> int:
        """Number of timesteps ``T``."""
        return self.rots.shape[1]
