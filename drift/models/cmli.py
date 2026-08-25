"""Cross-Modal Latent Imagination (CMLI) and its training-time dropout augmentation.

★ NOVEL. No origin paper — this is a DRIFT contribution. See ``docs/DESIGN_SPEC.md`` §2.4
and §0.

Motivation: sensors fail independently in the field (a blinded camera, a LiDAR dropout).
Rather than let a missing modality silently zero out the corresponding branch, CMLI
reconstructs ("imagines") the missing modality's latent from whatever modality survives
plus recent temporal context, so :class:`~drift.models.encoders.cross_modal_fusion.
CrossModalFusion` and everything downstream always sees a plausible input and forecasting
degrades gracefully instead of catastrophically.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

__all__ = ["CrossModalLatentImagination", "ModalityDropout"]


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= max_groups dividing channels."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _Generator(nn.Module):
    """Small 3D conv encoder-decoder: surviving modality (+ temporal context) -> imagined.

    Kept deliberately cheap (a handful of same-resolution conv layers, small `hidden`) so
    the whole CMLI module stays a small fraction of total model FLOPs, per spec ("must be
    cheap, <= 5% of total FLOPs"). Simplification: "encoder-decoder" here is same-
    resolution (no spatial down/up-sampling) rather than a full U-Net, since the latent
    grid is already coarse (spec §1: 128x128x10 or smaller).
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1),
            _group_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            _group_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossModalLatentImagination(nn.Module):
    """★ NOVEL. Reconstruct a missing/degraded modality's latent from the surviving one.

    Processes frames sequentially (causally) within the ``T`` dimension so that, when
    ``use_temporal_context=True``, the imagination generator at frame ``t`` can condition
    on the *resolved* (real-or-imagined) latents from frame ``t-1`` — a lightweight proxy
    for "the previous frame's fused latent" that does not require
    :class:`~drift.models.encoders.cross_modal_fusion.CrossModalFusion` to have already
    run (CMLI sits upstream of fusion in the DRIFT pipeline).

    Args:
        channels: Channel count shared by ``lidar_vol`` and ``cam_vol``.
        latent_size: Latent voxel grid ``(X, Y, Z)``.
        hidden: Hidden channel width of the imagination generators.
        use_temporal_context: If ``True``, condition each frame's generators on the
            previous frame's resolved lidar+cam latents (zeros at ``t=0``).
    """

    def __init__(
        self,
        channels: int,
        latent_size: Tuple[int, int, int] = (128, 128, 10),
        hidden: int = 64,
        use_temporal_context: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}.")
        if len(latent_size) != 3:
            raise ValueError(f"latent_size must be (X, Y, Z), got {latent_size!r}.")

        self.channels = channels
        self.latent_size = tuple(int(v) for v in latent_size)
        self.hidden = hidden
        self.use_temporal_context = use_temporal_context

        ctx_channels = 2 * channels if use_temporal_context else 0
        gen_in = channels + ctx_channels
        # Imagine LiDAR from the surviving camera latent (+ context).
        self.lidar_generator = _Generator(gen_in, channels, hidden)
        # Imagine camera from the surviving LiDAR latent (+ context).
        self.cam_generator = _Generator(gen_in, channels, hidden)

    def forward(
        self,
        lidar_vol: Optional[Tensor],
        cam_vol: Optional[Tensor],
        lidar_mask: Tensor,
        cam_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Substitute imagined latents wherever a modality is missing.

        Args:
            lidar_vol: ``(B, T, C, X, Y, Z)`` or ``None`` if entirely absent this batch
                (treated as all-zero; ``lidar_mask`` should then be all zero too).
            cam_vol: ``(B, T, C, X, Y, Z)`` or ``None``, symmetric to ``lidar_vol``.
            lidar_mask: ``(B, T)`` float in ``{0, 1}``; 1 = LiDAR present at that frame.
            cam_mask: ``(B, T)`` float in ``{0, 1}``; 1 = camera present at that frame.

        Returns:
            Tuple of:
                lidar_out: ``(B, T, C, X, Y, Z)`` — real where ``lidar_mask==1``,
                    imagined where ``lidar_mask==0``.
                cam_out: ``(B, T, C, X, Y, Z)`` — symmetric for camera.
                aux: dict with ``'imagined_lidar'``, ``'imagined_cam'`` (both
                    ``(B, T, C, X, Y, Z)``, for the consistency loss) and
                    ``'disagreement': (B, T, 1, X, Y, Z)`` cross-modal disagreement,
                    fed to the uncertainty head.

        Raises:
            ValueError: On mismatched shapes, or if both ``lidar_vol`` and ``cam_vol``
                are ``None``.
        """
        if lidar_vol is None and cam_vol is None:
            raise ValueError(
                "CrossModalLatentImagination requires at least one of lidar_vol/cam_vol "
                "to be a tensor (to infer batch size, time steps, and spatial shape)."
            )
        ref = lidar_vol if lidar_vol is not None else cam_vol
        if ref.dim() != 6:
            raise ValueError(f"Volumes must be (B, T, C, X, Y, Z), got shape {tuple(ref.shape)}.")
        B, T, C, X, Y, Z = ref.shape
        if C != self.channels:
            raise ValueError(f"Volume has {C} channels, expected channels={self.channels}.")
        if lidar_vol is not None and lidar_vol.shape != ref.shape:
            raise ValueError(
                f"lidar_vol shape {tuple(lidar_vol.shape)} must match cam_vol shape "
                f"{tuple(ref.shape)}."
            )
        if cam_vol is not None and cam_vol.shape != ref.shape:
            raise ValueError(
                f"cam_vol shape {tuple(cam_vol.shape)} must match lidar_vol shape "
                f"{tuple(ref.shape)}."
            )
        if tuple(lidar_mask.shape) != (B, T) or tuple(cam_mask.shape) != (B, T):
            raise ValueError(
                f"lidar_mask/cam_mask must both be (B, T)=({B}, {T}); got "
                f"{tuple(lidar_mask.shape)} and {tuple(cam_mask.shape)}."
            )

        device, dtype = ref.device, ref.dtype
        if lidar_vol is None:
            lidar_vol = torch.zeros(B, T, C, X, Y, Z, device=device, dtype=dtype)
        if cam_vol is None:
            cam_vol = torch.zeros(B, T, C, X, Y, Z, device=device, dtype=dtype)

        lidar_out_list, cam_out_list = [], []
        imagined_lidar_list, imagined_cam_list = [], []
        disagreement_list = []
        prev_context = (
            torch.zeros(B, 2 * C, X, Y, Z, device=device, dtype=dtype)
            if self.use_temporal_context
            else None
        )

        for t in range(T):
            l_t = lidar_vol[:, t]
            c_t = cam_vol[:, t]
            lm = lidar_mask[:, t].view(B, 1, 1, 1, 1).to(dtype)
            cm = cam_mask[:, t].view(B, 1, 1, 1, 1).to(dtype)

            gen_in_l = torch.cat([c_t, prev_context], dim=1) if prev_context is not None else c_t
            gen_in_c = torch.cat([l_t, prev_context], dim=1) if prev_context is not None else l_t
            imagined_l = self.lidar_generator(gen_in_l)
            imagined_c = self.cam_generator(gen_in_c)

            lidar_resolved = lm * l_t + (1.0 - lm) * imagined_l
            cam_resolved = cm * c_t + (1.0 - cm) * imagined_c

            disagreement_t = torch.linalg.vector_norm(
                lidar_resolved - cam_resolved, dim=1, keepdim=True
            )

            lidar_out_list.append(lidar_resolved)
            cam_out_list.append(cam_resolved)
            imagined_lidar_list.append(imagined_l)
            imagined_cam_list.append(imagined_c)
            disagreement_list.append(disagreement_t)

            if self.use_temporal_context:
                prev_context = torch.cat([lidar_resolved, cam_resolved], dim=1)

        lidar_out = torch.stack(lidar_out_list, dim=1)
        cam_out = torch.stack(cam_out_list, dim=1)
        aux = {
            "imagined_lidar": torch.stack(imagined_lidar_list, dim=1),
            "imagined_cam": torch.stack(imagined_cam_list, dim=1),
            "disagreement": torch.stack(disagreement_list, dim=1),
        }
        return lidar_out, cam_out, aux


class ModalityDropout(nn.Module):
    """Training-time sensor-dropout augmentation producing CMLI's presence masks.

    Per sample, per modality, per frame, two independent coin flips decide the outcome:
      1. Full drop, ``Bernoulli(drop_rate)``: the modality is entirely removed at that
         frame (mask -> 0; points cleared / image zeroed).
      2. Otherwise, degrade, ``Bernoulli(degrade_prob)``: the modality stays nominally
         "present" (mask stays 1) but is weakened — LiDAR keeps only a random fraction of
         its points (sampled from ``point_keep_range``), or the image is darkened by a
         random factor (sampled from ``darken_range``).

    A no-op (masks all ones, data unchanged) in eval mode (``self.training == False``).

    Args:
        lidar_drop_rate: Per-(sample, frame) probability of fully dropping LiDAR.
        cam_drop_rate: Per-(sample, frame) probability of fully dropping the camera.
        degrade_prob: Probability of the "degrade" (partial) mode, applied independently
            to each modality that was not fully dropped.
        point_keep_range: ``(min, max)`` fraction of points kept under degrade mode.
        darken_range: ``(min, max)`` multiplicative darkening factor under degrade mode.
    """

    def __init__(
        self,
        lidar_drop_rate: float = 0.1,
        cam_drop_rate: float = 0.1,
        degrade_prob: float = 0.3,
        point_keep_range: Tuple[float, float] = (0.1, 0.5),
        darken_range: Tuple[float, float] = (0.2, 0.6),
    ) -> None:
        super().__init__()
        for name, rate in (
            ("lidar_drop_rate", lidar_drop_rate),
            ("cam_drop_rate", cam_drop_rate),
            ("degrade_prob", degrade_prob),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {rate}.")
        self.lidar_drop_rate = lidar_drop_rate
        self.cam_drop_rate = cam_drop_rate
        self.degrade_prob = degrade_prob
        self.point_keep_range = point_keep_range
        self.darken_range = darken_range

    def sample_masks(
        self, batch_size: int, timesteps: int, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[Tensor, Tensor]:
        """Sample full-drop presence masks only (no physical degradation applied).

        Useful when dropout is applied *after* encoding (per the DRIFT pipeline diagram,
        where ``ModalityDropout`` sits between the encoders and CMLI): the caller can
        sample masks with this method and multiply the already-encoded volumes by them,
        without needing raw points/images.

        Args:
            batch_size: ``B``.
            timesteps: ``T``.
            device: Device for the returned masks.

        Returns:
            ``(lidar_mask, cam_mask)``, each ``(B, T)`` float in ``{0, 1}``. Returns all
            ones for both in eval mode.
        """
        if not self.training:
            ones = torch.ones(batch_size, timesteps, device=device)
            return ones, ones.clone()
        lidar_mask = (
            torch.rand(batch_size, timesteps, device=device) >= self.lidar_drop_rate
        ).float()
        cam_mask = (
            torch.rand(batch_size, timesteps, device=device) >= self.cam_drop_rate
        ).float()
        return lidar_mask, cam_mask

    def forward(
        self, points: list, imgs: Tensor
    ) -> Tuple[list, Tensor, Tensor, Tensor]:
        """Apply full-drop and degrade-mode augmentation to raw points and images.

        Args:
            points: Nested list ``[B][T]`` of ``(N_i, C)`` point tensors.
            imgs: ``(B, T, N_cam, 3, H, W)`` images.

        Returns:
            Tuple of ``(points_out, imgs_out, lidar_mask, cam_mask)`` where
            ``lidar_mask``/``cam_mask`` are ``(B, T)`` float in ``{0, 1}`` (full-drop
            only — degrade mode keeps the mask at 1) and ``points_out``/``imgs_out`` have
            the same structure/shape as the inputs, degraded in place where selected. A
            no-op in eval mode.

        Raises:
            ValueError: If ``points``/``imgs`` batch or time sizes disagree.
        """
        B = len(points)
        T = len(points[0]) if B > 0 else 0
        if imgs.dim() != 6:
            raise ValueError(f"imgs must be (B, T, N_cam, 3, H, W), got {tuple(imgs.shape)}.")
        if imgs.shape[0] != B or imgs.shape[1] != T:
            raise ValueError(
                f"imgs batch/time {(imgs.shape[0], imgs.shape[1])} does not match points "
                f"{(B, T)}."
            )
        device = imgs.device

        if not self.training:
            ones = torch.ones(B, T, device=device)
            return points, imgs, ones, ones.clone()

        lidar_mask = torch.ones(B, T, device=device)
        cam_mask = torch.ones(B, T, device=device)
        points_out = [[p for p in frame] for frame in points]
        imgs_out = imgs.clone()

        for b in range(B):
            for t in range(T):
                if torch.rand(()).item() < self.lidar_drop_rate:
                    lidar_mask[b, t] = 0.0
                    points_out[b][t] = points_out[b][t][:0]
                elif torch.rand(()).item() < self.degrade_prob:
                    keep_frac = torch.empty(()).uniform_(*self.point_keep_range).item()
                    pts = points_out[b][t]
                    n = pts.shape[0]
                    if n > 0:
                        keep_n = max(1, int(round(n * keep_frac)))
                        perm = torch.randperm(n, device=pts.device)[:keep_n]
                        points_out[b][t] = pts[perm]

                if torch.rand(()).item() < self.cam_drop_rate:
                    cam_mask[b, t] = 0.0
                    imgs_out[b, t] = 0.0
                elif torch.rand(()).item() < self.degrade_prob:
                    factor = torch.empty(()).uniform_(*self.darken_range).item()
                    imgs_out[b, t] = imgs_out[b, t] * factor

        return points_out, imgs_out, lidar_mask, cam_mask
