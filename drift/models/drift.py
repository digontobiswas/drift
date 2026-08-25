"""DRIFT: top-level model assembly. See ``docs/DESIGN_SPEC.md`` §2.15.

Wires together every already-specified submodule into the full pipeline:

```
imgs, points
  -> CameraEncoder, LidarEncoder                       (B,T_p,C,X,Y,Z) each
  -> ModalityDropout (train) -> CMLI                   fills missing modalities
  -> CoarseVoxelQueryGenerator -> CrossModalFusion      fused (B,T_p,C,X,Y,Z)
  -> warp past frames into present frame (ego motion)
  -> Observer                                           O_obs (B,T_p,C',X,Y,Z)
  -> DecoupledForecaster                                (B,T_o,C',X,Y,Z)
  -> Refiner                                             (B,T_o,C',X,Y,Z)
  -> OccupancyHead / FlowHead / UncertaintyHead
```

This file resolves five interface seams left open by the individually-verified
submodules (flagged in the implementation brief). Each is documented at its
call site below with a ``SEAM #n`` comment; summarized here:

1. **CMLI consistency-loss aux.** ``cmli_consistency_loss`` (``drift/losses/cmli.py``)
   expects ``aux['real_lidar']``/``aux['real_cam']`` — the pre-substitution
   encoder outputs — which ``CrossModalLatentImagination.forward`` does not
   itself emit (by design, per its own docstring). This module captures the
   encoder outputs *before* calling CMLI and injects them into ``aux``.
2. **Disagreement/horizon alignment.** CMLI's ``disagreement`` lives on the
   ``T_p`` (observation) time axis; ``UncertaintyHead`` requires it on the
   ``T_o`` (future-horizon) axis. There is no future *observation* to
   disagree over, so this module broadcasts the present frame's (most
   recent, most relevant) disagreement across every future horizon rather
   than learning a temporal projection — simple, parameter-free, and the
   right inductive bias for a signal that is fundamentally about "how much
   do the two modalities currently conflict".
3. **ModalityDropout ordering.** Its ``forward()`` (pre-encoder, physically
   degrades raw points/images) is used during training for realism; its
   full-drop masks are combined (multiplicatively, i.e. logical AND) with
   whatever presence masks the dataset itself provides, so a sensor the
   dataset already marks absent stays absent regardless of augmentation.
   ``sample_masks()`` (post-encoder) is unused here. In eval mode
   ``ModalityDropout`` is a no-op by its own construction.
4. **CVQG / CrossModalFusion channel wiring.** ``CrossModalFusion`` requires
   both inputs already at the same channel count, but ``CoarseVoxelQueryGenerator``
   internally projects lidar/camera volumes to its *own* output width and
   does not expose intermediate same-width tensors. This module owns two
   1x1 ``Conv3d`` "channel adapters" (encoder width -> ``embed_dims``) so
   ``CrossModalFusion`` can run on genuinely comparable projections of the
   *raw* lidar/camera volumes; its symmetric two-sided-gated output is then
   added to CVQG's geometry-anchored, camera-gated query volume. Both design
   contributions (§0's CVQG and CMF modifications) therefore feed the
   representation that reaches ``Observer``, matching the spec pipeline's
   informal "CVQG -> CMF -> fused" ordering while making the channel
   requirement explicit rather than silently reusing one module's internal
   projections for the other.
5. **T_o indexing convention.** The already-implemented, independently
   verified data and model modules (``drift/data/ego_motion.py``'s
   ``compose_future_transforms``, ``drift/data/cam4docc_dataset.py``'s GT
   rasterization, ``StaticForecastPath``, ``MotionForecaster``) all agree,
   consistently, that **every** ``T_o`` index is a genuine future frame:
   index ``k`` is the state ``(k+1) * 0.5s`` after the present frame. There
   is no "present-frame reconstruction" slot anywhere in the verified data
   contract (no present-time occupancy/flow/box ground truth is ever
   produced). ``drift/metrics/iou.py`` is written to *support* a present
   slot (a configurable ``present_index``, default 0) but does not require
   one to exist physically — it just labels whichever index is passed as
   "present" for bucketing purposes. This module therefore does **not**
   reshuffle or prepend anything: model output index ``k`` is used directly
   against ``batch['gt_occ'][:, k]`` / ``batch['gt_flow'][:, k]`` /
   ``batch['gt_boxes'][b][k]`` with no shift, and evaluation code calls
   ``OccupancyIoUMetric(..., present_index=0)`` with the documented
   understanding that "IoU_c" here reports the *nearest*-horizon (t=+0.5s)
   quality rather than a literal same-timestamp reconstruction (see
   ``tools/eval.py`` and the README's known-issues section). Changing this
   would require rewriting the (verified, internally self-consistent) data
   and forecasting modules for no numerical benefit — the two conventions
   are a strict relabeling of the same indices. ``tests/test_shapes.py``
   asserts this convention explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint  # explicit: `import torch` alone need not bind this submodule
from torch import Tensor, nn

from drift.data.ego_motion import cumulative_warp_to_present
from drift.losses.cmli import cmli_consistency_loss
from drift.losses.flow import flow_loss
from drift.losses.instance import instance_loss
from drift.losses.occupancy import downsample_target, occupancy_loss
from drift.losses.uncertainty import uncertainty_nll
from drift.models.cmli import CrossModalLatentImagination, ModalityDropout
from drift.models.encoders.camera_encoder import CameraEncoder
from drift.models.encoders.cross_modal_fusion import CrossModalFusion
from drift.models.encoders.lidar_encoder import LidarEncoder
from drift.models.encoders.voxel_query_generator import CoarseVoxelQueryGenerator
from drift.models.forecaster import DecoupledForecaster
from drift.models.observer import Observer
from drift.models.predictor import FlowHead, OccupancyHead, UncertaintyHead
from drift.models.refiner import Refiner
from drift.models.static_path import StaticForecastPath

if TYPE_CHECKING:  # pragma: no cover - type-checking only, avoids a hard runtime dependency
    from configs.base import LossConfig, ModelConfig

__all__ = ["DRIFT"]


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _bce_dice(logits: Tensor, target: Tensor, valid: Tensor, eps: float = 1.0) -> Tensor:
    """Binary cross-entropy + soft Dice over the valid voxels.

    Used for the auxiliary occupied/free supervision on the fused latent. Dice complements BCE
    when the occupied class is a small fraction of the volume, which it always is here.

    Args:
        logits: (B, X, Y, Z) raw logits.
        target: (B, X, Y, Z) float in {0, 1}.
        valid: (B, X, Y, Z) bool mask of supervised voxels.

    Returns:
        Scalar loss. Returns a graph-connected zero when no voxel is valid.
    """
    if valid.sum() == 0:
        return (logits * 0.0).sum()
    v = valid.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=v, reduction="sum") / v.sum()
    prob = torch.sigmoid(logits) * v
    tgt = target * v
    inter = (prob * tgt).sum()
    dice = 1.0 - (2.0 * inter + eps) / (prob.sum() + tgt.sum() + eps)
    return bce + dice


class _DenseFlowDynamicPath(nn.Module):
    """Ablation-only dense-flow dynamic-path fallback (``no_instance_path`` preset).

    Origin: DFIT-OccWorld (2412.13772)-style decoupling — a static branch reached by
    ego-pose warping (``StaticForecastPath``, used unchanged by the caller) plus a
    *dense per-voxel flow field* dynamic branch, in contrast to DRIFT's own ★ NOVEL
    instance-query dynamic path (``drift/models/instance_path.py``). Included so the
    ablation comparison in the README's ablation table is an apples-to-apples swap of
    only the dynamic-path mechanism, with everything else (static path, merge gate,
    refiner, heads) identical.

    A small conv head regresses a dense ``(T_o, 3, X, Y, Z)`` voxel-unit flow field from
    the present-frame observation latent in a single shot (no per-step recurrence, unlike
    ``MotionForecaster`` — a deliberate simplification since this is a baseline, not the
    paper's contribution); each future frame is then produced by backward-warping the
    present latent through its predicted flow via ``grid_sample``, mirroring
    ``StaticForecastPath``'s warp mechanics but with a *learned per-voxel* displacement
    instead of a single rigid ego transform.
    """

    def __init__(self, channels: int, num_future: int, hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.num_future = num_future
        self.flow_net = nn.Sequential(
            nn.Conv3d(channels, hidden, 3, padding=1),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, 3 * num_future, kernel_size=1),
        )
        self.feat_proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, obs_latent: Tensor, future_ego: Tensor) -> "tuple[Tensor, Tensor]":
        """
        Args:
            obs_latent: ``(B, T_p, C, X, Y, Z)``; only the present frame (index -1) is used.
            future_ego: ``(B, T_o, 4, 4)``, unused (kept for interface parity with
                ``DecoupledForecaster.forward`` / the dense-flow baseline not needing ego
                pose since it predicts motion directly in the present ego frame).

        Returns:
            ``dyn_feats: (B, T_o, C, X, Y, Z)``, ``dyn_occ: (B, T_o, 1, X, Y, Z)``.
        """
        del future_ego
        present = obs_latent[:, -1]  # (B,C,X,Y,Z)
        b, c, x_dim, y_dim, z_dim = present.shape
        device, dtype = present.device, present.dtype

        flow = self.flow_net(present).view(b, self.num_future, 3, x_dim, y_dim, z_dim)

        xs = torch.linspace(-1.0, 1.0, x_dim, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, y_dim, device=device, dtype=dtype)
        zs = torch.linspace(-1.0, 1.0, z_dim, device=device, dtype=dtype)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        base_grid = torch.stack([gz, gy, gx], dim=-1)  # (X,Y,Z,3), grid_sample (D,H,W)=(X,Y,Z) order

        scale = torch.tensor(
            [2.0 / max(x_dim - 1, 1), 2.0 / max(y_dim - 1, 1), 2.0 / max(z_dim - 1, 1)],
            device=device, dtype=dtype,
        )

        feats_out: List[Tensor] = []
        occ_out: List[Tensor] = []
        for t in range(self.num_future):
            f_t = flow[:, t]  # (B,3,X,Y,Z), voxel units (dx,dy,dz)
            disp = (f_t.permute(0, 2, 3, 4, 1) * scale)  # (B,X,Y,Z,3), normalized-grid units
            disp = disp[..., [2, 1, 0]]  # reorder to (dz,dy,dx) to match base_grid's axis order
            grid = base_grid.unsqueeze(0) + disp
            sampled = F.grid_sample(
                present, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            sampled = self.feat_proj(sampled)
            feats_out.append(sampled)
            occ_out.append(torch.sigmoid(f_t.norm(dim=1, keepdim=True) - 1.0))

        dyn_feats = torch.stack(feats_out, dim=1)
        dyn_occ = torch.stack(occ_out, dim=1)
        return dyn_feats, dyn_occ


class DRIFT(nn.Module):
    """Top-level assembly. See ``docs/DESIGN_SPEC.md`` §2.15.

    Args:
        cfg: Model configuration (``configs.base.ModelConfig``). Every hyperparameter is
            reachable from this config per spec §6; see ``configs/drift_fusion_nuscenes.py``
            for the named presets/ablations.
        loss_cfg: Loss-term weighting configuration (``configs.base.LossConfig``); required
            by :meth:`loss`.
    """

    def __init__(self, cfg: "ModelConfig", loss_cfg: "LossConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = loss_cfg

        X, Y, Z = cfg.latent_size
        C = cfg.fusion.embed_dims
        self.C = C
        self.latent_size = (X, Y, Z)
        self.grad_checkpoint = getattr(cfg, "grad_checkpoint", False)

        ratios = [o // l for o, l in zip(cfg.occ_size, cfg.latent_size)]
        if any(o % l != 0 for o, l in zip(cfg.occ_size, cfg.latent_size)) or len(set(ratios)) != 1:
            raise ValueError(
                f"DRIFT requires occ_size {cfg.occ_size} to be an integer, axis-uniform multiple "
                f"of latent_size {cfg.latent_size}; got per-axis ratios {ratios}."
            )
        self.latent_downsample = ratios[0]

        # -- 1. Encoders --------------------------------------------------------------
        self.camera_encoder: Optional[CameraEncoder] = (
            CameraEncoder(
                backbone=cfg.camera.backbone, out_channels=cfg.camera.out_channels,
                latent_size=cfg.latent_size, point_cloud_range=cfg.point_cloud_range,
                depth_bins=cfg.camera.depth_bins, pretrained=cfg.camera.pretrained,
            )
            if cfg.use_camera else None
        )
        self.lidar_encoder: Optional[LidarEncoder] = (
            LidarEncoder(
                in_channels=cfg.lidar.in_channels, out_channels=cfg.lidar.out_channels,
                latent_size=cfg.latent_size, point_cloud_range=cfg.point_cloud_range,
                backbone=cfg.lidar.backbone,
            )
            if cfg.use_lidar else None
        )
        self.cam_channels = cfg.camera.out_channels
        self.lidar_channels = cfg.lidar.out_channels

        # -- 2. ModalityDropout + CMLI (SEAM #1, #3) -----------------------------------
        if cfg.cmli.enabled:
            if self.cam_channels != self.lidar_channels:
                raise ValueError(
                    "DRIFT: CMLI requires cfg.camera.out_channels == cfg.lidar.out_channels "
                    f"(got {self.cam_channels} vs {self.lidar_channels}), since "
                    "CrossModalLatentImagination shares one channel width between modalities."
                )
            self.modality_dropout: Optional[ModalityDropout] = ModalityDropout(
                lidar_drop_rate=cfg.cmli.lidar_drop_rate, cam_drop_rate=cfg.cmli.cam_drop_rate,
                degrade_prob=cfg.cmli.degrade_prob, point_keep_range=cfg.cmli.point_keep_range,
                darken_range=cfg.cmli.darken_range,
            )
            self.cmli: Optional[CrossModalLatentImagination] = CrossModalLatentImagination(
                channels=self.cam_channels, latent_size=cfg.latent_size, hidden=cfg.cmli.hidden,
                use_temporal_context=cfg.cmli.use_temporal_context,
            )
        else:
            self.modality_dropout = None
            self.cmli = None

        # -- 3. CVQG + channel adapters + CrossModalFusion (SEAM #4) -------------------
        self.cvqg = CoarseVoxelQueryGenerator(
            embed_dims=C, latent_size=cfg.latent_size, point_cloud_range=cfg.point_cloud_range,
            fusion=cfg.fusion.cvqg_fusion, lidar_channels=self.lidar_channels, cam_channels=self.cam_channels,
        )
        self.lidar_adapter = nn.Conv3d(self.lidar_channels, C, kernel_size=1)
        self.cam_adapter = nn.Conv3d(self.cam_channels, C, kernel_size=1)
        self.cross_modal_fusion = CrossModalFusion(
            channels=C, two_sided=cfg.fusion.two_sided, aux_heads=cfg.fusion.aux_heads,
        )

        # -- 4. Observer ----------------------------------------------------------------
        self.observer = Observer(
            in_channels=C, embed_dims=C, timesteps=cfg.T_p,
            downsample_layers=cfg.observer.downsample_layers,
        )

        # -- 5. Forecaster (novel instance path, or dense-flow ablation fallback) -------
        if cfg.forecaster.use_instance_path:
            instance_cfg = {
                "extractor": {
                    "embed_dims": cfg.forecaster.instance_extractor.embed_dims,
                    "num_queries": cfg.forecaster.instance_extractor.num_queries,
                    # Design decision: the instance-classification vocabulary is the SAME
                    # occupancy semantic vocabulary as OccupancyHead (BoxSet.label, per the
                    # data contract, already holds occupancy class ids >=2; there is no
                    # separate, smaller "object type" taxonomy in this data contract). Using
                    # a different, smaller num_classes here (e.g. the spec default of 2)
                    # would make instance_loss's F.one_hot(label, num_classes) crash the
                    # instant a dynamic object's label exceeds it.
                    "num_classes": cfg.num_classes,
                    "num_decoder_layers": cfg.forecaster.instance_extractor.num_decoder_layers,
                },
                "motion": {"mode": cfg.forecaster.motion.mode},
                "splatter": {
                    "soft": cfg.forecaster.splatter.soft, "sigma": cfg.forecaster.splatter.sigma,
                    "window_voxels": cfg.forecaster.splatter.window_voxels,
                },
            }
            condition_cfg = None
            if cfg.forecaster.condition.enabled:
                condition_cfg = {
                    "in_timesteps": cfg.T_p,  # required explicitly; see module docstring seam list
                    "kernel_size": cfg.forecaster.condition.kernel_size,
                    "norm_and_act": cfg.forecaster.condition.norm_and_act,
                }
            self.forecaster: Optional[DecoupledForecaster] = DecoupledForecaster(
                channels=C, num_future=cfg.T_o,
                static_cfg={"learned_residual": cfg.forecaster.static.learned_residual},
                instance_cfg=instance_cfg, condition_cfg=condition_cfg, merge=cfg.forecaster.merge,
                latent_size=cfg.latent_size, point_cloud_range=cfg.point_cloud_range,
            )
            # Propagate the memory/compute trade into the forecaster's own internals,
            # which hold the largest volumes in the model.
            self.forecaster.grad_checkpoint = self.grad_checkpoint
            self.static_path = None
            self.dense_flow_fallback = None
            self.fallback_gate = None
        else:
            self.forecaster = None
            self.static_path = StaticForecastPath(
                latent_size=cfg.latent_size, point_cloud_range=cfg.point_cloud_range,
                learned_residual=cfg.forecaster.static.learned_residual, channels=C,
            )
            self.dense_flow_fallback = _DenseFlowDynamicPath(channels=C, num_future=cfg.T_o)
            self.fallback_gate = nn.Conv3d(2 * C + 1, 1, kernel_size=1)

        # -- 6. Refiner -------------------------------------------------------------------
        self.refiner = Refiner(
            channels=C, in_timesteps=cfg.T_p, out_timesteps=cfg.T_o,
            downsample_layers=cfg.refiner.downsample_layers,
        )

        # -- 7. Heads ----------------------------------------------------------------------
        self.occ_head = OccupancyHead(
            in_channels=C, num_classes=cfg.num_classes, hidden_channels=cfg.occ_head.hidden_channels,
        )
        self.flow_head = FlowHead(in_channels=C, hidden_channels=cfg.flow_head.hidden_channels)
        self.uncertainty_head: Optional[UncertaintyHead] = (
            UncertaintyHead(
                in_channels=C, num_future=cfg.T_o, mode=cfg.uncertainty.mode,
                use_disagreement=cfg.uncertainty.use_disagreement,
            )
            if cfg.uncertainty.enabled else None
        )

    def _ckpt(self, fn: Any, *args: Any) -> Any:
        """Run ``fn(*args)`` under gradient checkpointing when it is enabled.

        Checkpointing drops a submodule's intermediate activations after its
        forward and recomputes them during backward. The dense 5D latent volumes
        here are ~500 MB apiece at ``embed_dims=128``, so keeping every one alive
        across the whole pipeline overruns a 16 GB V100 before the backward pass
        even starts; recomputing costs ~30% step time and nothing else -- the
        arithmetic and therefore the results are unchanged.

        Only active while training with grad enabled: under ``eval()``/``no_grad``
        there is no backward pass to trade against, so checkpointing would be pure
        overhead (and ``checkpoint`` warns when nothing requires grad).

        ``use_reentrant=False`` is deliberate: it preserves RNG state, so
        ``ModalityDropout`` draws the same mask on the recomputed forward as on the
        first one. Safe against the usual double-forward hazard because every norm
        in this model is GroupNorm, which holds no running statistics to corrupt.
        """
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return fn(*args)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Run the full DRIFT pipeline on one batch.

        Args:
            batch: Dict following the contract in ``docs/DESIGN_SPEC.md`` §4 (see
                ``drift.data.collate.collate_fn`` / ``drift.data.cam4docc_dataset``).

        Returns:
            Dict with ``occ_logits (B,T_o,num_classes,X,Y,Z)``, ``flow_pred (B,T_o,3,X,Y,Z)``,
            ``log_var`` (``(B,T_o,1,X,Y,Z)`` or ``None`` if uncertainty is disabled),
            ``depth_pred`` (camera depth logits, or ``None``), ``lidar_mask``/``cam_mask``
            (effective, post-augmentation ``(B,T_p)`` presence masks), ``cmli_aux`` (dict,
            empty if CMLI disabled), ``fusion_aux`` (``CrossModalFusion``'s aux dict),
            ``instance_states`` (``List[InstanceState]`` of length ``T_o``, or ``None`` under
            the ``no_instance_path`` ablation), and ``dyn_occ``.
        """
        imgs: Tensor = batch["imgs"]
        cam_params = batch["cam_params"]
        points = batch["points"]
        ego_motion: Tensor = batch["ego_motion"]
        future_ego: Tensor = batch["future_ego"]

        B, T_p = imgs.shape[0], imgs.shape[1]
        device, dtype = imgs.device, imgs.dtype
        X, Y, Z = self.latent_size
        C = self.C
        cfg = self.cfg

        base_lidar_mask = batch.get("lidar_mask")
        base_cam_mask = batch.get("cam_mask")
        base_lidar_mask = (
            base_lidar_mask.to(device=device, dtype=dtype) if base_lidar_mask is not None
            else torch.ones(B, T_p, device=device, dtype=dtype)
        )
        base_cam_mask = (
            base_cam_mask.to(device=device, dtype=dtype) if base_cam_mask is not None
            else torch.ones(B, T_p, device=device, dtype=dtype)
        )

        # SEAM #3: ModalityDropout ordering -- pre-encoder forward() for physical realism;
        # combine its full-drop masks with the dataset's own presence masks (AND, via
        # multiplication) rather than overriding them.
        if self.modality_dropout is not None:
            points, imgs, aug_lidar_mask, aug_cam_mask = self.modality_dropout(points, imgs)
            lidar_mask = base_lidar_mask * aug_lidar_mask
            cam_mask = base_cam_mask * aug_cam_mask
        else:
            lidar_mask = base_lidar_mask
            cam_mask = base_cam_mask

        if self.lidar_encoder is not None:
            lidar_vol: Optional[Tensor] = self._ckpt(self.lidar_encoder, points)
        else:
            lidar_vol = None
            lidar_mask = torch.zeros(B, T_p, device=device, dtype=dtype)

        depth_pred: Optional[Tensor] = None
        if self.camera_encoder is not None:
            cam_vol: Optional[Tensor] = None
            # The camera encoder is the single largest activation consumer: T_p x N_cam
            # (3 x 6 = 18) images through a ResNet50 at 256x704, plus the LSS-style depth
            # splat. Checkpointing it is where most of the memory saving comes from.
            cam_vol, depth_pred = self._ckpt(self.camera_encoder, imgs, cam_params)
        else:
            cam_vol = None
            cam_mask = torch.zeros(B, T_p, device=device, dtype=dtype)

        cmli_aux: Dict[str, Tensor] = {}
        if self.cmli is not None:
            real_lidar, real_cam = lidar_vol, cam_vol
            lidar_vol, cam_vol, cmli_aux = self.cmli(lidar_vol, cam_vol, lidar_mask, cam_mask)
            # SEAM #1: inject the pre-substitution teacher latents cmli_consistency_loss expects.
            if real_lidar is not None:
                cmli_aux["real_lidar"] = real_lidar
            if real_cam is not None:
                cmli_aux["real_cam"] = real_cam
        else:
            # No imagination network: a "missing" modality is honestly zeroed (not imagined),
            # which is the correct no_cmli-ablation baseline behaviour.
            if lidar_vol is None:
                lidar_vol = torch.zeros(B, T_p, self.lidar_channels, X, Y, Z, device=device, dtype=dtype)
            else:
                lidar_vol = lidar_vol * lidar_mask.reshape(B, T_p, 1, 1, 1, 1)
            if cam_vol is None:
                cam_vol = torch.zeros(B, T_p, self.cam_channels, X, Y, Z, device=device, dtype=dtype)
            else:
                cam_vol = cam_vol * cam_mask.reshape(B, T_p, 1, 1, 1, 1)

        # SEAM #4: CVQG operates on the raw (post-CMLI) per-modality volumes at their own
        # channel widths; CrossModalFusion needs both at the same width, so we adapt.
        query_vol = self.cvqg(lidar_vol, cam_vol)  # (B,T_p,C,X,Y,Z)

        lidar_flat = lidar_vol.reshape(B * T_p, self.lidar_channels, X, Y, Z)
        cam_flat = cam_vol.reshape(B * T_p, self.cam_channels, X, Y, Z)
        lidar_adapted = self._ckpt(self.lidar_adapter, lidar_flat).reshape(B, T_p, C, X, Y, Z)
        cam_adapted = self._ckpt(self.cam_adapter, cam_flat).reshape(B, T_p, C, X, Y, Z)
        fused_cmf, fusion_aux = self.cross_modal_fusion(lidar_adapted, cam_adapted)
        fused = fused_cmf + query_vol  # (B,T_p,C,X,Y,Z)

        fused = cumulative_warp_to_present(
            fused, ego_motion, present_idx=T_p - 1, point_cloud_range=cfg.point_cloud_range,
        )

        obs = self._ckpt(self.observer, fused, ego_motion[:, :T_p])  # (B,T_p,C,X,Y,Z)

        if self.forecaster is not None:
            future_latent, forecaster_aux = self.forecaster(obs, future_ego)
        else:
            present = obs[:, -1]
            static_latent = self.static_path(present, future_ego)  # type: ignore[misc]
            dyn_latent, dyn_occ = self.dense_flow_fallback(obs, future_ego)  # type: ignore[misc]
            gate_in = torch.cat([static_latent, dyn_latent, dyn_occ], dim=2)
            gate_in = gate_in.reshape(B * cfg.T_o, 2 * C + 1, X, Y, Z)
            a = torch.sigmoid(self.fallback_gate(gate_in)).reshape(B, cfg.T_o, 1, X, Y, Z)  # type: ignore[misc]
            future_latent = static_latent * (1 - a) + dyn_latent * a
            forecaster_aux = {
                "instance_states": None, "dyn_occ": dyn_occ,
                "static_latent": static_latent, "dyn_latent": dyn_latent, "present_state": None,
            }

        refined = self._ckpt(self.refiner, obs, future_latent)  # (B,T_o,C,X,Y,Z)

        occ_logits = self._ckpt(self.occ_head, refined)
        flow_pred = self._ckpt(self.flow_head, refined)

        log_var: Optional[Tensor] = None
        if self.uncertainty_head is not None:
            disagreement_future: Optional[Tensor] = None
            if cfg.uncertainty.use_disagreement:
                # SEAM #2: broadcast the present-frame disagreement across every horizon.
                disagreement = cmli_aux.get("disagreement")
                if disagreement is not None:
                    disagreement_future = disagreement[:, -1:].expand(B, cfg.T_o, 1, X, Y, Z)
                else:
                    disagreement_future = torch.zeros(
                        B, cfg.T_o, 1, X, Y, Z, device=device, dtype=refined.dtype
                    )
            log_var = self.uncertainty_head(refined, disagreement_future)

        return {
            "occ_logits": occ_logits,
            "flow_pred": flow_pred,
            "log_var": log_var,
            "depth_pred": depth_pred,
            "lidar_mask": lidar_mask,
            "cam_mask": cam_mask,
            "cmli_aux": cmli_aux,
            "fusion_aux": fusion_aux,
            "instance_states": forecaster_aux.get("instance_states"),
            "present_state": forecaster_aux.get("present_state"),
            "dyn_occ": forecaster_aux.get("dyn_occ"),
        }

    def loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Compute every supervised loss term. See ``docs/DESIGN_SPEC.md`` §3.

        Args:
            outputs: The dict returned by :meth:`forward`.
            batch: The same batch dict passed to :meth:`forward`.

        Returns:
            Dict of scalar tensors, every key prefixed ``loss_``. The caller (the trainer)
            is responsible for summing the values into the total loss.
        """
        cfg = self.loss_cfg
        occ_logits = outputs["occ_logits"]
        device, dtype = occ_logits.device, occ_logits.dtype

        class_weights = None
        if cfg.class_weights is not None:
            class_weights = torch.tensor(cfg.class_weights, device=device, dtype=dtype)

        target_latent = downsample_target(batch["gt_occ"], ratio=self.latent_downsample)

        occ_terms = occupancy_loss(
            occ_logits, target_latent, class_weights=class_weights, weights_cfg=cfg.occ_weights,
        )
        losses: Dict[str, Tensor] = {}
        if self.uncertainty_head is not None and outputs.get("log_var") is not None:
            nll = uncertainty_nll(occ_logits, outputs["log_var"], target_latent)
            losses["loss_occ_ce"] = nll * cfg.uncertainty_weight
        else:
            losses["loss_occ_ce"] = occ_terms["loss_occ_ce"]
        losses["loss_occ_lovasz"] = occ_terms["loss_occ_lovasz"]
        losses["loss_occ_geo_scal"] = occ_terms["loss_occ_geo_scal"]
        losses["loss_occ_sem_scal"] = occ_terms["loss_occ_sem_scal"]

        losses["loss_flow"] = flow_loss(outputs["flow_pred"], batch["gt_flow"]) * cfg.flow_weight

        instance_states = outputs.get("instance_states")
        gt_boxes = batch.get("gt_boxes")
        if instance_states is not None and gt_boxes is not None:
            inst_terms = instance_loss(instance_states, gt_boxes)
            for k, v in inst_terms.items():
                losses[k] = v * cfg.instance_weight

            # Auxiliary present-frame detection on the extractor's own output. Without this the
            # extractor is supervised only indirectly through the rollout, leaving its objectness
            # and velocity heads without gradient (Doracamom Sec. III-G uses the same joint
            # detection auxiliary to shape the shared representation).
            present_state = outputs.get("present_state")
            if present_state is not None:
                # BUGFIX: `gt_boxes` is List[B][T_o] (batch-major -- see this module's
                # docstring and `drift.data.collate`), so the old `gt_boxes[:1]` sliced the
                # *batch* down to one element while `instance_loss` still looped over all B
                # states, raising IndexError for every batch size > 1. Every test ran B=1,
                # where the two slicings coincide, so it stayed hidden until a real
                # multi-sample run. Keep all B entries; take each one's first frame.
                present_gt = [boxes[:1] for boxes in gt_boxes]
                det_terms = instance_loss([present_state], present_gt)
                for k, v in det_terms.items():
                    losses[k.replace("loss_instance", "loss_det")] = v * cfg.detection_weight

        # Auxiliary binary occupancy on the fused latent (Doracamom Sec. III-F). Supervised at the
        # PRESENT observation frame only, since that is the one frame where the observation axis
        # (T_p, past->present) and the target axis (T_o, present->future) coincide.
        fusion_aux = outputs.get("fusion_aux") or {}
        occ_mask_logits = fusion_aux.get("occ_mask_logits")
        if occ_mask_logits is not None:
            present_logits = occ_mask_logits[:, -1]              # (B,1,X,Y,Z) present observation
            present_target = target_latent[:, 0]                 # (B,X,Y,Z)   present target
            valid = present_target != 255
            occupied = ((present_target != 0) & valid).to(dtype)
            losses["loss_aux_occ"] = _bce_dice(
                present_logits.squeeze(1), occupied, valid
            ) * cfg.aux_occ_weight

        if self.cmli is not None:
            masks = {"lidar_mask": outputs["lidar_mask"], "cam_mask": outputs["cam_mask"]}
            cmli_terms = cmli_consistency_loss(outputs["cmli_aux"], masks)
            for k, v in cmli_terms.items():
                losses[k] = v * cfg.cmli_weight

        return losses
