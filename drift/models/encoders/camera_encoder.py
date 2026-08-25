"""Multi-view camera image encoder with an LSS-style depth-splat lift.

REUSED (algorithm) from Lift-Splat-Shoot (Philion & Fischer, ECCV 2020) and its adoption
in BEVDet-style detectors (Huang et al., 2021) and OccProphet (ICLR 2025); reimplemented
clean-room for DRIFT with no copied code. See ``docs/DESIGN_SPEC.md`` §2.1.

Pipeline: ResNet(+FPN-lite) backbone -> per-pixel depth distribution + context features ->
outer product (depth x context) -> "frustum" point cloud in the ego/LiDAR frame via the
camera extrinsics/intrinsics and image augmentation -> voxelize into the shared latent
grid via ``scatter_add_``.

Simplifications relative to a production LSS/BEVPool implementation (noted per
§7 "mark simplifications"):
  * The FPN is a single lateral fusion of two ResNet stages (not a full multi-level FPN).
  * The frustum -> voxel splat uses a plain ``scatter_add_`` rather than the
    cumulative-sum "BEVPool" trick; this is O(N log N) sort-free but is not the fastest
    possible implementation. Marked ``# TODO(perf)`` at the call site.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from drift.models.encoders.camera_params import CameraParams

__all__ = ["CameraEncoder"]


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Build a GroupNorm with the largest group count <= max_groups that divides channels.

    GroupNorm is used instead of BatchNorm throughout the encoders so the modules behave
    identically at batch size 1 (common in this research codebase's smoke tests).
    """
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _SimpleConvBackbone(nn.Module):
    """Torchvision-free fallback backbone.

    Produces two feature maps at strides 8 and 16 (mirroring resnet50's layer2/layer3
    outputs) so :class:`CameraEncoder` works identically whether or not torchvision is
    installed. Used only when torchvision cannot be imported.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            _group_norm(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            _group_norm(64),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            _group_norm(128),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            _group_norm(256),
            nn.ReLU(inplace=True),
        )
        self.out_channels_stage2 = 128
        self.out_channels_stage3 = 256

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        feat2 = self.stage2(x)  # stride 8
        feat3 = self.stage3(feat2)  # stride 16
        return feat2, feat3


class _ResNetBackbone(nn.Module):
    """Torchvision ResNet backbone truncated to layer2/layer3 (strides 8 and 16)."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm  # local import: see module-level guard note

        builder = getattr(tvm, name, None)
        if builder is None:
            raise ValueError(
                f"Unknown torchvision backbone '{name}'. Expected e.g. 'resnet50', "
                "'resnet34'."
            )
        net = None
        if pretrained:
            try:
                # Use torchvision's own registry to resolve the weights enum for
                # `name` instead of guessing its class-name capitalization (e.g.
                # "resnet50" -> `ResNet50_Weights`, not the naive "Resnet50_Weights").
                weights = tvm.get_model_weights(name).DEFAULT
                net = builder(weights=weights)
            except Exception as exc:  # offline / no internet / weights unavailable
                warnings.warn(
                    f"Could not load pretrained weights for '{name}' ({exc!r}); "
                    "falling back to randomly initialized weights."
                )
                net = None
        if net is None:
            net = builder(weights=None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.out_channels_stage2 = self.layer2[-1].conv3.out_channels if hasattr(
            self.layer2[-1], "conv3"
        ) else self.layer2[-1].conv2.out_channels
        self.out_channels_stage3 = self.layer3[-1].conv3.out_channels if hasattr(
            self.layer3[-1], "conv3"
        ) else self.layer3[-1].conv2.out_channels

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)  # stride 8
        feat3 = self.layer3(feat2)  # stride 16
        return feat2, feat3


def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int, int]:
    """Build the image backbone, falling back to a plain CNN if torchvision is absent."""
    try:
        backbone = _ResNetBackbone(name=name, pretrained=pretrained)
    except ImportError:
        warnings.warn(
            "torchvision is not installed; CameraEncoder is using a small fallback "
            "conv backbone instead of a real ResNet. Install torchvision for the "
            "intended backbone."
        )
        backbone = _SimpleConvBackbone()
    return backbone, backbone.out_channels_stage2, backbone.out_channels_stage3


class CameraEncoder(nn.Module):
    """Multi-view images -> latent voxel volume via LSS depth-splat lift.

    REUSED (algorithm) from OccProphet / BEVDet-style LSS lifting; see module docstring.

    Args:
        backbone: torchvision backbone name (e.g. ``"resnet50"``). Ignored (a small
            fallback CNN is used) when torchvision is not importable.
        out_channels: Channels of the lifted voxel feature.
        latent_size: Target latent voxel grid ``(X, Y, Z)``.
        point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in metres.
        depth_bins: Number of discrete depth bins for the LSS depth distribution.
        pretrained: Whether to attempt loading pretrained ImageNet weights. Silently
            falls back to random init if unavailable (e.g. offline).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        out_channels: int = 64,
        latent_size: Tuple[int, int, int] = (128, 128, 10),
        point_cloud_range: List[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        depth_bins: int = 112,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if len(latent_size) != 3:
            raise ValueError(f"latent_size must be (X, Y, Z), got {latent_size!r}.")
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"point_cloud_range must have 6 elements, got {point_cloud_range!r}."
            )
        if depth_bins < 1:
            raise ValueError(f"depth_bins must be >= 1, got {depth_bins}.")

        self.out_channels = out_channels
        self.latent_size = tuple(int(v) for v in latent_size)
        self.point_cloud_range = [float(v) for v in point_cloud_range]
        self.depth_bins = depth_bins

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        X, Y, Z = self.latent_size
        self.voxel_size = (
            (x_max - x_min) / X,
            (y_max - y_min) / Y,
            (z_max - z_min) / Z,
        )
        # Depth bin bounds: cover roughly the diagonal extent of the point cloud range.
        self.d_min = 1.0
        self.d_max = max(8.0, math.hypot(x_max, y_max))

        self.backbone, c2, c3 = _build_backbone(backbone, pretrained)
        neck_channels = max(64, out_channels)
        self.lateral3 = nn.Conv2d(c3, neck_channels, 1)
        self.lateral2 = nn.Conv2d(c2, neck_channels, 1)
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            _group_norm(neck_channels),
            nn.ReLU(inplace=True),
        )
        # Depth-distribution + context head, LSS-style: predict depth_bins logits and
        # out_channels of context per pixel in one conv, then take the outer product.
        self.depth_context_head = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            _group_norm(neck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_channels, depth_bins + out_channels, 1),
        )

    def _frustum(
        self, h_feat: int, w_feat: int, h_img: int, w_img: int, device: torch.device
    ) -> Tensor:
        """Build the (unprojected) image-space frustum grid, shared across all cameras.

        Returns:
            ``(D, H_feat, W_feat, 3)`` tensor of ``(u, v, depth)`` in *original* pixel
            coordinates (i.e. before post-augmentation), to be multiplied by depth and
            un-augmented in :meth:`_lift_points`.
        """
        d_coords = torch.linspace(self.d_min, self.d_max, self.depth_bins, device=device)
        d_coords = d_coords.view(-1, 1, 1).expand(-1, h_feat, w_feat)
        u_coords = torch.linspace(0, w_img - 1, w_feat, device=device)
        u_coords = u_coords.view(1, 1, -1).expand(self.depth_bins, h_feat, -1)
        v_coords = torch.linspace(0, h_img - 1, h_feat, device=device)
        v_coords = v_coords.view(1, -1, 1).expand(self.depth_bins, -1, w_feat)
        return torch.stack([u_coords, v_coords, d_coords], dim=-1)

    def _lift_points(
        self, frustum: Tensor, cam_params: CameraParams
    ) -> Tensor:
        """Un-project the frustum to ego/LiDAR-frame 3D points for every camera/frame.

        Args:
            frustum: ``(D, H_feat, W_feat, 3)``.
            cam_params: calibration for the current ``(B, T)`` slice, each field
                ``(B, T, N, ...)``.

        Returns:
            ``(B, T, N, D, H_feat, W_feat, 3)`` points in ego/LiDAR coordinates.
        """
        B, T, N = cam_params.rots.shape[:3]
        D, Hf, Wf, _ = frustum.shape
        pts = frustum.view(1, 1, 1, D, Hf, Wf, 3).expand(B, T, N, D, Hf, Wf, 3)

        post_trans = cam_params.post_trans.view(B, T, N, 1, 1, 1, 3)
        post_rots_inv = torch.inverse(cam_params.post_rots).view(B, T, N, 1, 1, 1, 3, 3)
        pts = pts - post_trans
        pts = (post_rots_inv @ pts.unsqueeze(-1)).squeeze(-1)

        # Multiply (u, v) by depth to homogenize; leave depth channel as-is.
        pts = torch.cat([pts[..., :2] * pts[..., 2:3], pts[..., 2:3]], dim=-1)

        intrins_inv = torch.inverse(cam_params.intrins).view(B, T, N, 1, 1, 1, 3, 3)
        rots = cam_params.rots.view(B, T, N, 1, 1, 1, 3, 3)
        combine = rots @ intrins_inv
        pts = (combine @ pts.unsqueeze(-1)).squeeze(-1)
        pts = pts + cam_params.trans.view(B, T, N, 1, 1, 1, 3)
        return pts

    def _voxelize(self, points: Tensor, volume: Tensor) -> Tensor:
        """Splat per-point features into the latent voxel grid with ``scatter_add_``.

        Args:
            points: ``(B, T, N, D, H_feat, W_feat, 3)`` ego-frame xyz.
            volume: ``(B, T, N, D, H_feat, W_feat, C)`` per-point features
                (depth-prob x context outer product).

        Returns:
            ``(B, T, out_channels, X, Y, Z)`` splatted voxel grid.
        """
        X, Y, Z = self.latent_size
        x_min, y_min, z_min, _, _, _ = self.point_cloud_range
        vx, vy, vz = self.voxel_size
        B, T, N, D, Hf, Wf, _ = points.shape
        C = volume.shape[-1]

        ix = torch.floor((points[..., 0] - x_min) / vx).long()
        iy = torch.floor((points[..., 1] - y_min) / vy).long()
        iz = torch.floor((points[..., 2] - z_min) / vz).long()
        valid = (
            (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        )
        # Flatten order per spec §1: idx = (x * Y + y) * Z + z
        flat_idx = (ix.clamp(0, X - 1) * Y + iy.clamp(0, Y - 1)) * Z + iz.clamp(0, Z - 1)
        flat_idx = flat_idx.view(B, T, -1)  # (B, T, N*D*Hf*Wf)
        feats = volume.view(B, T, -1, C)
        valid = valid.view(B, T, -1, 1).float()
        feats = feats * valid

        out = torch.zeros(B, T, X * Y * Z, C, device=points.device, dtype=volume.dtype)
        idx_expand = flat_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        # TODO(perf): plain scatter_add_ over all frustum points; a sorted cumsum
        # ("BEVPool") trick would avoid materializing (N*D*Hf*Wf, C) but is not
        # required for correctness at the tiny/smoke-test config.
        out.scatter_add_(2, idx_expand, feats)
        out = out.view(B, T, X, Y, Z, C).permute(0, 1, 5, 2, 3, 4).contiguous()
        return out

    def forward(self, imgs: Tensor, cam_params: CameraParams) -> Tuple[Tensor, Tensor]:
        """Lift multi-view images into the shared latent voxel grid.

        Args:
            imgs: ``(B, T, N_cam, 3, H_img, W_img)`` normalized RGB images.
            cam_params: calibration/augmentation for the same ``(B, T, N_cam)``.

        Returns:
            Tuple of:
                voxel_feats: ``(B, T, out_channels, X, Y, Z)``.
                depth_pred: ``(B*T, N_cam, depth_bins, H_feat, W_feat)`` raw depth
                    logits (pre-softmax), exposed for an optional depth supervision loss.

        Raises:
            ValueError: On malformed input shapes or shape mismatch with ``cam_params``.
        """
        if imgs.dim() != 6:
            raise ValueError(
                f"imgs must be (B, T, N_cam, 3, H, W), got shape {tuple(imgs.shape)}."
            )
        B, T, N, C_in, H_img, W_img = imgs.shape
        if C_in != 3:
            raise ValueError(f"imgs must have 3 input channels, got {C_in}.")
        if tuple(cam_params.rots.shape[:3]) != (B, T, N):
            raise ValueError(
                f"cam_params batch/time/camera shape {tuple(cam_params.rots.shape[:3])} "
                f"does not match imgs {(B, T, N)}."
            )

        flat_imgs = imgs.reshape(B * T * N, 3, H_img, W_img)
        feat2, feat3 = self.backbone(flat_imgs)
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.lateral3(feat3_up) + self.lateral2(feat2)
        fused = self.fpn_fuse(fused)  # (B*T*N, neck, Hf, Wf)

        dc = self.depth_context_head(fused)  # (B*T*N, depth_bins + out_channels, Hf, Wf)
        depth_logits, context = dc.split([self.depth_bins, self.out_channels], dim=1)
        Hf, Wf = depth_logits.shape[-2:]
        depth_prob = depth_logits.softmax(dim=1)

        # Outer product: (B*T*N, out_channels, depth_bins, Hf, Wf)
        volume = depth_prob.unsqueeze(1) * context.unsqueeze(2)
        volume = volume.view(B, T, N, self.out_channels, self.depth_bins, Hf, Wf)
        volume = volume.permute(0, 1, 2, 4, 5, 6, 3).contiguous()  # (...,D,Hf,Wf,C)

        frustum = self._frustum(Hf, Wf, H_img, W_img, imgs.device)
        points = self._lift_points(frustum, cam_params)  # (B,T,N,D,Hf,Wf,3)

        voxel_feats = self._voxelize(points, volume)

        depth_pred = depth_logits.view(B * T, N, self.depth_bins, Hf, Wf)
        return voxel_feats, depth_pred
