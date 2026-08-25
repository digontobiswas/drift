"""Cam4DOcc-protocol dataset(s) for DRIFT.

Provides two `torch.utils.data.Dataset` implementations that both emit the
exact sample contract described in docs/DESIGN_SPEC.md SS4:

- `Cam4DOccDataset`: reads pre-processed per-sample files following the
  on-disk schema documented in its class docstring. The nuScenes/Cam4DOcc raw
  -> pre-processed pipeline itself (LiDAR rasterization, instance tracking
  across sweeps, box-in-range filtering, etc.) is an offline script outside
  this module's scope; this class defines and loads the *result* of that
  pipeline.
- `SyntheticOccDataset`: generates random-but-shape-correct and semantically
  plausible samples on the fly, with no external data dependency, so the rest
  of the pipeline (collate, losses, metrics, and eventually the model) is
  fully testable in CI / on a laptop.

Both datasets share `_build_future_ego` (thin wrapper over
`drift.data.ego_motion.compose_future_transforms`) so that `future_ego` is
always derived the same way regardless of data source.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from drift.data.ego_motion import compose_future_transforms

__all__ = ["BoxSet", "Cam4DOccDataset", "SyntheticOccDataset", "CAM_PARAM_FIELDS"]

# `CameraParams` (drift.models.encoders.camera_params) validates a *batched*
# (B,T,N,...) shape on construction, so a per-sample (no-B) Dataset item cannot
# hold a real CameraParams instance. Each dataset instead returns this sample's
# "cam_params" as a plain dict of unbatched (T,N,...) tensors with these exact
# field names; `drift.data.collate.collate_fn` stacks them into a proper
# `CameraParams` (with the B dim CameraParams requires) at batch time.
CAM_PARAM_FIELDS = ("rots", "trans", "intrins", "post_rots", "post_trans")

_DEFAULT_PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_DEFAULT_OCC_SIZE = (512, 512, 40)
_DEFAULT_LATENT_SIZE = (128, 128, 10)
_LATENT_DOWNSAMPLE = 4
_VOXEL_SIZE_LATENT = 0.8  # metres, per spec SS0
_IGNORE_FLOW = 255.0
_IGNORE_OCC = 255


@dataclass
class BoxSet:
    """One frame's ground-truth object boxes, for the instance auxiliary loss.

    Attributes:
        center: `(N, 3)` metric xyz, present-frame (or that frame's own)
            ego coordinates.
        size: `(N, 3)` length, width, height in metres.
        yaw: `(N, 1)` heading, radians.
        velocity: `(N, 3)` m/s.
        label: `(N,)` int64 class index.
        track_id: `(N,)` int64 instance id, consistent across frames within
            one sample/scene, used to follow a matched query's ground-truth
            trajectory across horizons in `drift.losses.instance.instance_loss`.
    """

    center: Tensor
    size: Tensor
    yaw: Tensor
    velocity: Tensor
    label: Tensor
    track_id: Tensor

    def to(self, device: Union[str, torch.device]) -> "BoxSet":
        """Return a copy with every tensor moved to `device`."""
        return BoxSet(
            center=self.center.to(device),
            size=self.size.to(device),
            yaw=self.yaw.to(device),
            velocity=self.velocity.to(device),
            label=self.label.to(device),
            track_id=self.track_id.to(device),
        )

    @staticmethod
    def empty(dtype: torch.dtype = torch.float32) -> "BoxSet":
        """An empty `BoxSet` (no ground-truth objects in this frame)."""
        return BoxSet(
            center=torch.zeros(0, 3, dtype=dtype),
            size=torch.zeros(0, 3, dtype=dtype),
            yaw=torch.zeros(0, 1, dtype=dtype),
            velocity=torch.zeros(0, 3, dtype=dtype),
            label=torch.zeros(0, dtype=torch.long),
            track_id=torch.zeros(0, dtype=torch.long),
        )


def _build_future_ego(ego_motion: Tensor, present_idx: int, num_future: int) -> Tensor:
    """Thin wrapper over `compose_future_transforms`, shared by both datasets below."""
    return compose_future_transforms(ego_motion.unsqueeze(0), present_idx, num_future).squeeze(0)


# ImageNet statistics, applied to lazily-loaded JPEGs so that a `pretrained=True`
# ResNet backbone sees the distribution it was trained on.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# nuScenes camera images are 1600x900. The default here is the standard BEVDet /
# Cam4DOcc input crop: resize by 0.44 then crop the top away (the sky carries no
# occupancy information and the crop keeps the road surface).
NUSC_IMG_HW = (900, 1600)


class Cam4DOccDataset(Dataset):
    """Protocol-compatible dataset reading pre-processed Cam4DOcc-style samples.

    On-disk contract, produced by ``tools/prepare_nuscenes.py``: ``ann_file`` is
    a JSON list of per-sample entries, each

    ```json
    {
      "sample_token": "...",
      "npz_path":   "samples/<token>.npz",
      "points_paths": ["<abs>/LIDAR_TOP/....pcd.bin", ...],      // length T_p
      "img_paths":  [["<abs>/CAM_FRONT/....jpg", ... x6], ...]   // (T_p, N_cam)
    }
    ```

    Point clouds and images are **not** copied into the derived dataset -- the
    paths point back into the read-only raw nuScenes tree and are decoded lazily
    here. Only ground truth and calibration live in the ``.npz``.

    Two ``.npz`` layouts are accepted:

    *Sparse* (what ``tools/prepare_nuscenes.py`` writes; ~12 MB/sample):
      ``sparse_coords`` ``(M,3)`` int16, ``sparse_occ`` ``(M,)`` uint8,
      ``sparse_instance`` ``(M,)`` uint16, ``sparse_counts`` ``(T_o,)`` int64,
      plus ``occ_size``/``latent_size``. Free voxels are omitted entirely.

    *Dense* (legacy / externally produced): ``gt_occ`` ``(T_o,X,Y,Z)`` int64,
      ``gt_instance`` same shape, ``gt_flow`` ``(T_o,3,x,y,z)`` float32.

    Both layouts additionally carry ``rots``/``trans``/``intrins``
    ``(T_p,N_cam,...)``, ``ego_motion`` ``(T_seq,4,4)``, ``lidar_mask`` /
    ``cam_mask`` ``(T_p,)``, and the per-frame ``box_*`` object arrays.
    ``post_rots``/``post_trans`` are derived from the image resize/crop applied
    here (or read from the ``.npz`` if the preprocessing baked them in).

    Args:
        data_root: Root the relative ``npz_path`` is resolved against.
        ann_file: Path to the JSON index described above.
        present_idx: Index of the present frame within ``T_p`` (default ``T_p-1``).
        in_channels: Point feature channels kept from each ``.bin``.
        img_hw: ``(H, W)`` the loaded images are resized/cropped to. Must match
            ``DataConfig.H_img`` / ``W_img``.
        point_dims_on_disk: Channels stored per point in the raw ``.bin``.
            nuScenes LiDAR sweeps store 5 (x, y, z, intensity, ring index); the
            first ``in_channels`` are kept.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        present_idx: Optional[int] = None,
        in_channels: int = 4,
        img_hw: Tuple[int, int] = (256, 704),
        point_dims_on_disk: int = 5,
    ) -> None:
        self.data_root = Path(data_root)
        ann_path = Path(ann_file)
        if not ann_path.is_absolute():
            ann_path = self.data_root / ann_path
        if not ann_path.exists():
            raise ValueError(
                f"Cam4DOccDataset: annotation file not found: {ann_path}. Build it first with\n"
                f"  python tools/prepare_nuscenes.py --nuscenes-root <raw> --out-root {self.data_root} --split ..."
            )
        with open(ann_path, "r") as f:
            self._index: List[Dict[str, Any]] = json.load(f)
        if not self._index:
            raise ValueError(f"Cam4DOccDataset: annotation file {ann_path} is empty.")
        self.present_idx = present_idx
        self.in_channels = in_channels
        self.img_hw = (int(img_hw[0]), int(img_hw[1]))
        self.point_dims_on_disk = int(point_dims_on_disk)

    def __len__(self) -> int:
        return len(self._index)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.data_root / path

    # -- image loading ------------------------------------------------------

    def _load_image(self, path: str) -> Tuple[Tensor, float, Tuple[int, int]]:
        """Decode one JPEG -> `(3,H,W)` normalized tensor, plus its resize/crop params."""
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Pillow is required to load nuScenes camera images. Install it on the login node: "
                "pip install Pillow"
            ) from exc

        H_out, W_out = self.img_hw
        with Image.open(self._resolve(path)) as im:
            im = im.convert("RGB")
            W_src, H_src = im.size
            # Resize so width matches exactly, then crop the *top* off to height.
            scale = W_out / W_src
            H_res = int(round(H_src * scale))
            im = im.resize((W_out, H_res), Image.BILINEAR)
            crop_top = max(0, H_res - H_out)
            im = im.crop((0, crop_top, W_out, crop_top + H_out))
            arr = np.asarray(im, dtype=np.float32) / 255.0

        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return tensor, scale, (crop_top, 0)

    @staticmethod
    def _post_transform(scale: float, crop: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """The 3x3 / 3-vector image-plane augmentation `CameraParams` expects.

        A pixel `u` in the *original* image maps to `post_rot @ u + post_trans`
        in the loaded tensor, so the projection code can undo the resize/crop.
        """
        post_rot = np.eye(3, dtype=np.float32) * scale
        post_rot[2, 2] = 1.0
        crop_top, crop_left = crop
        post_trans = np.array([-float(crop_left), -float(crop_top), 0.0], dtype=np.float32)
        return post_rot, post_trans

    # -- ground-truth decoding ---------------------------------------------

    @staticmethod
    def _decode_sparse(npz: Any) -> Tuple[Tensor, Tensor]:
        """Sparse `(coords, occ, instance, counts)` -> dense `(T_o,X,Y,Z)` occ / instance."""
        occ_size = tuple(int(v) for v in npz["occ_size"])
        counts = np.asarray(npz["sparse_counts"], dtype=np.int64)
        coords = np.asarray(npz["sparse_coords"], dtype=np.int64)
        vals_occ = np.asarray(npz["sparse_occ"], dtype=np.int64)
        vals_inst = np.asarray(npz["sparse_instance"], dtype=np.int64)

        T_o = int(counts.shape[0])
        occ = torch.zeros((T_o, *occ_size), dtype=torch.long)
        inst = torch.zeros((T_o, *occ_size), dtype=torch.long)
        offset = 0
        for t in range(T_o):
            n = int(counts[t])
            if n == 0:
                continue
            c = coords[offset : offset + n]
            occ[t, c[:, 0], c[:, 1], c[:, 2]] = torch.from_numpy(vals_occ[offset : offset + n])
            inst[t, c[:, 0], c[:, 1], c[:, 2]] = torch.from_numpy(vals_inst[offset : offset + n])
            offset += n
        return occ, inst

    @staticmethod
    def _decode_boxes(npz: Any, T_o: int) -> List[BoxSet]:
        gt_boxes: List[BoxSet] = []
        centers, sizes, yaws, vels, labels, tids = (
            npz["box_center"], npz["box_size"], npz["box_yaw"],
            npz["box_velocity"], npz["box_label"], npz["box_track_id"],
        )
        for t in range(T_o):
            c = np.asarray(centers[t], dtype=np.float32).reshape(-1, 3)
            if c.shape[0] == 0:
                gt_boxes.append(BoxSet.empty())
                continue
            gt_boxes.append(
                BoxSet(
                    center=torch.from_numpy(c),
                    size=torch.from_numpy(np.asarray(sizes[t], dtype=np.float32).reshape(-1, 3)),
                    yaw=torch.from_numpy(np.asarray(yaws[t], dtype=np.float32).reshape(-1, 1)),
                    velocity=torch.from_numpy(np.asarray(vels[t], dtype=np.float32).reshape(-1, 3)),
                    label=torch.from_numpy(np.asarray(labels[t], dtype=np.int64).reshape(-1)),
                    track_id=torch.from_numpy(np.asarray(tids[t], dtype=np.int64).reshape(-1)),
                )
            )
        return gt_boxes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self._index[idx]
        npz_path = self._resolve(entry["npz_path"])
        if not npz_path.exists():
            raise ValueError(
                f"Cam4DOccDataset[{idx}]: missing pre-processed file {npz_path}. Run "
                "tools/prepare_nuscenes.py first (it is resumable, so a partial run can be continued)."
            )

        with np.load(npz_path, allow_pickle=True) as npz:
            keys = set(npz.files)

            ego_motion = torch.from_numpy(npz["ego_motion"]).float()
            gt_flow = torch.from_numpy(np.asarray(npz["gt_flow"], dtype=np.float32)).float()
            lidar_mask = torch.from_numpy(npz["lidar_mask"]).float()
            cam_mask = torch.from_numpy(npz["cam_mask"]).float()

            if "sparse_counts" in keys:
                gt_occ, gt_instance = self._decode_sparse(npz)
            else:
                gt_occ = torch.from_numpy(npz["gt_occ"]).long()
                gt_instance = torch.from_numpy(npz["gt_instance"]).long()
            T_o = gt_occ.shape[0]

            rots = torch.from_numpy(npz["rots"]).float()
            trans = torch.from_numpy(npz["trans"]).float()
            intrins = torch.from_numpy(npz["intrins"]).float()
            baked_post = "post_rots" in keys and "post_trans" in keys
            if baked_post:
                post_rots = torch.from_numpy(npz["post_rots"]).float()
                post_trans = torch.from_numpy(npz["post_trans"]).float()

            gt_boxes = self._decode_boxes(npz, T_o)

            # Images: either baked into the .npz (legacy) or lazily decoded from JPEG.
            if "imgs" in keys:
                imgs = torch.from_numpy(npz["imgs"]).float()
                if not baked_post:
                    T_p_, N_ = imgs.shape[0], imgs.shape[1]
                    post_rots = torch.eye(3).expand(T_p_, N_, 3, 3).clone()
                    post_trans = torch.zeros(T_p_, N_, 3)
                imgs_from_disk = False
            else:
                imgs_from_disk = True

        if imgs_from_disk:
            if "img_paths" not in entry:
                raise ValueError(
                    f"Cam4DOccDataset[{idx}]: {npz_path} has no 'imgs' array and the annotation "
                    "entry has no 'img_paths'. Rebuild the index with tools/prepare_nuscenes.py."
                )
            frames, pr_frames, pt_frames = [], [], []
            for cam_paths in entry["img_paths"]:
                views, prs, pts_ = [], [], []
                for p in cam_paths:
                    img, scale, crop = self._load_image(p)
                    pr, pt = self._post_transform(scale, crop)
                    views.append(img)
                    prs.append(torch.from_numpy(pr))
                    pts_.append(torch.from_numpy(pt))
                frames.append(torch.stack(views))
                pr_frames.append(torch.stack(prs))
                pt_frames.append(torch.stack(pts_))
            imgs = torch.stack(frames)          # (T_p, N_cam, 3, H, W)
            post_rots = torch.stack(pr_frames)  # (T_p, N_cam, 3, 3)
            post_trans = torch.stack(pt_frames)  # (T_p, N_cam, 3)

        cam_params = {
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "post_rots": post_rots,
            "post_trans": post_trans,
        }

        T_p = imgs.shape[0]
        present_idx = self.present_idx if self.present_idx is not None else T_p - 1

        points: List[Tensor] = []
        for p in entry["points_paths"]:
            arr = np.fromfile(self._resolve(p), dtype=np.float32)
            d = self.point_dims_on_disk
            if arr.size % d != 0:
                # Fall back to in_channels for .bin files written by other pipelines.
                d = self.in_channels
            arr = arr.reshape(-1, d)[:, : self.in_channels]
            points.append(torch.from_numpy(np.ascontiguousarray(arr)))

        future_ego = _build_future_ego(ego_motion, present_idx, T_o)

        return {
            "imgs": imgs,
            "cam_params": cam_params,
            "points": points,
            "ego_motion": ego_motion,
            "future_ego": future_ego,
            "gt_occ": gt_occ,
            "gt_flow": gt_flow,
            "gt_instance": gt_instance,
            "gt_boxes": gt_boxes,
            "lidar_mask": lidar_mask,
            "cam_mask": cam_mask,
        }


class SyntheticOccDataset(Dataset):
    """Deterministic synthetic dataset, shape- and protocol-compatible with `Cam4DOccDataset`.

    Every sample is generated from a per-index RNG seed (so `__getitem__` is a
    pure function of `idx`, safe under multi-worker `DataLoader` shuffling).
    Occupancy/instance/flow ground truth is generated *at latent resolution*
    first -- a "ground" class filling the lowest z-layers plus a handful of
    randomly placed, rigidly-moving 3D boxes with class labels and track ids
    -- and only then nearest-upsampled by `LATENT_DOWNSAMPLE` to the
    full-resolution grid, so `downsample_target` applied to `gt_occ` exactly
    round-trips back to the latent grid used to generate it (a strong internal
    consistency property, not just "random noise with the right shape").

    Tracks are only ever introduced at the present/first frame (never spawn
    mid-sequence, matching "instances first appearing in the future are
    excluded"), and a track that would leave `point_cloud_range` at some
    future frame is simply omitted from that frame's rasterization and
    `BoxSet` (matching "objects whose OBB is not fully inside pc_range are
    dropped for that frame").

    Args:
        num_samples: Dataset length.
        T_p: Number of past+present input frames.
        T_f: Number of future frames the protocol evaluates (informational;
            does not affect generation beyond documenting intent).
        T_o: Number of output frames the model is expected to emit.
        N_cam: Number of camera views.
        H_img, W_img: Image resolution.
        num_classes: Occupancy class count (0 = free).
        latent_size: `(X, Y, Z)` latent grid.
        occ_size: `(X, Y, Z)` full-resolution grid; must equal
            `latent_size * LATENT_DOWNSAMPLE` per axis.
        point_cloud_range: Metric extent shared by every frame.
        in_channels: Point feature channels (xyz + intensity by default).
        num_points_range: `(min, max)` random point count per frame.
        num_boxes_range: `(min, max)` random dynamic-object count per sample.
        modality_dropout_p: Per-frame probability of a modality being marked
            absent in `lidar_mask` / `cam_mask` (independent Bernoulli); `0.0`
            disables synthetic dropout.
        seed: Base seed; sample `idx` uses `seed + idx`.
    """

    def __init__(
        self,
        num_samples: int = 32,
        T_p: int = 3,
        T_f: int = 4,
        T_o: int = 6,
        N_cam: int = 6,
        H_img: int = 64,
        W_img: int = 96,
        num_classes: int = 5,
        latent_size: Tuple[int, int, int] = _DEFAULT_LATENT_SIZE,
        occ_size: Tuple[int, int, int] = _DEFAULT_OCC_SIZE,
        point_cloud_range: Sequence[float] = _DEFAULT_PC_RANGE,
        in_channels: int = 4,
        num_points_range: Tuple[int, int] = (300, 900),
        num_boxes_range: Tuple[int, int] = (0, 6),
        modality_dropout_p: float = 0.0,
        seed: int = 0,
    ) -> None:
        ratio = _LATENT_DOWNSAMPLE
        expected_occ = tuple(s * ratio for s in latent_size)
        if tuple(occ_size) != expected_occ:
            raise ValueError(
                f"occ_size {tuple(occ_size)} must equal latent_size*{ratio} = {expected_occ}"
            )
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2 (0=free, 1=ground/static, ...), got {num_classes}")
        if T_o < 1:
            raise ValueError(f"T_o must be >= 1, got {T_o}")

        self.num_samples = num_samples
        self.T_p = T_p
        self.T_f = T_f
        self.T_o = T_o
        self.N_cam = N_cam
        self.H_img = H_img
        self.W_img = W_img
        self.num_classes = num_classes
        self.latent_size = tuple(latent_size)
        self.occ_size = tuple(occ_size)
        self.point_cloud_range = list(point_cloud_range)
        self.in_channels = in_channels
        self.num_points_range = num_points_range
        self.num_boxes_range = num_boxes_range
        self.modality_dropout_p = modality_dropout_p
        self.seed = seed
        self.present_idx = T_p - 1
        self.T_seq = T_p - 1 + T_o  # exact span needed by both warp-to-present and future compose

    def __len__(self) -> int:
        return self.num_samples

    # -- generation helpers -------------------------------------------------

    def _voxel_size(self) -> Tuple[float, float, float]:
        X, Y, Z = self.latent_size
        x0, y0, z0, x1, y1, z1 = self.point_cloud_range
        return (x1 - x0) / X, (y1 - y0) / Y, (z1 - z0) / Z

    def _latent_to_metric(self, idx_xyz: np.ndarray) -> np.ndarray:
        """`(N,3)` continuous latent voxel index -> `(N,3)` metric xyz (voxel centers)."""
        vx, vy, vz = self._voxel_size()
        x0, y0, z0 = self.point_cloud_range[:3]
        origin = np.array([x0, y0, z0], dtype=np.float32)
        voxel = np.array([vx, vy, vz], dtype=np.float32)
        return origin + (idx_xyz + 0.5) * voxel

    def _make_ego_motion(self, rng: np.random.Generator) -> Tensor:
        """`(T_seq,4,4)` mild forward-driving ego motion with small random yaw rate."""
        mats = []
        for _ in range(self.T_seq):
            yaw = float(rng.normal(0.0, 0.03))
            fwd = float(rng.uniform(0.5, 2.0))  # metres/step forward drive
            lat = float(rng.normal(0.0, 0.05))
            c, s = math.cos(yaw), math.sin(yaw)
            m = np.eye(4, dtype=np.float32)
            m[:2, :2] = [[c, -s], [s, c]]
            m[0, 3] = fwd
            m[1, 3] = lat
            mats.append(m)
        return torch.from_numpy(np.stack(mats, axis=0))

    def _make_tracks(self, rng: np.random.Generator) -> List[Dict[str, np.ndarray]]:
        """Per-track constant-velocity trajectories in latent-voxel index space, defined for t=-1..T_o-1."""
        X, Y, Z = self.latent_size
        lo, hi = self.num_boxes_range
        n_boxes = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        vx, vy, vz = self._voxel_size()
        tracks = []
        for k in range(n_boxes):
            start = np.array(
                [rng.uniform(X * 0.15, X * 0.85), rng.uniform(Y * 0.15, Y * 0.85), rng.uniform(Z * 0.2, Z * 0.6)],
                dtype=np.float32,
            )
            # velocity in latent voxels/step, capped to a plausible metric speed (<= ~10 m/s)
            speed_mps = rng.uniform(0.0, 8.0)
            heading = rng.uniform(0, 2 * math.pi)
            vel_metric = np.array([speed_mps * math.cos(heading), speed_mps * math.sin(heading), 0.0], dtype=np.float32)
            vel_voxel = vel_metric * np.array([1.0 / vx, 1.0 / vy, 1.0 / vz], dtype=np.float32) * 0.5  # 0.5s/step
            extent = np.array(
                [rng.uniform(1.5, 5.0), rng.uniform(1.2, 3.0), rng.uniform(1.2, 2.5)], dtype=np.float32
            )
            label = int(rng.integers(2, self.num_classes))  # classes >=2 are "dynamic"
            yaw = float(rng.uniform(-math.pi, math.pi))
            tracks.append(
                {
                    "track_id": k + 1,
                    "start": start,  # position at t=-1 (present)
                    "vel_voxel": vel_voxel,
                    "extent_voxel": extent,
                    "label": label,
                    "yaw": yaw,
                }
            )
        return tracks

    def _rasterize(
        self, tracks: List[Dict[str, np.ndarray]], t: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, BoxSet]:
        """Rasterize all in-range tracks at future-horizon index `t` (0-based) into latent grids.

        Returns:
            occ_latent: `(X,Y,Z)` int64
            inst_latent: `(X,Y,Z)` int64 (0 = background)
            flow_latent: `(3,X,Y,Z)` float32, `_IGNORE_FLOW` = ignore
            boxes: `BoxSet` for this frame
        """
        X, Y, Z = self.latent_size
        occ = np.zeros((X, Y, Z), dtype=np.int64)
        inst = np.zeros((X, Y, Z), dtype=np.int64)
        flow = np.full((3, X, Y, Z), _IGNORE_FLOW, dtype=np.float32)

        # Static "ground/structure" layer occupying the lowest latent z-slice everywhere.
        occ[:, :, 0] = 1

        centers, sizes, yaws, vels, labels, tids = [], [], [], [], [], []
        vx, vy, vz = self._voxel_size()

        for tr in tracks:
            pos_t = tr["start"] + tr["vel_voxel"] * (t + 1)  # t+1 steps forward of "present" (t=-1)
            pos_prev = tr["start"] + tr["vel_voxel"] * t  # previous frame's centroid (t-1 in spec notation)
            half = tr["extent_voxel"] / 2.0
            lo = pos_t - half
            hi = pos_t + half
            if lo[0] < 0 or lo[1] < 0 or lo[2] < 0 or hi[0] > X or hi[1] > Y or hi[2] > Z:
                continue  # OBB not fully inside range at this frame -> dropped

            ix0, iy0, iz0 = np.floor(lo).astype(int).clip(0, [X - 1, Y - 1, Z - 1])
            ix1, iy1, iz1 = np.ceil(hi).astype(int).clip(1, [X, Y, Z])
            occ[ix0:ix1, iy0:iy1, iz0:iz1] = tr["label"]
            inst[ix0:ix1, iy0:iy1, iz0:iz1] = tr["track_id"]

            # Backward centroid-offset flow: centroid(t-1) - own_index(t), per voxel of this
            # instance, in latent-voxel units.
            xs, ys, zs = np.meshgrid(
                np.arange(ix0, ix1), np.arange(iy0, iy1), np.arange(iz0, iz1), indexing="ij"
            )
            own_idx = np.stack([xs, ys, zs], axis=0).astype(np.float32) + 0.5  # voxel centers
            offset = pos_prev.reshape(3, 1, 1, 1) - own_idx
            flow[:, ix0:ix1, iy0:iy1, iz0:iz1] = offset

            centers.append(self._latent_to_metric(pos_t.reshape(1, 3))[0])
            sizes.append(np.array([tr["extent_voxel"][0] * vx, tr["extent_voxel"][1] * vy, tr["extent_voxel"][2] * vz], dtype=np.float32))
            yaws.append(tr["yaw"])
            vels.append(np.concatenate([tr["vel_voxel"][:2] * [vx, vy] / 0.5, [0.0]]).astype(np.float32))
            labels.append(tr["label"])
            tids.append(tr["track_id"])

        if centers:
            boxes = BoxSet(
                center=torch.from_numpy(np.stack(centers).astype(np.float32)),
                size=torch.from_numpy(np.stack(sizes).astype(np.float32)),
                yaw=torch.from_numpy(np.array(yaws, dtype=np.float32).reshape(-1, 1)),
                velocity=torch.from_numpy(np.stack(vels).astype(np.float32)),
                label=torch.from_numpy(np.array(labels, dtype=np.int64)),
                track_id=torch.from_numpy(np.array(tids, dtype=np.int64)),
            )
        else:
            boxes = BoxSet.empty()

        return occ, inst, flow, boxes

    def _upsample_occ(self, latent: np.ndarray) -> Tensor:
        t = torch.from_numpy(latent)
        r = _LATENT_DOWNSAMPLE
        return t.repeat_interleave(r, dim=0).repeat_interleave(r, dim=1).repeat_interleave(r, dim=2)

    def _make_points(self, rng: np.random.Generator, occ_latent: np.ndarray) -> Tensor:
        """A ragged point cloud: background sprinkle + clustered points near occupied voxels."""
        n_pts = int(rng.integers(*self.num_points_range))
        x0, y0, z0, x1, y1, z1 = self.point_cloud_range
        n_bg = n_pts // 2
        bg = rng.uniform([x0, y0, z0], [x1, y1, z1], size=(n_bg, 3)).astype(np.float32)

        occ_xyz = np.argwhere(occ_latent > 0)
        n_fg = n_pts - n_bg
        if occ_xyz.shape[0] > 0:
            picks = occ_xyz[rng.integers(0, occ_xyz.shape[0], size=n_fg)]
            centers = self._latent_to_metric(picks.astype(np.float32))
            vx, vy, vz = self._voxel_size()
            jitter = rng.normal(0, [vx, vy, vz], size=(n_fg, 3)).astype(np.float32)
            fg = centers + jitter
        else:
            fg = rng.uniform([x0, y0, z0], [x1, y1, z1], size=(n_fg, 3)).astype(np.float32)

        xyz = np.concatenate([bg, fg], axis=0)
        intensity = rng.uniform(0, 1, size=(n_pts, 1)).astype(np.float32)
        extra = self.in_channels - 4
        feats = [xyz, intensity]
        if extra > 0:
            feats.append(rng.normal(0, 1, size=(n_pts, extra)).astype(np.float32))
        pts = np.concatenate(feats, axis=1)[:, : self.in_channels]
        return torch.from_numpy(pts)

    def _make_cam_params(self, rng: np.random.Generator) -> Dict[str, Tensor]:
        T_p, N = self.T_p, self.N_cam
        rots = np.tile(np.eye(3, dtype=np.float32), (T_p, N, 1, 1))
        # small per-camera yaw spread around the vehicle, roughly evenly spaced
        for n in range(N):
            yaw = 2 * math.pi * n / N
            c, s = math.cos(yaw), math.sin(yaw)
            rots[:, n] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        trans = np.zeros((T_p, N, 3), dtype=np.float32)
        trans[..., 2] = 1.5  # camera mount height
        intrins = np.tile(np.eye(3, dtype=np.float32), (T_p, N, 1, 1))
        f = 0.7 * max(self.H_img, self.W_img)
        intrins[..., 0, 0] = f
        intrins[..., 1, 1] = f
        intrins[..., 0, 2] = self.W_img / 2.0
        intrins[..., 1, 2] = self.H_img / 2.0
        post_rots = np.tile(np.eye(3, dtype=np.float32), (T_p, N, 1, 1))
        post_trans = np.zeros((T_p, N, 3), dtype=np.float32)
        return {
            "rots": torch.from_numpy(rots),
            "trans": torch.from_numpy(trans),
            "intrins": torch.from_numpy(intrins),
            "post_rots": torch.from_numpy(post_rots),
            "post_trans": torch.from_numpy(post_trans),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"index {idx} out of range for dataset of length {self.num_samples}")
        rng = np.random.default_rng(self.seed + idx)

        imgs = torch.from_numpy(
            rng.normal(0.5, 0.15, size=(self.T_p, self.N_cam, 3, self.H_img, self.W_img)).clip(0, 1).astype(np.float32)
        )
        cam_params = self._make_cam_params(rng)

        ego_motion = self._make_ego_motion(rng)
        future_ego = _build_future_ego(ego_motion, self.present_idx, self.T_o)

        tracks = self._make_tracks(rng)

        occ_latents, inst_latents, flow_latents, boxes_seq = [], [], [], []
        for t in range(self.T_o):
            occ_l, inst_l, flow_l, boxes = self._rasterize(tracks, t)
            occ_latents.append(occ_l)
            inst_latents.append(inst_l)
            flow_latents.append(flow_l)
            boxes_seq.append(boxes)

        occ_latent_stack = np.stack(occ_latents, axis=0)  # (T_o,X,Y,Z)
        inst_latent_stack = np.stack(inst_latents, axis=0)
        gt_flow = torch.from_numpy(np.stack(flow_latents, axis=0))  # (T_o,3,X,Y,Z)

        gt_occ = torch.stack([self._upsample_occ(occ_latent_stack[t]) for t in range(self.T_o)], dim=0).long()
        gt_instance = torch.stack([self._upsample_occ(inst_latent_stack[t]) for t in range(self.T_o)], dim=0).long()

        # Points: generated from the *first future* frame's occupancy for the present frame,
        # and reuse each past frame's own (independently sampled but similarly-shaped) scene.
        points: List[Tensor] = []
        for t in range(self.T_p):
            points.append(self._make_points(rng, occ_latent_stack[0]))

        if self.modality_dropout_p > 0:
            lidar_mask = (rng.uniform(size=self.T_p) >= self.modality_dropout_p).astype(np.float32)
            cam_mask = (rng.uniform(size=self.T_p) >= self.modality_dropout_p).astype(np.float32)
        else:
            lidar_mask = np.ones(self.T_p, dtype=np.float32)
            cam_mask = np.ones(self.T_p, dtype=np.float32)

        return {
            "imgs": imgs,
            "cam_params": cam_params,
            "points": points,
            "ego_motion": ego_motion,
            "future_ego": future_ego,
            "gt_occ": gt_occ,
            "gt_flow": gt_flow,
            "gt_instance": gt_instance,
            "gt_boxes": boxes_seq,
            "lidar_mask": torch.from_numpy(lidar_mask),
            "cam_mask": torch.from_numpy(cam_mask),
        }
