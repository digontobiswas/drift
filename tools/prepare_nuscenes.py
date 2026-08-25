"""Offline preprocessing: raw nuScenes -> DRIFT/Cam4DOcc-protocol samples.

This is the script `drift.data.cam4docc_dataset.Cam4DOccDataset`'s docstring
refers to as "an offline preprocessing script outside this module's scope".
It reads a **read-only** raw nuScenes tree and writes derived ground truth to a
**separate** output directory. It never writes into the nuScenes root.

What is written (per sample, one compressed ``.npz``):
  - calibration for the T_p input frames (rots/trans/intrins/post_rots/post_trans)
  - ``ego_motion`` ``(T_seq,4,4)``, ``ego_motion[t] = T_{t+1<-t}`` in ego coords
  - ``gt_occ`` / ``gt_instance`` stored **sparsely** (shared int16 coords +
    uint8 labels + uint16 instance ids) -- a dense ``(T_o,512,512,40)`` int64
    array would be ~500 MB *per sample*.
  - ``gt_flow`` at latent resolution as float16
  - per-frame ``lidar_mask`` / ``cam_mask``
  - per-frame object boxes (center/size/yaw/velocity/label/track_id)

What is **not** written: images and LiDAR point clouds. The annotation JSON
stores absolute paths back into the read-only nuScenes tree and the dataset
loads them lazily. This keeps the derived dataset ~100x smaller.

Semantic protocol (``--protocol gmo``, the default and the one that matches the
Cam4DOcc benchmark this project compares against):
    0 = free, 1 = GSO (general static occupancy), 2 = GMO (general movable object)
``--protocol gmo`` needs no nuScenes-lidarseg download. ``--protocol lidarseg``
(17 classes) additionally requires the separate nuScenes-lidarseg archive.

Usage (login node -- compute nodes have no internet, but this needs none):

    python tools/prepare_nuscenes.py \
        --nuscenes-root /home/biswabandhurj/nuscenes-trainval \
        --out-root $SCRATCH/drift_data/cam4docc_gmo \
        --split train --workers 8

Then again with ``--split val``. Both write into ``--out-root``; the two
annotation JSONs (``drift_train.json`` / ``drift_val.json``) sit side by side.

The script is **resumable**: a sample whose ``.npz`` already exists is skipped
unless ``--overwrite`` is passed, so a job that hits its walltime can simply be
resubmitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Repo root on sys.path so `configs` / `drift` import when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base import LATENT_DOWNSAMPLE, OCC_SIZE, POINT_CLOUD_RANGE  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

CAM_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)

# nuScenes category prefixes counted as "general movable object" (GMO) by the
# Cam4DOcc protocol. `movable_object.*` (cones, barriers) are deliberately NOT
# included: they are movable in principle but static in practice, and Cam4DOcc
# scores them as static occupancy.
GMO_PREFIXES = ("vehicle.", "human.")

FREE_IDX = 0
GSO_IDX = 1
GMO_IDX = 2

IGNORE_FLOW = 255.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _quat_to_rotmat(q: Sequence[float]) -> np.ndarray:
    """nuScenes stores rotations as ``[w, x, y, z]`` quaternions."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(rotation: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    """Build the 4x4 homogeneous transform for a nuScenes pose record."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = _quat_to_rotmat(rotation)
    m[:3, 3] = np.asarray(translation, dtype=np.float64)
    return m


def _yaw_from_rotmat(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _box_corners_local(size_wlh: Sequence[float]) -> np.ndarray:
    """8 corners of an axis-aligned box centred at the origin, ``(3,8)``.

    nuScenes ``size`` is ``[width, length, height]`` = ``[y, x, z]`` extent.
    """
    w, l, h = (float(v) for v in size_wlh)
    x = l / 2.0 * np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    y = w / 2.0 * np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float64)
    z = h / 2.0 * np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    return np.stack([x, y, z], axis=0)


# ---------------------------------------------------------------------------
# Voxel grid
# ---------------------------------------------------------------------------


class VoxelGrid:
    """Ego-centric voxel grid over a fixed metric ``point_cloud_range``."""

    def __init__(self, pc_range: Sequence[float], size: Tuple[int, int, int]) -> None:
        self.pc_range = np.asarray(pc_range, dtype=np.float64)
        self.size = tuple(int(s) for s in size)
        self.origin = self.pc_range[:3]
        self.voxel = (self.pc_range[3:] - self.pc_range[:3]) / np.asarray(self.size, dtype=np.float64)

    def metric_to_index(self, pts: np.ndarray) -> np.ndarray:
        """``(N,3)`` metric xyz -> ``(N,3)`` float voxel index (may be out of range)."""
        return (pts - self.origin) / self.voxel

    def index_to_metric(self, idx: np.ndarray) -> np.ndarray:
        """``(N,3)`` voxel index -> ``(N,3)`` metric xyz at voxel centres."""
        return self.origin + (idx + 0.5) * self.voxel

    def in_range(self, idx_i: np.ndarray) -> np.ndarray:
        X, Y, Z = self.size
        return (
            (idx_i[:, 0] >= 0)
            & (idx_i[:, 0] < X)
            & (idx_i[:, 1] >= 0)
            & (idx_i[:, 1] < Y)
            & (idx_i[:, 2] >= 0)
            & (idx_i[:, 2] < Z)
        )


def _rasterize_box(grid: VoxelGrid, center: np.ndarray, size_wlh: Sequence[float], yaw: float) -> np.ndarray:
    """Voxel indices ``(M,3)`` int32 whose centres fall inside an oriented box.

    Works by taking the box's axis-aligned bounding box in index space, then
    rejecting voxel centres that fall outside the *oriented* box after rotating
    them into the box's own frame. Cheap and exact at voxel-centre resolution.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    corners = R @ _box_corners_local(size_wlh) + center.reshape(3, 1)

    lo_idx = np.floor(grid.metric_to_index(corners.min(axis=1)[None, :])[0]).astype(np.int64)
    hi_idx = np.ceil(grid.metric_to_index(corners.max(axis=1)[None, :])[0]).astype(np.int64)
    X, Y, Z = grid.size
    lo_idx = np.clip(lo_idx, [0, 0, 0], [X, Y, Z])
    hi_idx = np.clip(hi_idx, [0, 0, 0], [X, Y, Z])
    if np.any(hi_idx <= lo_idx):
        return np.zeros((0, 3), dtype=np.int32)

    ax = np.arange(lo_idx[0], hi_idx[0])
    ay = np.arange(lo_idx[1], hi_idx[1])
    az = np.arange(lo_idx[2], hi_idx[2])
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    cand = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    if cand.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)

    centres = grid.index_to_metric(cand.astype(np.float64))
    local = (centres - center.reshape(1, 3)) @ R  # R^T applied on the right
    w, l, h = (float(v) for v in size_wlh)
    inside = (
        (np.abs(local[:, 0]) <= l / 2.0)
        & (np.abs(local[:, 1]) <= w / 2.0)
        & (np.abs(local[:, 2]) <= h / 2.0)
    )
    return cand[inside].astype(np.int32)


# ---------------------------------------------------------------------------
# Per-frame ground-truth construction
# ---------------------------------------------------------------------------


class FrameGT:
    """Ground truth rasterized for a single output frame, in that frame's ego coords."""

    __slots__ = ("occ", "instance", "boxes")

    def __init__(self, occ: np.ndarray, instance: np.ndarray, boxes: Dict[str, np.ndarray]) -> None:
        self.occ = occ
        self.instance = instance
        self.boxes = boxes


def _load_lidar_points(nusc, sample: Dict[str, Any], nuscenes_root: Path) -> np.ndarray:
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    path = nuscenes_root / sd["filename"]
    pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]  # x, y, z, intensity


def _sensor_to_ego(nusc, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return _transform(cs["rotation"], cs["translation"])


def _ego_to_global(nusc, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return _transform(pose["rotation"], pose["translation"])


def _lidar_to_global(nusc, sample: Dict[str, Any]) -> np.ndarray:
    tok = sample["data"]["LIDAR_TOP"]
    return _ego_to_global(nusc, tok) @ _sensor_to_ego(nusc, tok)


def _build_frame_gt(
    nusc,
    sample: Dict[str, Any],
    nuscenes_root: Path,
    grid: VoxelGrid,
    track_ids: Dict[str, int],
    inflate_gmo: bool,
) -> FrameGT:
    """Rasterize one keyframe's occupancy / instance GT in its own LiDAR frame."""
    X, Y, Z = grid.size
    occ = np.zeros((X, Y, Z), dtype=np.uint8)
    inst = np.zeros((X, Y, Z), dtype=np.uint16)

    # --- static occupancy from the LiDAR sweep ---------------------------
    pts = _load_lidar_points(nusc, sample, nuscenes_root)[:, :3].astype(np.float64)
    idx_f = grid.metric_to_index(pts)
    idx_i = np.floor(idx_f).astype(np.int64)
    keep = grid.in_range(idx_i)
    idx_i = idx_i[keep]
    if idx_i.shape[0] > 0:
        occ[idx_i[:, 0], idx_i[:, 1], idx_i[:, 2]] = GSO_IDX

    # --- dynamic objects -------------------------------------------------
    l2g = _lidar_to_global(nusc, sample)
    g2l = np.linalg.inv(l2g)

    centers, sizes, yaws, vels, labels, tids = [], [], [], [], [], []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not ann["category_name"].startswith(GMO_PREFIXES):
            continue
        if ann.get("num_lidar_pts", 1) < 1:
            continue

        center_g = np.asarray(ann["translation"], dtype=np.float64)
        R_g = _quat_to_rotmat(ann["rotation"])
        center_l = (g2l @ np.append(center_g, 1.0))[:3]
        R_l = g2l[:3, :3] @ R_g
        yaw_l = _yaw_from_rotmat(R_l)

        # Cam4DOcc drops an object whose box is not fully inside pc_range.
        half = np.array(
            [float(ann["size"][1]) / 2.0, float(ann["size"][0]) / 2.0, float(ann["size"][2]) / 2.0]
        )
        if np.any(center_l - half < grid.pc_range[:3]) or np.any(center_l + half > grid.pc_range[3:]):
            continue

        inst_tok = ann["instance_token"]
        if inst_tok not in track_ids:
            track_ids[inst_tok] = len(track_ids) + 1  # 0 reserved for background
        tid = track_ids[inst_tok]

        if inflate_gmo:
            vox = _rasterize_box(grid, center_l, ann["size"], yaw_l)
            if vox.shape[0] > 0:
                occ[vox[:, 0], vox[:, 1], vox[:, 2]] = GMO_IDX
                inst[vox[:, 0], vox[:, 1], vox[:, 2]] = tid

        try:
            vel_g = np.asarray(nusc.box_velocity(ann_token), dtype=np.float64)
        except Exception:
            vel_g = np.array([np.nan, np.nan, np.nan])
        if not np.all(np.isfinite(vel_g)):
            vel_g = np.zeros(3, dtype=np.float64)
        vel_l = g2l[:3, :3] @ vel_g

        centers.append(center_l)
        sizes.append([float(ann["size"][1]), float(ann["size"][0]), float(ann["size"][2])])  # l,w,h
        yaws.append(yaw_l)
        vels.append(vel_l)
        labels.append(GMO_IDX)
        tids.append(tid)

    boxes = {
        "center": np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        "size": np.asarray(sizes, dtype=np.float32).reshape(-1, 3),
        "yaw": np.asarray(yaws, dtype=np.float32).reshape(-1, 1),
        "velocity": np.asarray(vels, dtype=np.float32).reshape(-1, 3),
        "label": np.asarray(labels, dtype=np.int64).reshape(-1),
        "track_id": np.asarray(tids, dtype=np.int64).reshape(-1),
    }
    return FrameGT(occ, inst, boxes)


def _build_flow(
    frames: List[FrameGT],
    latent_size: Tuple[int, int, int],
    ratio: int,
    grid_latent: VoxelGrid,
) -> np.ndarray:
    """Backward centroid-offset flow at latent resolution, ``(T_o,3,X,Y,Z)`` float16.

    For every voxel belonging to instance ``k`` at frame ``t``, the value is
    ``centroid_k(t-1) - own_latent_index(voxel)`` in latent-voxel units, matching
    ``SyntheticOccDataset``. Voxels with no known previous centroid (a track's
    first frame, or background) are ``IGNORE_FLOW``.
    """
    X, Y, Z = latent_size
    T_o = len(frames)
    flow = np.full((T_o, 3, X, Y, Z), IGNORE_FLOW, dtype=np.float32)

    # Latent-resolution instance maps (max-pool the id over each ratio^3 block).
    inst_latent = []
    for f in frames:
        i = f.instance
        i = i.reshape(X, ratio, Y, ratio, Z, ratio).max(axis=(1, 3, 5))
        inst_latent.append(i)

    prev_centroid: Dict[int, np.ndarray] = {}
    for t in range(T_o):
        il = inst_latent[t]
        ids = np.unique(il)
        ids = ids[ids > 0]
        cur_centroid: Dict[int, np.ndarray] = {}
        for k in ids:
            vox = np.argwhere(il == k).astype(np.float32) + 0.5
            cur_centroid[int(k)] = vox.mean(axis=0)
        for k in ids:
            k = int(k)
            if k not in prev_centroid:
                continue
            mask = il == k
            own = np.argwhere(mask).astype(np.float32) + 0.5
            off = prev_centroid[k].reshape(1, 3) - own
            xs, ys, zs = own[:, 0].astype(int), own[:, 1].astype(int), own[:, 2].astype(int)
            flow[t, 0, xs, ys, zs] = off[:, 0]
            flow[t, 1, xs, ys, zs] = off[:, 1]
            flow[t, 2, xs, ys, zs] = off[:, 2]
        prev_centroid = cur_centroid
    return flow.astype(np.float16)


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------


def _camera_calib(nusc, sample: Dict[str, Any], nuscenes_root: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Image paths and per-camera extrinsics/intrinsics for one keyframe.

    Extrinsics are returned as camera-to-**LiDAR** (not camera-to-ego), because
    every DRIFT grid is defined in LiDAR frame coordinates.
    """
    l2g = _lidar_to_global(nusc, sample)
    g2l = np.linalg.inv(l2g)

    paths, rots, trans, intrins = [], [], [], []
    for cam in CAM_NAMES:
        sd_tok = sample["data"][cam]
        sd = nusc.get("sample_data", sd_tok)
        paths.append(str(nuscenes_root / sd["filename"]))
        cam2global = _ego_to_global(nusc, sd_tok) @ _sensor_to_ego(nusc, sd_tok)
        cam2lidar = g2l @ cam2global
        rots.append(cam2lidar[:3, :3])
        trans.append(cam2lidar[:3, 3])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        intrins.append(np.asarray(cs["camera_intrinsic"], dtype=np.float64))

    calib = {
        "rots": np.stack(rots).astype(np.float32),
        "trans": np.stack(trans).astype(np.float32),
        "intrins": np.stack(intrins).astype(np.float32),
    }
    return paths, calib


def _sparse_encode(occ: np.ndarray, inst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense ``(X,Y,Z)`` occ/instance -> shared coords + values, dropping free voxels."""
    nz = np.argwhere(occ != FREE_IDX)
    if nz.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.int16),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint16),
        )
    coords = nz.astype(np.int16)
    return coords, occ[nz[:, 0], nz[:, 1], nz[:, 2]], inst[nz[:, 0], nz[:, 1], nz[:, 2]]


def _keyframe_chain(nusc, sample_token: str, n_prev: int, n_next: int) -> Optional[List[str]]:
    """``n_prev`` keyframes before + this one + ``n_next`` after, or None if unavailable."""
    prev_chain: List[str] = []
    tok = sample_token
    for _ in range(n_prev):
        tok = nusc.get("sample", tok)["prev"]
        if not tok:
            return None
        prev_chain.append(tok)
    prev_chain.reverse()

    next_chain: List[str] = []
    tok = sample_token
    for _ in range(n_next):
        tok = nusc.get("sample", tok)["next"]
        if not tok:
            return None
        next_chain.append(tok)
    return prev_chain + [sample_token] + next_chain


def process_sample(
    nusc,
    sample_token: str,
    nuscenes_root: Path,
    out_dir: Path,
    T_p: int,
    T_o: int,
    occ_size: Tuple[int, int, int],
    pc_range: Sequence[float],
    inflate_gmo: bool,
    overwrite: bool,
) -> Optional[Dict[str, Any]]:
    """Build one sample; returns its annotation entry, or None if unusable."""
    out_npz = out_dir / f"{sample_token}.npz"

    # T_p-1 past keyframes, the present, and T_o future keyframes.
    chain = _keyframe_chain(nusc, sample_token, T_p - 1, T_o)
    if chain is None:
        return None

    input_tokens = chain[:T_p]
    output_tokens = chain[T_p - 1 : T_p - 1 + T_o]

    points_paths = []
    for tok in input_tokens:
        s = nusc.get("sample", tok)
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        points_paths.append(str(nuscenes_root / sd["filename"]))

    img_paths: List[List[str]] = []
    rots, trans, intrins = [], [], []
    for tok in input_tokens:
        s = nusc.get("sample", tok)
        paths, calib = _camera_calib(nusc, s, nuscenes_root)
        img_paths.append(paths)
        rots.append(calib["rots"])
        trans.append(calib["trans"])
        intrins.append(calib["intrins"])

    entry = {
        "sample_token": sample_token,
        "npz_path": out_npz.name,
        "points_paths": points_paths,
        "img_paths": img_paths,
    }

    if out_npz.exists() and not overwrite:
        return entry  # resumable: GT already built

    # --- ego motion over the whole T_seq span ---------------------------
    # ego_motion[t] = T_{t+1<-t} between consecutive LiDAR frames.
    T_seq = T_p - 1 + T_o
    poses = [_lidar_to_global(nusc, nusc.get("sample", tok)) for tok in chain]
    ego_motion = np.zeros((T_seq, 4, 4), dtype=np.float32)
    for t in range(T_seq):
        if t + 1 < len(poses):
            ego_motion[t] = (np.linalg.inv(poses[t + 1]) @ poses[t]).astype(np.float32)
        else:
            ego_motion[t] = np.eye(4, dtype=np.float32)

    # --- per-output-frame ground truth ----------------------------------
    grid = VoxelGrid(pc_range, occ_size)
    latent_size = tuple(s // LATENT_DOWNSAMPLE for s in occ_size)
    grid_latent = VoxelGrid(pc_range, latent_size)

    track_ids: Dict[str, int] = {}
    frames: List[FrameGT] = []
    for tok in output_tokens:
        frames.append(
            _build_frame_gt(nusc, nusc.get("sample", tok), nuscenes_root, grid, track_ids, inflate_gmo)
        )

    coords_l, occ_l, inst_l, counts = [], [], [], []
    for f in frames:
        c, o, i = _sparse_encode(f.occ, f.instance)
        coords_l.append(c)
        occ_l.append(o)
        inst_l.append(i)
        counts.append(c.shape[0])

    gt_flow = _build_flow(frames, latent_size, LATENT_DOWNSAMPLE, grid_latent)

    payload: Dict[str, Any] = {
        "rots": np.stack(rots).astype(np.float32),
        "trans": np.stack(trans).astype(np.float32),
        "intrins": np.stack(intrins).astype(np.float32),
        "ego_motion": ego_motion,
        "occ_size": np.asarray(occ_size, dtype=np.int32),
        "latent_size": np.asarray(latent_size, dtype=np.int32),
        # Sparse GT: one concatenated coord array + per-frame counts.
        "sparse_coords": np.concatenate(coords_l, axis=0) if counts else np.zeros((0, 3), np.int16),
        "sparse_occ": np.concatenate(occ_l, axis=0) if counts else np.zeros((0,), np.uint8),
        "sparse_instance": np.concatenate(inst_l, axis=0) if counts else np.zeros((0,), np.uint16),
        "sparse_counts": np.asarray(counts, dtype=np.int64),
        "gt_flow": gt_flow,
        "lidar_mask": np.ones(T_p, dtype=np.float32),
        "cam_mask": np.ones(T_p, dtype=np.float32),
    }
    for field in ("center", "size", "yaw", "velocity", "label", "track_id"):
        payload[f"box_{field}"] = np.array([f.boxes[field] for f in frames], dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed() automatically appends .npz, so pass the stem without it.
    # If we passed "file.npz.tmp", it would create "file.npz.tmp.npz".
    # Use string concatenation to avoid with_suffix() replacing the .tmp part.
    tmp_stem = str(out_npz).replace(".npz", ".tmp")  # e.g., file.tmp (no .npz)
    np.savez_compressed(tmp_stem, **payload)  # numpy creates file.tmp.npz
    tmp = Path(tmp_stem + ".npz")  # explicitly add .npz to get file.tmp.npz
    os.replace(tmp, out_npz)
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--nuscenes-root",
        type=str,
        default=os.environ.get("NUSCENES_ROOT", ""),
        help="Read-only raw nuScenes root (contains samples/, sweeps/, v1.0-trainval/).",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default=os.environ.get("DRIFT_DATA_ROOT", ""),
        help="Writable output directory for derived GT. MUST NOT be inside --nuscenes-root.",
    )
    p.add_argument("--version", type=str, default="v1.0-trainval")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--protocol", type=str, default="gmo", choices=["gmo"],
                   help="'gmo' = Cam4DOcc 3-class (free/GSO/GMO); needs no lidarseg download.")
    p.add_argument("--T-p", type=int, default=3, help="Past+present input frames.")
    p.add_argument("--T-o", type=int, default=6, help="Output frames (>= T_f+1).")
    p.add_argument("--occ-size", type=int, nargs=3, default=list(OCC_SIZE))
    p.add_argument("--pc-range", type=float, nargs=6, default=list(POINT_CLOUD_RANGE))
    p.add_argument("--no-inflate-gmo", dest="inflate_gmo", action="store_false", default=True,
                   help="Mark only LiDAR-hit voxels as GMO instead of the full box volume.")
    p.add_argument("--scene-limit", type=int, default=None, help="Process at most N scenes (subset runs).")
    p.add_argument("--max-samples", type=int, default=None, help="Stop after N samples.")
    p.add_argument("--shard", type=int, default=0, help="This worker's shard index (Slurm array).")
    p.add_argument("--num-shards", type=int, default=1, help="Total shards (Slurm array size).")
    p.add_argument("--overwrite", action="store_true", help="Rebuild .npz files that already exist.")
    p.add_argument("--dry-run", action="store_true", help="Report the plan and disk estimate, write nothing.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.nuscenes_root:
        raise SystemExit("--nuscenes-root (or $NUSCENES_ROOT) is required.")
    if not args.out_root:
        raise SystemExit("--out-root (or $DRIFT_DATA_ROOT) is required.")

    nuscenes_root = Path(args.nuscenes_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root == nuscenes_root or nuscenes_root in out_root.parents:
        raise SystemExit(
            f"--out-root ({out_root}) is inside --nuscenes-root ({nuscenes_root}). "
            "The nuScenes tree is shared/read-only; choose an output directory outside it."
        )
    if not (nuscenes_root / args.version).is_dir():
        raise SystemExit(f"{nuscenes_root / args.version} not found -- is --nuscenes-root correct?")

    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils import splits as nusc_splits
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "nuscenes-devkit is required for preprocessing. Install it on the LOGIN node "
            "(compute nodes have no internet):  pip install nuscenes-devkit"
        ) from exc

    print(f"[prepare] loading {args.version} metadata from {nuscenes_root} ...", flush=True)
    nusc = NuScenes(version=args.version, dataroot=str(nuscenes_root), verbose=False)

    split_scenes = set(
        nusc_splits.train if args.split == "train" else nusc_splits.val
    )
    scenes = [s for s in nusc.scene if s["name"] in split_scenes]
    if args.scene_limit is not None:
        scenes = scenes[: args.scene_limit]

    sample_tokens: List[str] = []
    for scene in scenes:
        tok = scene["first_sample_token"]
        while tok:
            sample_tokens.append(tok)
            tok = nusc.get("sample", tok)["next"]

    if args.num_shards > 1:
        sample_tokens = sample_tokens[args.shard :: args.num_shards]
    if args.max_samples is not None:
        sample_tokens = sample_tokens[: args.max_samples]

    out_dir = out_root / "samples"
    print(
        f"[prepare] split={args.split} scenes={len(scenes)} candidate samples={len(sample_tokens)} "
        f"shard={args.shard}/{args.num_shards}",
        flush=True,
    )
    if args.dry_run:
        est_mb = 12.0
        print(f"[prepare] dry run -- would write ~{len(sample_tokens) * est_mb / 1024:.1f} GB to {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    skipped = 0
    for n, tok in enumerate(sample_tokens):
        entry = process_sample(
            nusc,
            tok,
            nuscenes_root,
            out_dir,
            T_p=args.T_p,
            T_o=args.T_o,
            occ_size=tuple(args.occ_size),
            pc_range=args.pc_range,
            inflate_gmo=args.inflate_gmo,
            overwrite=args.overwrite,
        )
        if entry is None:
            skipped += 1
        else:
            entry["npz_path"] = str(Path("samples") / entry["npz_path"])
            entries.append(entry)
        if (n + 1) % 100 == 0:
            print(f"[prepare] {n + 1}/{len(sample_tokens)} kept={len(entries)} skipped={skipped}", flush=True)

    suffix = "" if args.num_shards == 1 else f".shard{args.shard:03d}"
    ann_path = out_root / f"drift_{args.split}{suffix}.json"
    with open(ann_path, "w") as f:
        json.dump(entries, f)
    print(
        f"[prepare] done: {len(entries)} samples written, {skipped} skipped "
        f"(too close to a scene boundary). Annotation: {ann_path}",
        flush=True,
    )
    if args.num_shards > 1:
        print(f"[prepare] merge shards with: python tools/merge_shards.py --out-root {out_root} --split {args.split}")


if __name__ == "__main__":
    main()
