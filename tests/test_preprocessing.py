"""Geometry checks for `tools/prepare_nuscenes.py`.

These run without nuScenes installed or downloaded: they exercise the pure
geometry helpers, which are where a silent error would corrupt every derived
ground-truth file without ever raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_nuscenes import (  # noqa: E402
    VoxelGrid,
    _quat_to_rotmat,
    _rasterize_box,
    _sparse_encode,
    _yaw_from_rotmat,
)

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
OCC_SIZE = (512, 512, 40)
# nuScenes `size` is [width, length, height]; a mid-size car.
CAR_WLH = [2.0, 4.0, 1.6]


@pytest.fixture
def grid() -> VoxelGrid:
    return VoxelGrid(PC_RANGE, OCC_SIZE)


class TestQuaternion:
    def test_identity(self):
        assert np.allclose(_quat_to_rotmat([1, 0, 0, 0]), np.eye(3))

    def test_ninety_degrees_about_z(self):
        """nuScenes quaternions are [w, x, y, z] -- getting the order wrong
        silently rotates every box in the dataset."""
        R = _quat_to_rotmat([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)
        assert _yaw_from_rotmat(R) == pytest.approx(np.pi / 2)

    def test_random_quaternion_is_a_rotation(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.normal(size=4)
            R = _quat_to_rotmat(q / np.linalg.norm(q))
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
            assert np.linalg.det(R) == pytest.approx(1.0)

    def test_degenerate_quaternion_does_not_blow_up(self):
        assert np.allclose(_quat_to_rotmat([0, 0, 0, 0]), np.eye(3))


class TestBoxRasterization:
    def test_voxel_count_matches_analytic_volume(self, grid):
        vox = _rasterize_box(grid, np.zeros(3), CAR_WLH, 0.0)
        vx, vy, vz = grid.voxel
        expected = (CAR_WLH[1] / vx) * (CAR_WLH[0] / vy) * (CAR_WLH[2] / vz)
        assert 0.7 * expected < len(vox) < 1.4 * expected

    def test_rotation_preserves_volume_but_moves_voxels(self, grid):
        base = _rasterize_box(grid, np.zeros(3), CAR_WLH, 0.0)
        rot45 = _rasterize_box(grid, np.zeros(3), CAR_WLH, np.pi / 4)
        assert 0.7 * len(base) < len(rot45) < 1.4 * len(base)
        # If yaw were ignored, these two sets would be identical.
        assert {tuple(v) for v in base} != {tuple(v) for v in rot45}

    def test_box_outside_range_yields_nothing(self, grid):
        assert _rasterize_box(grid, np.array([500.0, 0.0, 0.0]), CAR_WLH, 0.0).shape[0] == 0

    def test_offset_box_lands_at_its_own_centre(self, grid):
        centre = np.array([20.0, 10.0, 0.0])
        vox = _rasterize_box(grid, centre, CAR_WLH, 0.0)
        recovered = grid.index_to_metric(vox.astype(float).mean(axis=0)[None])[0]
        assert np.allclose(recovered[:2], centre[:2], atol=0.5)

    def test_index_metric_round_trip(self, grid):
        pts = np.array([[0.0, 0.0, 0.0], [20.0, -30.0, 1.0], [-51.0, 51.0, -4.0]])
        back = grid.index_to_metric(np.floor(grid.metric_to_index(pts)))
        assert np.allclose(back, pts, atol=max(grid.voxel))


class TestSparseEncoding:
    def test_drops_free_voxels_and_keeps_pairs(self):
        occ = np.zeros((16, 16, 4), np.uint8)
        inst = np.zeros((16, 16, 4), np.uint16)
        occ[1, 2, 3], inst[1, 2, 3] = 2, 7
        occ[5, 5, 0] = 1
        coords, occ_v, inst_v = _sparse_encode(occ, inst)
        assert coords.shape == (2, 3)
        assert set(occ_v.tolist()) == {1, 2}
        assert 7 in inst_v.tolist()

    def test_all_free_grid_encodes_empty(self):
        coords, occ_v, inst_v = _sparse_encode(
            np.zeros((8, 8, 2), np.uint8), np.zeros((8, 8, 2), np.uint16)
        )
        assert coords.shape == (0, 3) and occ_v.shape == (0,) and inst_v.shape == (0,)

    def test_round_trips_through_the_dataset_decoder(self):
        """`_sparse_encode` and `Cam4DOccDataset._decode_sparse` are two halves of
        one format; this pins them together."""
        import torch

        from drift.data.cam4docc_dataset import Cam4DOccDataset

        rng = np.random.default_rng(0)
        size = (16, 16, 4)
        dense_occ = (rng.random(size) < 0.1).astype(np.uint8) * rng.integers(1, 3, size).astype(np.uint8)
        dense_inst = (dense_occ > 0).astype(np.uint16) * rng.integers(0, 5, size).astype(np.uint16)

        coords, occ_v, inst_v = _sparse_encode(dense_occ, dense_inst)
        npz = {
            "occ_size": np.asarray(size, np.int32),
            "sparse_counts": np.asarray([coords.shape[0]], np.int64),
            "sparse_coords": coords,
            "sparse_occ": occ_v,
            "sparse_instance": inst_v,
        }
        occ, inst = Cam4DOccDataset._decode_sparse(npz)
        assert torch.equal(occ[0], torch.from_numpy(dense_occ.astype(np.int64)))
        assert torch.equal(inst[0], torch.from_numpy(dense_inst.astype(np.int64)))
