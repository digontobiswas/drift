"""Warp correctness tests. See docs/DESIGN_SPEC.md §8.

Identity ego transforms must be a no-op; a pure axis-aligned translation must shift
content by the expected integer number of voxels (exactly, since a translation that is
an integer multiple of the voxel size lands `grid_sample` exactly on voxel centers, with
no bilinear blending).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from drift.data.ego_motion import compose_future_transforms, cumulative_warp_to_present
from drift.models.static_path import StaticForecastPath

_PC_RANGE = [-4.0, -4.0, -4.0, 4.0, 4.0, 4.0]  # 8 voxels/axis @ 1.0 m/voxel
_SIZE = (8, 8, 8)


def _translation(dx: float, dy: float = 0.0, dz: float = 0.0, batch: int = 1) -> torch.Tensor:
    mat = torch.eye(4).unsqueeze(0).expand(batch, 4, 4).clone()
    mat[:, 0, 3] = dx
    mat[:, 1, 3] = dy
    mat[:, 2, 3] = dz
    return mat


def _spike_volume(index: tuple) -> torch.Tensor:
    """(1, 1, X, Y, Z) volume, all zero except a single unit spike at `index`."""
    vol = torch.zeros(1, 1, *_SIZE)
    vol[0, 0, index[0], index[1], index[2]] = 1.0
    return vol


def _argmax_index(vol: torch.Tensor) -> tuple:
    flat = vol[0, 0].reshape(-1)
    idx = int(torch.argmax(flat).item())
    x = idx // (_SIZE[1] * _SIZE[2])
    rem = idx % (_SIZE[1] * _SIZE[2])
    y = rem // _SIZE[2]
    z = rem % _SIZE[2]
    return (x, y, z)


class TestCumulativeWarpToPresent:
    def test_identity_is_noop(self) -> None:
        spike = (2, 3, 4)
        vol = _spike_volume(spike)
        feats = torch.stack([vol, vol], dim=1)  # (1,T_p=2,C=1,X,Y,Z) same content both frames
        ego_motion = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 2, 4, 4)
        out = cumulative_warp_to_present(feats, ego_motion, present_idx=1, point_cloud_range=_PC_RANGE)
        assert torch.allclose(out, feats, atol=1e-5)

    def test_pure_translation_shifts_by_expected_voxels(self) -> None:
        spike = (2, 4, 4)
        past = _spike_volume(spike)
        present = torch.zeros_like(past)  # present frame's own content is irrelevant to this check
        feats = torch.stack([past, present], dim=1)  # (1,2,1,X,Y,Z)

        dx_metres = 2.0  # == 2 voxels at 1.0 m/voxel
        ego_motion = _translation(dx_metres).unsqueeze(1)  # (1,1,4,4): ego_motion[:,0] = T_{1<-0}

        out = cumulative_warp_to_present(feats, ego_motion, present_idx=1, point_cloud_range=_PC_RANGE)
        warped_past = out[:, 0]
        expected = (spike[0] + 2, spike[1], spike[2])
        got = _argmax_index(warped_past)
        assert got == expected, f"expected spike at {expected}, got {got}"
        assert warped_past.sum().item() == 1.0  # exact voxel hit, no interpolation blur/loss

    def test_present_frame_passes_through_unchanged(self) -> None:
        spike = (5, 1, 6)
        vol = _spike_volume(spike)
        other = torch.zeros_like(vol)
        feats = torch.stack([other, vol], dim=1)  # present (index 1) holds the spike
        ego_motion = _translation(3.0).unsqueeze(1)
        out = cumulative_warp_to_present(feats, ego_motion, present_idx=1, point_cloud_range=_PC_RANGE)
        assert torch.equal(out[:, 1], vol)


class TestComposeFutureTransforms:
    def test_identity_ego_gives_identity_future(self) -> None:
        ego_motion = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 5, 4, 4)
        future_ego = compose_future_transforms(ego_motion, present_idx=2, num_future=3)
        expected = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 3, 4, 4)
        assert torch.allclose(future_ego, expected, atol=1e-6)

    def test_cumulative_composition(self) -> None:
        # Two successive 1m x-translations compose to 2m at k=1.
        step = _translation(1.0)
        ego_motion = torch.stack([step, step, torch.eye(4).unsqueeze(0).expand(1, 4, 4).clone()], dim=1)
        future_ego = compose_future_transforms(ego_motion, present_idx=0, num_future=2)
        assert torch.allclose(future_ego[:, 0, 0, 3], torch.tensor([1.0]), atol=1e-6)
        assert torch.allclose(future_ego[:, 1, 0, 3], torch.tensor([2.0]), atol=1e-6)


class TestHalfPrecisionInverse:
    """Regression test for the smoke-test crash on PARAM Shakti (job 1269089):

        RuntimeError: linalg.inv: Low precision dtypes not supported. Got Half

    Under AMP autocast, `feats`/`obs_latent` and the ego-motion transforms that
    accompany them arrive as `torch.float16`. `torch.linalg.inv` refuses Half
    inputs outright, so both `_warp_feature_volume` (via
    `cumulative_warp_to_present`) and `StaticForecastPath.forward` must invert in
    fp32 internally and cast back, never call `linalg.inv` directly on a Half
    tensor. Requires CUDA: CPU `grid_sample` does not reliably support Half
    across torch versions, and that gap is unrelated to the bug being guarded
    against here.
    """

    _requires_cuda = torch.cuda.is_available()

    def test_cumulative_warp_to_present_accepts_half(self) -> None:
        if not self._requires_cuda:
            import pytest

            pytest.skip("needs CUDA: CPU grid_sample has inconsistent Half support")
        device = torch.device("cuda")
        spike = (2, 3, 4)
        vol = _spike_volume(spike).to(device=device, dtype=torch.float16)
        feats = torch.stack([vol, vol], dim=1)  # (1,T_p=2,C=1,X,Y,Z)
        ego_motion = _translation(2.0, batch=1).to(device=device, dtype=torch.float16).unsqueeze(1)
        out = cumulative_warp_to_present(feats, ego_motion, present_idx=1, point_cloud_range=_PC_RANGE)
        assert out.dtype == torch.float16

    def test_static_forecast_path_accepts_half(self) -> None:
        if not self._requires_cuda:
            import pytest

            pytest.skip("needs CUDA: CPU grid_sample has inconsistent Half support")
        device = torch.device("cuda")
        path = StaticForecastPath(
            latent_size=_SIZE, point_cloud_range=_PC_RANGE, learned_residual=False, channels=1,
        ).to(device=device, dtype=torch.float16)
        path.eval()
        obs = _spike_volume((2, 4, 4)).to(device=device, dtype=torch.float16)
        future_ego = _translation(2.0).to(device=device, dtype=torch.float16).unsqueeze(1)
        out = path(obs, future_ego)
        assert out.dtype == torch.float16


class TestStaticForecastPath:
    def test_identity_future_ego_is_noop_when_residual_is_zero_init(self) -> None:
        # StaticForecastPath's learned residual conv is zero-initialized (see its module docstring
        # / implementation), so at init, an identity future_ego must reproduce the input exactly.
        path = StaticForecastPath(
            latent_size=_SIZE, point_cloud_range=_PC_RANGE, learned_residual=True, channels=1,
        )
        path.eval()
        spike = (3, 3, 3)
        obs = _spike_volume(spike)  # (B=1,C=1,X,Y,Z)
        future_ego = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 4, 4, 4)
        out = path(obs, future_ego)
        for t in range(out.shape[1]):
            assert torch.allclose(out[:, t], obs, atol=1e-5)

    def test_pure_translation_shifts_by_expected_voxels(self) -> None:
        path = StaticForecastPath(
            latent_size=_SIZE, point_cloud_range=_PC_RANGE, learned_residual=False, channels=1,
        )
        path.eval()
        spike = (2, 4, 4)
        obs = _spike_volume(spike)
        dx_metres = 2.0
        future_ego = _translation(dx_metres).unsqueeze(1)  # (1,1,4,4)
        out = path(obs, future_ego)
        got = _argmax_index(out[:, 0])
        expected = (spike[0] + 2, spike[1], spike[2])
        assert got == expected, f"expected spike at {expected}, got {got}"
        assert out[0, 0].sum().item() == 1.0
