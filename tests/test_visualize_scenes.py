"""A qualitative figure that is subtly wrong is worse than no figure, because it is believed.

Nothing downstream of `tools/visualize_scenes.py` checks its output -- a BEV rendered with the
axes swapped, or with predictions and ground truth silently misaligned, looks exactly as
convincing as a correct one and goes straight into a paper. The geometry and the semantics are
therefore pinned here rather than eyeballed once.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from tools.visualize_scenes import (
    GMO_CLASS_NAMES,
    bev_from_occupancy,
    class_labels,
    disagreement_map,
    legend_handles,
    lidar_bev,
    pick_indices,
    render_scene,
)

RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]


class TestBevCollapse:
    def test_a_movable_object_is_not_hidden_behind_static_structure(self) -> None:
        """The reason the collapse is a max and not, say, the first non-free hit. A car
        (class 2) at the top of a column and a wall (class 1) below it must render as a car:
        the movable objects are the entire subject of the forecast."""
        occ = np.zeros((2, 2, 4), dtype=np.int64)
        occ[0, 0, 0] = 1
        occ[0, 0, 3] = 2
        assert bev_from_occupancy(occ)[0, 0] == 2

    def test_empty_columns_stay_free(self) -> None:
        assert bev_from_occupancy(np.zeros((3, 3, 5), dtype=np.int64)).sum() == 0

    def test_shape_is_preserved(self) -> None:
        assert bev_from_occupancy(np.zeros((7, 5, 3), dtype=np.int64)).shape == (7, 5)

    def test_a_non_volumetric_input_is_rejected(self) -> None:
        """Passing (T_o, X, Y, Z) by mistake would otherwise collapse the wrong axis and
        produce a picture of nothing in particular."""
        with pytest.raises(ValueError, match="X, Y, Z"):
            bev_from_occupancy(np.zeros((4, 8, 8, 2), dtype=np.int64))


class TestLidarBinning:
    def test_a_point_lands_in_the_cell_its_coordinates_name(self) -> None:
        """A half-cell offset here shifts the whole LiDAR panel relative to the occupancy
        panels above and below it, which makes a correct forecast look misaligned."""
        pts = np.array([[-39.9, -39.9, 0.0], [39.9, 39.9, 0.0]], dtype=np.float32)
        grid = lidar_bev(pts, RANGE, (8, 8))
        assert grid[0, 0] == 1.0
        assert grid[7, 7] == 1.0
        assert grid.sum() == 2.0

    def test_points_outside_the_range_are_dropped_not_clamped(self) -> None:
        """Clamping piles every distant return onto the border cells, drawing a bright frame
        around the scene that no sensor measured -- and it looks like real structure."""
        pts = np.array([[500.0, 0.0, 0.0], [-500.0, 0.0, 0.0]], dtype=np.float32)
        assert lidar_bev(pts, RANGE, (8, 8)).sum() == 0.0

    def test_repeated_points_accumulate(self) -> None:
        pts = np.zeros((5, 3), dtype=np.float32)
        assert lidar_bev(pts, RANGE, (4, 4)).max() == 5.0

    def test_an_empty_sweep_is_an_empty_image_not_a_crash(self) -> None:
        assert lidar_bev(np.zeros((0, 3), dtype=np.float32), RANGE, (4, 4)).sum() == 0.0


class TestDisagreementMap:
    def test_the_three_failure_modes_stay_separate(self) -> None:
        """Merging them into one "error" colour hides the distinction a reader most needs:
        a model that misses objects and one that invents them fail in opposite directions."""
        gt = np.array([[0, 1, 2, 1]])
        pred = np.array([[0, 0, 2, 2]])
        assert disagreement_map(gt, pred).tolist() == [[0, 1, 0, 3]]

    def test_a_hallucination_is_marked_spurious(self) -> None:
        assert disagreement_map(np.array([[0]]), np.array([[2]]))[0, 0] == 2

    def test_mismatched_shapes_raise_rather_than_broadcast(self) -> None:
        """NumPy would happily broadcast a (1, N) against an (N, N) and produce a full,
        plausible-looking error map computed against the wrong ground truth."""
        with pytest.raises(ValueError, match="shape mismatch"):
            disagreement_map(np.zeros((1, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64))


class TestSceneSelection:
    def test_scenes_are_spread_across_the_split(self) -> None:
        """The first N samples of a driving split are one contiguous stretch of one drive.
        A gallery drawn from them characterizes one road, while appearing to characterize
        the split."""
        picked = pick_indices(6000, 6)
        assert picked[0] == 0 and picked[-1] > 4000
        assert len(set(picked)) == 6

    def test_asking_for_more_scenes_than_exist_is_not_an_error(self) -> None:
        assert pick_indices(3, 10) == [0, 1, 2]

    def test_an_empty_split_yields_nothing(self) -> None:
        assert pick_indices(0, 4) == []


class TestLegendNamesTheClasses:
    def test_the_gmo_preset_gets_real_names_not_indices(self) -> None:
        """A paper figure legend reading "class 1 / class 2" tells a reader nothing. The
        3-class preset's meanings are fixed by the config, so the legend can say them."""
        assert class_labels(3) == GMO_CLASS_NAMES
        assert "movable" in class_labels(3)[2]

    def test_the_seventeen_class_vocabulary_is_used_when_it_applies(self) -> None:
        labels = class_labels(17)
        assert labels[0] == "free" and "pedestrian" in labels

    def test_an_unknown_class_count_falls_back_rather_than_guessing(self) -> None:
        """Inventing names for a class count nobody defined would put confident, wrong
        labels in a figure -- worse than an honest index."""
        assert class_labels(5) == ["class 0", "class 1", "class 2", "class 3", "class 4"]

    def test_free_space_gets_no_swatch(self) -> None:
        """Free is drawn as the page colour. A legend swatch matching the background reads
        as a printing fault, and it is the one class a reader never needs pointed out."""
        _, labels = legend_handles(3)
        assert "free" not in labels
        assert labels[:2] == GMO_CLASS_NAMES[1:]

    def test_disagreement_entries_can_be_left_out(self) -> None:
        """A layout with no disagreement panel must not advertise those colours -- the
        reader goes looking for a colour that is not in the picture."""
        _, with_errors = legend_handles(3)
        _, without = legend_handles(3, include_disagreement=False)
        assert len(with_errors) == len(without) + 3
        assert not [l for l in without if "missed" in l or "spurious" in l]


class TestRenderRunsEndToEnd:
    def test_a_full_page_is_produced_with_every_row(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        T_o, X, Y, Z, n_cam = 3, 16, 16, 4, 6
        gt = rng.integers(0, 3, size=(T_o, X, Y, Z))
        pred = rng.integers(0, 3, size=(T_o, X, Y, Z))
        imgs = rng.random((n_cam, 3, 24, 40), dtype=np.float32)
        pts = rng.uniform(-30, 30, size=(500, 3)).astype(np.float32)

        fig = render_scene(imgs, pts, gt, pred, RANGE, title="smoke")
        out = tmp_path / "scene.png"
        fig.savefig(out)
        assert out.stat().st_size > 5_000, "figure rendered but is suspiciously empty"

    def test_a_scene_with_no_lidar_returns_still_renders(self, tmp_path) -> None:
        """A dropped LiDAR frame is a scenario the robustness study deliberately creates;
        the figure for it must show an empty sweep, not fail to draw."""
        gt = np.zeros((2, 8, 8, 2), dtype=np.int64)
        fig = render_scene(
            np.zeros((6, 3, 8, 8), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            gt, gt, RANGE, title="no lidar",
        )
        out = tmp_path / "scene.png"
        fig.savefig(out)
        assert out.exists()
