"""A video that cuts between two cities looks like a tracking failure, not a scene change.

`tools/make_video.py`'s `temporal` mode animates consecutive keyframes. The annotation file
has no scene id, so the run boundary is recovered from the nuScenes LiDAR filename -- log
prefix plus microsecond timestamp. Get that wrong and the animation walks straight past the
end of one drive into another, at which point the model appears to lose the entire world in a
single frame. Nothing downstream catches it; it just looks like a bad result.

The frame-size rule is pinned for a duller reason: frames rasterized at differing sizes make
the GIF jitter and make ffmpeg refuse the stream outright.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from tools.make_video import (
    KEYFRAME_INTERVAL_US,
    _frame_to_image,
    contiguous_run,
    drive_and_timestamp,
    render_rollout_figure,
    render_rollout_frame,
    render_temporal_frame,
    temporal_run,
    write_gif,
)

DRIVE_A = "n008-2018-08-01-15-16-36-0400"
DRIVE_B = "n015-2018-07-24-11-22-45+0800"


def _entry(drive: str, ts: int) -> dict:
    return {"points_paths": [f"/data/{drive}__LIDAR_TOP__{ts}.pcd.bin"]}


def _run(drive: str, start_ts: int, n: int, step: int = KEYFRAME_INTERVAL_US) -> list:
    return [_entry(drive, start_ts + i * step) for i in range(n)]


class TestFilenameParsing:
    def test_a_nuscenes_lidar_name_yields_drive_and_timestamp(self) -> None:
        got = drive_and_timestamp(f"/x/y/{DRIVE_A}__LIDAR_TOP__1533151603547590.pcd.bin")
        assert got == (DRIVE_A, 1533151603547590)

    def test_an_unrecognized_name_is_not_an_error(self) -> None:
        """Synthetic data has no nuScenes filenames. That is a normal mode of operation, so
        it must return None rather than raise -- the caller falls back to a plain range."""
        assert drive_and_timestamp("sample_0007.bin") is None
        assert drive_and_timestamp("") is None


class TestSceneBoundaries:
    def test_a_continuous_drive_is_followed(self) -> None:
        assert contiguous_run(_run(DRIVE_A, 1_000_000_000, 10), 0, 40) == list(range(10))

    def test_the_run_stops_when_the_drive_changes(self) -> None:
        """The bug this exists to prevent. Frames 0-4 are one drive, 5 onward another; the
        animation must end at 4 rather than cut to a different city while appearing to be
        continuous footage."""
        entries = _run(DRIVE_A, 1_000_000_000, 5) + _run(DRIVE_B, 9_000_000_000, 5)
        assert contiguous_run(entries, 0, 40) == [0, 1, 2, 3, 4]

    def test_a_large_timestamp_jump_ends_the_run(self) -> None:
        """Same log can contain more than one scene. A multi-second gap is a cut, even
        though the drive prefix never changes."""
        entries = _run(DRIVE_A, 1_000_000_000, 3)
        entries += _run(DRIVE_A, 1_000_000_000 + 30_000_000, 3)
        assert contiguous_run(entries, 0, 40) == [0, 1, 2]

    def test_one_dropped_keyframe_does_not_end_the_run(self) -> None:
        """A single missing keyframe doubles the gap to 1.0 s. Treating that as a scene
        change would chop most drives into useless two-frame clips."""
        entries = _run(DRIVE_A, 1_000_000_000, 2)
        entries.append(_entry(DRIVE_A, 1_000_000_000 + 3 * KEYFRAME_INTERVAL_US))
        assert contiguous_run(entries, 0, 40) == [0, 1, 2]

    def test_max_frames_is_respected(self) -> None:
        assert contiguous_run(_run(DRIVE_A, 1_000_000_000, 100), 0, 12) == list(range(12))

    def test_starting_mid_drive_works(self) -> None:
        assert contiguous_run(_run(DRIVE_A, 1_000_000_000, 10), 6, 40) == [6, 7, 8, 9]

    def test_time_must_move_forward(self) -> None:
        """Entries out of order, or a repeated timestamp, would otherwise produce a video
        that stutters or runs backwards while looking perfectly valid."""
        entries = _run(DRIVE_A, 1_000_000_000, 3)
        entries.append(_entry(DRIVE_A, 1_000_000_000))  # back in time
        assert contiguous_run(entries, 0, 40) == [0, 1, 2]

    def test_synthetic_data_falls_back_to_a_plain_range(self) -> None:
        entries = [{"points_paths": ["sample_0.bin"]} for _ in range(10)]
        assert contiguous_run(entries, 2, 4) == [2, 3, 4, 5]

    def test_an_out_of_range_start_yields_nothing(self) -> None:
        assert contiguous_run(_run(DRIVE_A, 1_000_000_000, 3), 99, 10) == []


class TestRunsAreFoundByTimeNotByIndex:
    """The bug that produced a one-frame "animation" on the real split.

    `tools/prepare_nuscenes.py` shards the sample list with `sample_tokens[shard::num_shards]`,
    so the merged annotation file interleaves shards. Consecutive entries are seconds apart, or
    in different drives, and walking the index finds no neighbours at all.
    """

    def _sharded(self, n_drives: int, per_drive: int, shards: int) -> list:
        """An annotation file built the way the preprocessing actually builds one."""
        ordered = []
        for d in range(n_drives):
            drive = f"n{d:03d}-2018-08-01-15-16-36-0400"
            for k in range(per_drive):
                ordered.append(_entry(drive, 1_000_000_000 + k * KEYFRAME_INTERVAL_US))
        interleaved = []
        for s in range(shards):
            interleaved.extend(ordered[s::shards])
        return interleaved

    def test_a_sharded_index_still_yields_a_full_run(self) -> None:
        """Index order gives one frame; timestamp order gives the whole drive."""
        entries = self._sharded(n_drives=2, per_drive=20, shards=8)
        assert len(contiguous_run(entries, 0, 40)) == 1, "precondition: the index is not in time order"
        assert len(temporal_run(entries, 40)) == 20

    def test_frames_come_back_in_time_order(self) -> None:
        """Returned out of order, the animation would jump back and forth while looking
        like a plausible video."""
        entries = self._sharded(n_drives=1, per_drive=12, shards=4)
        run = temporal_run(entries, 40)
        stamps = [drive_and_timestamp(entries[i]["points_paths"][-1])[1] for i in run]
        assert stamps == sorted(stamps)

    def test_the_longest_drive_is_chosen(self) -> None:
        entries = _run(DRIVE_A, 1_000_000_000, 4) + _run(DRIVE_B, 9_000_000_000, 11)
        assert len(temporal_run(entries, 40)) == 11

    def test_a_run_never_spans_two_drives(self) -> None:
        entries = _run(DRIVE_A, 1_000_000_000, 6) + _run(DRIVE_B, 1_000_000_000, 6)
        run = temporal_run(entries, 40)
        drives = {drive_and_timestamp(entries[i]["points_paths"][-1])[0] for i in run}
        assert len(drives) == 1, f"the animation crossed drives: {drives}"

    def test_an_anchor_index_starts_the_run_there(self) -> None:
        entries = _run(DRIVE_A, 1_000_000_000, 10)
        assert temporal_run(entries, 40, start=4) == [4, 5, 6, 7, 8, 9]

    def test_an_anchor_outside_any_run_falls_back_to_the_longest(self) -> None:
        """A stale --index must not silently produce an empty video."""
        entries = _run(DRIVE_A, 1_000_000_000, 5)
        assert len(temporal_run(entries, 40, start=999)) == 5

    def test_max_frames_still_caps_the_result(self) -> None:
        assert len(temporal_run(_run(DRIVE_A, 1_000_000_000, 80), 12)) == 12

    def test_synthetic_data_falls_back_to_index_order(self) -> None:
        entries = [{"points_paths": ["sample_0.bin"]} for _ in range(10)]
        assert temporal_run(entries, 4, start=2) == [2, 3, 4, 5]


class TestFramesAreVideoSafe:
    def test_every_frame_has_identical_dimensions(self) -> None:
        """Frames of differing size make the GIF jitter and make ffmpeg reject the stream.
        This is what `bbox_inches="tight"` would silently cause, since it crops to whatever
        happens to be drawn in that particular frame."""
        rng = np.random.default_rng(0)
        gt = rng.integers(0, 3, size=(4, 12, 12, 3))
        pred = rng.integers(0, 3, size=(4, 12, 12, 3))
        sizes = {
            _frame_to_image(render_rollout_frame(gt, pred, t, 0.5, "t")).size
            for t in range(gt.shape[0])
        }
        assert len(sizes) == 1, f"frames came out at differing sizes: {sizes}"

    def test_a_temporal_frame_renders_with_and_without_camera_data(self) -> None:
        gt = np.zeros((3, 8, 8, 2), dtype=np.int64)
        imgs = np.random.default_rng(1).random((6, 3, 12, 20), dtype=np.float32)
        with_cam = _frame_to_image(render_temporal_frame(imgs, gt, gt, 0, 0, 3, 0.5, "t"))
        without = _frame_to_image(render_temporal_frame(
            np.zeros((0, 3, 1, 1)), gt, gt, 0, 1, 3, 0.5, "t"))
        assert with_cam.size == without.size, "a dropped camera must not resize the frame"


class TestPaperFigure:
    def test_one_column_per_horizon_and_a_legend(self, tmp_path) -> None:
        """The figure that goes in the PDF: every horizon side by side, three rows, and a
        legend naming the classes -- the animation cannot go in a paper."""
        rng = np.random.default_rng(2)
        gt = rng.integers(0, 3, size=(6, 12, 12, 3))
        pred = rng.integers(0, 3, size=(6, 12, 12, 3))
        fig = render_rollout_figure(gt, pred, 0.5, "figure")

        assert len(fig.axes) == 18, "expected 3 rows x 6 horizons"
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert "+0.5s" in titles and "+3.0s" in titles
        assert fig.legends, "a paper figure without a legend cannot be read"

        out = tmp_path / "fig.pdf"
        fig.savefig(out, bbox_inches="tight")
        assert out.stat().st_size > 5_000

    def test_mismatched_shapes_raise(self) -> None:
        """Rendering a 6-horizon ground truth against a 3-horizon prediction would silently
        pair up the wrong times and produce a figure that misreports the model."""
        with pytest.raises(ValueError, match="shape mismatch"):
            render_rollout_figure(
                np.zeros((6, 8, 8, 2), dtype=np.int64),
                np.zeros((3, 8, 8, 2), dtype=np.int64), 0.5, "x",
            )


class TestGifWriting:
    def test_a_gif_is_written_and_holds_every_frame(self, tmp_path) -> None:
        from PIL import Image

        frames = [Image.new("RGB", (64, 48), (i * 40 % 255, 0, 0)) for i in range(5)]
        out = tmp_path / "clip.gif"
        write_gif(frames, out, fps=4.0)
        assert out.exists()
        with Image.open(out) as gif:
            assert gif.n_frames == 5

    def test_writing_no_frames_fails_loudly(self, tmp_path) -> None:
        """A zero-frame render is a failed run, not an empty video; silently writing a
        broken file would hide that until someone tried to play it."""
        with pytest.raises(ValueError, match="no frames"):
            write_gif([], tmp_path / "empty.gif", fps=4.0)
