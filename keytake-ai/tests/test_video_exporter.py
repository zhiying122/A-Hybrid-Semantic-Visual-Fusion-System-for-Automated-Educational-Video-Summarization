import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.output.video_exporter_optimized import (
    _build_timeline_mapping,
    _normalize_segments,
    export_summary_video,
)


def test_normalized_overlapping_clips_preserve_source_ranges():
    clips = _normalize_segments(
        [
            {"start": 2.0, "end": 4.0},
            {"start": 3.5, "end": 5.0},
        ],
        duration=10.0,
        padding_sec=0.5,
    )

    assert len(clips) == 1
    assert clips[0]["clip_start"] == pytest.approx(1.5)
    assert clips[0]["clip_end"] == pytest.approx(5.5)
    assert clips[0]["source_ranges"] == [
        {"source_start": 2.0, "source_end": 4.0},
        {"source_start": 3.5, "source_end": 5.0},
    ]


def test_timeline_mapping_uses_encoded_and_actual_summary_durations():
    clips = [
        {
            "clip_start": 1.5,
            "clip_end": 5.5,
            "source_ranges": [
                {"source_start": 2.0, "source_end": 4.0},
                {"source_start": 3.5, "source_end": 5.0},
            ],
        }
    ]

    mapping = _build_timeline_mapping(clips, [3.8], summary_duration=3.6)

    assert mapping[0]["output_start"] == pytest.approx(0.45)
    assert mapping[0]["output_end"] == pytest.approx(2.25)
    assert mapping[1]["output_start"] == pytest.approx(1.8)
    assert mapping[1]["output_end"] == pytest.approx(3.15)


def test_timeline_path_cannot_overwrite_source_or_summary(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    destination = tmp_path / "summary.mp4"
    segments = [{"start": 0.0, "end": 1.0}]

    with pytest.raises(ValueError, match="時間對照檔"):
        export_summary_video(
            str(source),
            segments,
            str(destination),
            timeline_output_path=str(source),
        )

    with pytest.raises(ValueError, match="時間對照檔"):
        export_summary_video(
            str(source),
            segments,
            str(destination),
            timeline_output_path=str(destination),
        )
