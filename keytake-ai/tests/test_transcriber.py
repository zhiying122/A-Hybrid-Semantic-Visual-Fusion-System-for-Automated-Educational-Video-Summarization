import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic.transcriber_optimized import _deduplicate_segments
from src.semantic import transcriber_optimized


def test_identical_overlapping_segments_are_merged():
    result = _deduplicate_segments(
        [
            {"start": 1.0, "end": 3.0, "text": "same phrase"},
            {"start": 1.1, "end": 3.1, "text": "same phrase"},
        ]
    )

    assert result == [{"start": 1.0, "end": 3.1, "text": "same phrase"}]


def test_legitimate_repetition_with_time_gap_is_preserved():
    result = _deduplicate_segments(
        [
            {"start": 1.0, "end": 2.0, "text": "重要公式"},
            {"start": 2.5, "end": 3.5, "text": "重要公式"},
        ]
    )

    assert len(result) == 2


def test_partially_overlapping_repetition_is_preserved():
    result = _deduplicate_segments(
        [
            {"start": 1.0, "end": 3.0, "text": "same phrase"},
            {"start": 2.0, "end": 4.0, "text": "same phrase"},
        ]
    )

    assert len(result) == 2


def test_invalid_segments_are_removed():
    result = _deduplicate_segments(
        [
            {"start": -1, "end": 2, "text": "invalid"},
            {"start": 0, "end": 0, "text": "invalid"},
            {"start": 1, "end": 2, "text": "  valid   phrase "},
        ]
    )

    assert result == [{"start": 1.0, "end": 2.0, "text": "valid phrase"}]


def test_transcription_metadata_records_selected_backend(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("WHISPER_MODE", "local")
    monkeypatch.setattr(
        transcriber_optimized,
        "_transcribe_local",
        lambda path, metadata: (
            metadata.update({"backend": "local", "model": "base"})
            or [{"start": 0.0, "end": 1.0, "text": "words"}]
        ),
    )

    result = transcriber_optimized.transcribe_with_metadata(str(audio_path))

    assert result["segments"][0]["text"] == "words"
    assert result["metadata"] == {
        "requested_backend": "local",
        "backend": "local",
        "model": "base",
        "fallback_reason": None,
    }
