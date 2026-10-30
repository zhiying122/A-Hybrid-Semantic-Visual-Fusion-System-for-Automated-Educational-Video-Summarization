import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.semantic.prosodic_analyzer_optimized as prosodic_module
from src.semantic.prosodic_analyzer_optimized import ProsodicAnalyzer
from main import _apply_prosodic_adjustment


def test_speech_rate_uses_transcript_characters_per_second():
    assert ProsodicAnalyzer._estimate_speech_rate("你好 世界", 2.0) == pytest.approx(2.0)


def test_speech_rate_handles_missing_text_or_duration():
    assert ProsodicAnalyzer._estimate_speech_rate("", 2.0) == 0.0
    assert ProsodicAnalyzer._estimate_speech_rate("有文字", 0.0) == 0.0


def test_missing_librosa_marks_prosodic_feature_unavailable(monkeypatch):
    monkeypatch.setattr(prosodic_module, "LIBROSA_AVAILABLE", False)
    segments = [{"start": 0.0, "end": 1.0}]

    analyzed = ProsodicAnalyzer().analyze_segments("missing.wav", segments)

    assert analyzed[0]["s_prosodic"] == 0.5
    assert analyzed[0]["prosodic_available"] is False


def test_invalid_segment_is_excluded_from_batch_normalization(monkeypatch):
    monkeypatch.setattr(prosodic_module, "LIBROSA_AVAILABLE", True)
    monkeypatch.setattr(
        prosodic_module,
        "librosa",
        SimpleNamespace(load=lambda *_args, **_kwargs: (np.ones(22050), 22050)),
    )
    analyzer = ProsodicAnalyzer()
    monkeypatch.setattr(
        analyzer,
        "_extract_features",
        lambda *_args, **_kwargs: {
            "speech_rate": 1.0,
            "volume": 0.2,
            "pitch_variation": 0.3,
            "pause_ratio": 0.1,
        },
    )
    segments = [
        {"start": 0.0, "end": 0.01},
        {"start": 0.0, "end": 1.0},
    ]

    analyzed = analyzer.analyze_segments("available.wav", segments)

    assert analyzed[0]["prosodic_available"] is False
    assert analyzed[0]["s_prosodic"] == 0.5
    assert analyzed[1]["prosodic_available"] is True
    assert 0.0 <= analyzed[1]["s_prosodic"] <= 1.0


def test_unavailable_prosodic_feature_does_not_change_fusion_score():
    assert _apply_prosodic_adjustment(
        0.6,
        {"s_prosodic": 0.9, "prosodic_available": False},
    ) == pytest.approx(0.6)


def test_available_prosodic_feature_adjusts_fusion_score():
    assert _apply_prosodic_adjustment(
        0.6,
        {"s_prosodic": 1.0, "prosodic_available": True},
    ) > 0.6
