import os
import sys
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visual.clip_scorer import CLIPScorer


def test_keyword_scores_are_averaged_individually(monkeypatch):
    scorer = object.__new__(CLIPScorer)
    scores = iter([0.2, 0.6, 1.0])
    compute_score = Mock(side_effect=lambda *args, **kwargs: next(scores))
    monkeypatch.setattr(scorer, "compute_clip_score", compute_score)

    result = scorer.compute_clip_score_with_keywords(
        np.zeros((2, 2, 3), dtype=np.uint8),
        [" first ", "", "second", "third"],
    )

    assert result == pytest.approx(0.6)
    assert compute_score.call_count == 3


def test_invalid_color_format_is_reported():
    scorer = object.__new__(CLIPScorer)
    scorer.available = True

    with pytest.raises(ValueError, match="color_format"):
        scorer.compute_clip_score(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "text",
            color_format="rgba",
        )


def test_failed_model_initialization_can_retry(monkeypatch):
    monkeypatch.setattr(CLIPScorer, "_instance", None)
    monkeypatch.setattr(CLIPScorer, "_initialized", False)
    attempts = []

    def load_model(self):
        attempts.append(True)
        self.available = len(attempts) > 1

    monkeypatch.setattr(CLIPScorer, "_load_model", load_model)

    first = CLIPScorer()
    second = CLIPScorer()

    assert first is second
    assert len(attempts) == 2
    assert second.is_available
