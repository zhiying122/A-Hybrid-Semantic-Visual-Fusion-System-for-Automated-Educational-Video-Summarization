"""
Adaptive Fusion — Property-Based & Unit Tests
─────────────────────────────────────────────────────────────────────
驗證 adaptive_fusion.py 的核心行為：
  1. fuse_scores() 結果永遠在 [0, 1]
  2. visual_reliability=1.0 時等同靜態加權（向後兼容）
  3. visual_reliability=0.0 時退化為純語意（β_eff=0）
  4. semantic_sliding_window() 邊界條件
  5. grid_search_weights() 回傳合法的 α/β
"""

import sys
import os
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import (
    fuse_scores,
    semantic_sliding_window,
    grid_search_weights,
    _compute_recall,
)
from config import FUSION_SCORE_THRESHOLD


# ────────────────────────────────────────────────────────────────────
# Strategies
# ────────────────────────────────────────────────────────────────────

def score_float():
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def weight_pair():
    """α + β = 1.0 的合法權重對"""
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).map(
        lambda a: (round(a, 2), round(1.0 - a, 2))
    )


def segment_strategy():
    return st.builds(
        lambda s, length, st_, sv: {
            "start": s,
            "end": s + length,
            "s_text": st_,
            "s_visual": sv,
            "text": "test",
        },
        s=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        length=st.floats(min_value=1.0, max_value=60.0, allow_nan=False, allow_infinity=False),
        st_=score_float(),
        sv=score_float(),
    )


# ────────────────────────────────────────────────────────────────────
# fuse_scores Properties
# ────────────────────────────────────────────────────────────────────

class TestFuseScoresProperties:

    @settings(max_examples=100)
    @given(
        s_text=score_float(),
        s_visual=score_float(),
        alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        reliability=score_float(),
    )
    def test_fuse_scores_always_in_range(self, s_text, s_visual, alpha, reliability):
        """融合分數永遠在 [0, 1]"""
        beta = 1.0 - alpha
        score = fuse_scores(s_text, s_visual, alpha, beta, visual_reliability=reliability)
        assert 0.0 <= score <= 1.0, \
            f"fuse_scores({s_text}, {s_visual}, α={alpha}, r={reliability}) = {score}"

    @settings(max_examples=50)
    @given(s_text=score_float(), s_visual=score_float(), alpha=score_float())
    def test_reliability_1_equals_static_fusion(self, s_text, s_visual, alpha):
        """R_visual=1.0 時應等同靜態加權（向後兼容）"""
        beta = 1.0 - alpha
        static_score = alpha * s_text + beta * s_visual
        adaptive_score = fuse_scores(s_text, s_visual, alpha, beta, visual_reliability=1.0)
        assert abs(adaptive_score - static_score) < 1e-6, \
            f"Reliability=1.0 should match static fusion: {adaptive_score} vs {static_score}"

    @settings(max_examples=50)
    @given(s_text=score_float(), s_visual=score_float(), alpha=score_float())
    def test_reliability_0_degrades_to_semantic_only(self, s_text, s_visual, alpha):
        """R_visual=0.0 時應退化為純語意（β_eff=0，score = s_text）"""
        beta = 1.0 - alpha
        score = fuse_scores(s_text, s_visual, alpha, beta, visual_reliability=0.0)
        # β_eff = β × 0 = 0，α_eff = 1.0
        assert abs(score - s_text) < 1e-6, \
            f"Reliability=0 should give s_text={s_text}, got {score}"

    @settings(max_examples=50)
    @given(s_text=score_float(), s_visual=score_float(), alpha=score_float(), reliability=score_float())
    def test_higher_s_visual_increases_score_when_reliability_positive(self, s_text, s_visual, alpha, reliability):
        """視覺可靠度 > 0 且 s_visual2 > s_visual1 時，融合分數應非遞減"""
        assume(reliability > 0.0)
        beta = 1.0 - alpha
        assume(beta > 0.0)
        s_visual2 = min(1.0, s_visual + 0.1)
        score1 = fuse_scores(s_text, s_visual, alpha, beta, visual_reliability=reliability)
        score2 = fuse_scores(s_text, s_visual2, alpha, beta, visual_reliability=reliability)
        assert score2 >= score1 - 1e-9, \
            f"Higher s_visual should not decrease fused score when reliability > 0"


# ────────────────────────────────────────────────────────────────────
# semantic_sliding_window
# ────────────────────────────────────────────────────────────────────

class TestSlidingWindow:

    def test_empty_segments_returns_empty(self):
        result = semantic_sliding_window([], [])
        assert result == []

    def test_all_below_threshold_returns_empty(self):
        segs = [
            {"start": 0, "end": 10, "s_text": 0.1, "s_visual": 0.1, "text": ""},
            {"start": 10, "end": 20, "s_text": 0.05, "s_visual": 0.05, "text": ""},
        ]
        scores = [0.05, 0.05]
        result = semantic_sliding_window(segs, scores)
        assert result == [], f"All below threshold should return empty, got {result}"

    def test_high_score_segment_selected(self):
        segs = [
            {"start": 0, "end": 10, "s_text": 0.9, "s_visual": 0.9, "text": "重要"},
            {"start": 12, "end": 22, "s_text": 0.05, "s_visual": 0.05, "text": "不重要"},
        ]
        scores = [0.9, 0.05]
        result = semantic_sliding_window(segs, scores)
        assert len(result) >= 1
        assert result[0]["start"] == 0

    def test_selected_segments_have_required_keys(self):
        segs = [
            {"start": 0, "end": 30, "s_text": 0.8, "s_visual": 0.7, "text": "重點"},
        ]
        scores = [0.8]
        result = semantic_sliding_window(segs, scores)
        if result:
            for seg in result:
                for key in ("start", "end", "s_text", "s_visual", "text"):
                    assert key in seg, f"Missing key '{key}' in selected segment"


# ────────────────────────────────────────────────────────────────────
# grid_search_weights
# ────────────────────────────────────────────────────────────────────

class TestGridSearch:

    def test_returns_valid_weights(self):
        segs = [
            {"start": i * 30.0, "end": (i + 1) * 30.0,
             "s_text": np.random.uniform(0, 1), "s_visual": np.random.uniform(0, 1),
             "text": f"片段{i}"}
            for i in range(8)
        ]
        gt = [(segs[0]["start"], segs[0]["end"]), (segs[3]["start"], segs[3]["end"])]
        alpha, beta = grid_search_weights(segs, gt)

        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= beta <= 1.0
        assert abs(alpha + beta - 1.0) < 1e-6, f"α+β should be 1.0, got {alpha+beta}"

    def test_empty_gt_returns_default(self):
        segs = [{"start": 0, "end": 10, "s_text": 0.5, "s_visual": 0.5, "text": ""}]
        alpha, beta = grid_search_weights(segs, [])
        assert 0.0 <= alpha <= 1.0
        assert abs(alpha + beta - 1.0) < 1e-6


# ────────────────────────────────────────────────────────────────────
# _compute_recall (internal)
# ────────────────────────────────────────────────────────────────────

class TestComputeRecall:

    def test_empty_gt_returns_zero(self):
        selected = [{"start": 0, "end": 10}]
        assert _compute_recall(selected, []) == 0.0

    def test_perfect_match(self):
        selected = [{"start": 0, "end": 10}]
        gt = [(0.0, 10.0)]
        assert _compute_recall(selected, gt) == 1.0

    def test_no_overlap(self):
        selected = [{"start": 100, "end": 110}]
        gt = [(0.0, 10.0)]
        assert _compute_recall(selected, gt) == 0.0

    def test_partial_overlap_50pct_counts_as_hit(self):
        # GT: 0~10, selected: 5~15 → overlap 5/10 = 50% → 命中
        selected = [{"start": 5, "end": 15}]
        gt = [(0.0, 10.0)]
        recall = _compute_recall(selected, gt)
        assert recall == 1.0, f"50% overlap should count as hit, got {recall}"

    def test_partial_overlap_below_50pct_not_a_hit(self):
        # GT: 0~10, selected: 6~16 → overlap 4/10 = 40% → 不命中
        selected = [{"start": 6, "end": 16}]
        gt = [(0.0, 10.0)]
        recall = _compute_recall(selected, gt)
        assert recall == 0.0, f"40% overlap should not count as hit, got {recall}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
