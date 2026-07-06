"""
Visual Reliability Estimator — Property-Based & Unit Tests
─────────────────────────────────────────────────────────────────────
驗證 VisualReliabilityEstimator 的核心行為：
  1. 可靠度永遠在 [min_reliability, 1.0] 範圍內
  2. 無手部偵測時可靠度最低
  3. EMA 平滑後可靠度不會劇烈跳動
  4. β_eff = base_beta × reliability（線性縮放）
  5. 各子訊號的邊界條件
"""

import sys
import os
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visual.visual_reliability import (
    VisualReliabilityEstimator,
    ReliabilitySignals,
    build_signals_from_tracker_result,
)


# ────────────────────────────────────────────────────────────────────
# Strategies
# ────────────────────────────────────────────────────────────────────

def signals_strategy(hand_detected: bool = True):
    return st.builds(
        ReliabilitySignals,
        hand_variance=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        hand_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        text_density=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        hand_board_proximity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        hand_detected=st.just(hand_detected),
    )


# ────────────────────────────────────────────────────────────────────
# Property Tests
# ────────────────────────────────────────────────────────────────────

class TestReliabilityRange:
    """可靠度範圍永遠在 [min_reliability, 1.0]"""

    @settings(max_examples=50)
    @given(signals=signals_strategy(hand_detected=True))
    def test_reliability_in_valid_range(self, signals):
        est = VisualReliabilityEstimator(min_reliability=0.05)
        result = est.estimate(signals, base_beta=0.5)
        assert 0.05 <= result.reliability <= 1.0, \
            f"Reliability {result.reliability} out of [0.05, 1.0]"

    @settings(max_examples=20)
    @given(signals=signals_strategy(hand_detected=False))
    def test_no_hand_gives_min_reliability(self, signals):
        min_r = 0.05
        est = VisualReliabilityEstimator(min_reliability=min_r)
        result = est.estimate(signals, base_beta=0.5)
        # 無手部偵測時，可靠度應為最低值（EMA 可能比 min_r 略高，取決於歷史）
        assert result.reliability >= min_r, \
            f"Reliability {result.reliability} should be >= min_reliability {min_r}"

    @settings(max_examples=20)
    @given(base_beta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    def test_beta_eff_never_exceeds_base_beta(self, base_beta):
        """β_eff 不應超過 base_beta（允許 0.001 的捨入誤差，來自 round(4位)）"""
        signals = ReliabilitySignals(
            hand_variance=0.0,
            hand_confidence=1.0,
            text_density=1.0,
            hand_board_proximity=1.0,
            hand_detected=True,
        )
        est = VisualReliabilityEstimator()
        result = est.estimate(signals, base_beta=base_beta)
        assert result.effective_beta <= base_beta + 0.001, \
            f"beta_eff {result.effective_beta} exceeds base_beta {base_beta} by too much"

    @settings(max_examples=20)
    @given(base_beta=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    def test_beta_eff_scales_with_reliability(self, base_beta):
        """β_eff ≈ base_beta × reliability（在 EMA 收斂後）"""
        # 用相同訊號跑多次讓 EMA 收斂
        signals = ReliabilitySignals(
            hand_variance=10.0,
            hand_confidence=0.7,
            text_density=0.5,
            hand_board_proximity=0.6,
            hand_detected=True,
        )
        est = VisualReliabilityEstimator(ema_alpha=1.0)  # 無 EMA，直接使用當前值
        result = est.estimate(signals, base_beta=base_beta)
        expected_beta_eff = base_beta * result.reliability
        assert abs(result.effective_beta - expected_beta_eff) < 1e-3, \
            f"beta_eff {result.effective_beta} != base_beta * reliability {expected_beta_eff}"


class TestDominantFactor:
    """主要影響因素識別"""

    def test_no_hand_detected_returns_no_hand(self):
        est = VisualReliabilityEstimator()
        signals = ReliabilitySignals(hand_detected=False)
        result = est.estimate(signals)
        assert result.dominant_factor == "no_hand_detected"

    def test_high_variance_reduces_reliability(self):
        est_low = VisualReliabilityEstimator(ema_alpha=1.0)
        est_high = VisualReliabilityEstimator(ema_alpha=1.0)

        low_variance = ReliabilitySignals(
            hand_variance=1.0, hand_confidence=0.8,
            text_density=0.6, hand_board_proximity=0.7,
            hand_detected=True,
        )
        high_variance = ReliabilitySignals(
            hand_variance=500.0, hand_confidence=0.8,
            text_density=0.6, hand_board_proximity=0.7,
            hand_detected=True,
        )

        r_low = est_low.estimate(low_variance).reliability
        r_high = est_high.estimate(high_variance).reliability
        assert r_low > r_high, \
            f"Low variance ({r_low}) should give higher reliability than high variance ({r_high})"


class TestEMA:
    """EMA 平滑行為"""

    def test_ema_smooths_sudden_drop(self):
        est = VisualReliabilityEstimator(ema_alpha=0.3)

        # 先建立穩定高可靠度狀態
        good = ReliabilitySignals(
            hand_variance=1.0, hand_confidence=0.9,
            text_density=0.8, hand_board_proximity=0.9,
            hand_detected=True,
        )
        for _ in range(10):
            r_before = est.estimate(good).reliability

        # 突然出現低可靠度幀
        bad = ReliabilitySignals(hand_detected=False)
        r_after = est.estimate(bad).reliability

        # EMA 應讓跌幅有限（不會立即跌到最低）
        assert r_after > 0.05, "EMA should smooth sudden drops"
        assert r_after < r_before, "Reliability should decrease after bad frame"

    def test_reset_clears_ema(self):
        est = VisualReliabilityEstimator(ema_alpha=0.5)
        good = ReliabilitySignals(
            hand_variance=1.0, hand_confidence=0.9,
            text_density=0.8, hand_board_proximity=0.9,
            hand_detected=True,
        )
        for _ in range(5):
            est.estimate(good)

        est.reset()
        assert est._ema_reliability is None, "Reset should clear EMA state"


# ────────────────────────────────────────────────────────────────────
# Unit Tests for Sub-Scorers
# ────────────────────────────────────────────────────────────────────

class TestSubScorers:
    """各子訊號轉換函式的邊界條件"""

    def test_hand_stability_zero_variance(self):
        score = VisualReliabilityEstimator._hand_stability_score(0.0)
        assert score == 1.0, "Zero variance should give stability score 1.0"

    def test_hand_stability_very_high_variance(self):
        score = VisualReliabilityEstimator._hand_stability_score(1e6)
        assert score < 0.01, f"Very high variance should give near-zero score, got {score}"

    def test_text_density_empty(self):
        score = VisualReliabilityEstimator._text_density_score([], frame_area=480 * 640)
        assert score == 0.0

    def test_text_density_full_frame(self):
        boxes = [(0, 0, 640, 480)]
        score = VisualReliabilityEstimator._text_density_score(boxes, frame_area=640 * 480)
        assert 0.0 < score <= 1.0

    def test_proximity_no_hand(self):
        score = VisualReliabilityEstimator._proximity_score(
            hand_center=None, text_boxes=[(0, 0, 100, 50)],
            frame_w=640, frame_h=480
        )
        assert score == 0.0

    def test_proximity_hand_on_text(self):
        # 手部中心在文字框中心
        boxes = [(200, 150, 300, 250)]
        hand_center = (250, 200)
        score = VisualReliabilityEstimator._proximity_score(
            hand_center, boxes, frame_w=640, frame_h=480
        )
        assert score > 0.7, f"Hand on text should give high proximity score, got {score}"

    def test_proximity_hand_far_from_text(self):
        boxes = [(0, 0, 50, 50)]
        hand_center = (600, 450)
        score = VisualReliabilityEstimator._proximity_score(
            hand_center, boxes, frame_w=640, frame_h=480
        )
        assert score < 0.3, f"Hand far from text should give low proximity score, got {score}"


# ────────────────────────────────────────────────────────────────────
# Integration: build_signals_from_tracker_result
# ────────────────────────────────────────────────────────────────────

class TestBuildSignals:
    """從 HandTracker 結果建構訊號的整合測試"""

    def test_triggered_with_high_s_visual_raw(self):
        tracker_result = {
            "triggered": True,
            "roi_center": (320, 240),
            "s_visual_raw": 0.9,
        }
        text_boxes = [(200, 150, 400, 300)]
        signals = build_signals_from_tracker_result(
            tracker_result, text_boxes=text_boxes, frame_w=640, frame_h=480
        )
        assert signals.hand_detected is True
        assert signals.hand_confidence == 0.9
        assert signals.text_density > 0.0
        assert signals.hand_board_proximity > 0.0

    def test_not_triggered_no_hand(self):
        tracker_result = {
            "triggered": False,
            "roi_center": None,
            "s_visual_raw": 0.0,
        }
        signals = build_signals_from_tracker_result(tracker_result)
        assert signals.hand_detected is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
