"""
Gesture Classifier — Unit & Property Tests
──────────────────────────────────────────────────────────────────
驗證 GestureClassifier 的核心行為：
  1. 規則型分類器：邊界條件和意圖判斷正確性
  2. 特徵萃取：輸出 shape 和數值範圍正確
  3. 意圖分數：每種意圖對應正確的重要性分數
  4. 分類器介面：不論是 GRU 還是規則型，API 一致
"""

import sys
import os
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visual.gesture_classifier import (
    GestureClassifier,
    GestureIntent,
    GestureResult,
    INTENT_SCORE_MAP,
    rule_based_classify,
    extract_trajectory_features,
    pad_or_truncate,
    _make_result,
)


# ────────────────────────────────────────────────────────────────────
# 特徵萃取
# ────────────────────────────────────────────────────────────────────

class TestFeatureExtraction:

    def test_output_shape_matches_input(self):
        coords = [(100, 200), (105, 205), (110, 210)]
        features = extract_trajectory_features(coords, frame_w=640, frame_h=480)
        assert features.shape == (3, 4), f"Expected (3, 4), got {features.shape}"

    def test_position_normalized_to_0_1(self):
        coords = [(640, 480)]
        features = extract_trajectory_features(coords, frame_w=640, frame_h=480)
        x_norm, y_norm = features[0, 0], features[0, 1]
        assert 0.0 <= x_norm <= 1.0
        assert 0.0 <= y_norm <= 1.0

    def test_empty_coords_returns_zeros(self):
        features = extract_trajectory_features([], frame_w=640, frame_h=480)
        assert features.shape[1] == 4
        assert np.all(features == 0)

    def test_velocity_zero_for_single_point(self):
        coords = [(100, 100)]
        features = extract_trajectory_features(coords, frame_w=640, frame_h=480)
        assert features[0, 2] == 0.0  # dx
        assert features[0, 3] == 0.0  # dy

    def test_velocity_nonzero_for_moving_point(self):
        coords = [(0, 0), (100, 100)]
        features = extract_trajectory_features(coords, frame_w=640, frame_h=480)
        # 第二幀應有非零速度
        assert abs(features[1, 2]) > 0 or abs(features[1, 3]) > 0

    def test_pad_or_truncate_extends_short(self):
        features = np.ones((5, 4), dtype=np.float32)
        padded = pad_or_truncate(features, seq_len=20)
        assert padded.shape == (20, 4)
        # 前 15 行應是零填充
        assert np.all(padded[:15] == 0)

    def test_pad_or_truncate_truncates_long(self):
        features = np.ones((30, 4), dtype=np.float32)
        truncated = pad_or_truncate(features, seq_len=20)
        assert truncated.shape == (20, 4)

    @settings(max_examples=30)
    @given(
        n_points=st.integers(min_value=1, max_value=50),
        seq_len=st.integers(min_value=5, max_value=30),
    )
    def test_pad_always_returns_correct_shape(self, n_points, seq_len):
        features = np.random.rand(n_points, 4).astype(np.float32)
        result = pad_or_truncate(features, seq_len=seq_len)
        assert result.shape == (seq_len, 4)


# ────────────────────────────────────────────────────────────────────
# 規則型分類器
# ────────────────────────────────────────────────────────────────────

class TestRuleBasedClassifier:

    def test_empty_coords_returns_idle(self):
        result = rule_based_classify([])
        assert result.intent == GestureIntent.IDLE

    def test_single_point_returns_idle(self):
        result = rule_based_classify([(100, 100)])
        assert result.intent == GestureIntent.IDLE

    def test_stationary_points_return_pointing(self):
        # 幾乎不動的座標序列 → POINTING
        coords = [(100 + i*0.1, 200 + i*0.05) for i in range(20)]
        result = rule_based_classify(coords, variance_threshold=50.0)
        assert result.intent == GestureIntent.POINTING
        assert result.intent_score == INTENT_SCORE_MAP[GestureIntent.POINTING]

    def test_large_displacement_returns_transition(self):
        # 大幅位移 → TRANSITION
        coords = [(i * 20, 100) for i in range(15)]  # 從 x=0 到 x=280
        result = rule_based_classify(coords)
        assert result.intent == GestureIntent.TRANSITION

    def test_reversal_returns_emphasis(self):
        # 反覆來回 → EMPHASIS
        coords = []
        for i in range(15):
            x = 200 + (30 if i % 2 == 0 else -30)
            coords.append((x, 150))
        result = rule_based_classify(coords)
        assert result.intent in (GestureIntent.EMPHASIS, GestureIntent.WRITING)

    def test_result_probabilities_sum_to_one(self):
        coords = [(100, 100), (101, 101), (102, 102)]
        result = rule_based_classify(coords)
        assert abs(sum(result.probabilities) - 1.0) < 0.01

    def test_confidence_matches_max_probability(self):
        coords = [(100 + i*0.1, 100) for i in range(10)]
        result = rule_based_classify(coords)
        assert abs(result.confidence - max(result.probabilities)) < 0.01


# ────────────────────────────────────────────────────────────────────
# 意圖分數對應
# ────────────────────────────────────────────────────────────────────

class TestIntentScores:

    def test_pointing_has_highest_score(self):
        scores = list(INTENT_SCORE_MAP.values())
        assert INTENT_SCORE_MAP[GestureIntent.POINTING] == max(scores)

    def test_idle_has_lowest_score(self):
        scores = list(INTENT_SCORE_MAP.values())
        assert INTENT_SCORE_MAP[GestureIntent.IDLE] == min(scores)

    def test_all_scores_in_range(self):
        for intent, score in INTENT_SCORE_MAP.items():
            assert 0.0 <= score <= 1.0, f"{intent} score {score} out of range"

    def test_make_result_sets_correct_intent_score(self):
        result = _make_result(GestureIntent.POINTING, 0.9)
        assert result.intent_score == INTENT_SCORE_MAP[GestureIntent.POINTING]


# ────────────────────────────────────────────────────────────────────
# GestureClassifier 主介面
# ────────────────────────────────────────────────────────────────────

class TestGestureClassifier:

    def test_initializes_without_error(self):
        # 即使模型不存在，也應該初始化成功（降級為規則型）
        clf = GestureClassifier()
        assert clf.mode in ("gru", "rule")

    def test_classify_returns_gesture_result(self):
        clf = GestureClassifier()
        coords = [(100 + i, 200) for i in range(10)]
        result = clf.classify(coords, frame_w=640, frame_h=480)
        assert isinstance(result, GestureResult)

    def test_no_hand_returns_idle(self):
        clf = GestureClassifier()
        result = clf.classify([], hand_detected=False)
        assert result.intent == GestureIntent.IDLE

    def test_intent_score_in_range(self):
        clf = GestureClassifier()
        coords = [(100 + i*2, 150) for i in range(20)]
        result = clf.classify(coords)
        assert 0.0 <= result.intent_score <= 1.0

    def test_confidence_in_range(self):
        clf = GestureClassifier()
        coords = [(200, 200)] * 15
        result = clf.classify(coords)
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_length_is_5(self):
        clf = GestureClassifier()
        coords = [(100, 100)] * 10
        result = clf.classify(coords)
        assert len(result.probabilities) == 5

    @settings(max_examples=20)
    @given(
        n_coords=st.integers(min_value=0, max_value=30),
        frame_w=st.integers(min_value=320, max_value=1920),
        frame_h=st.integers(min_value=240, max_value=1080),
    )
    def test_classify_never_raises(self, n_coords, frame_w, frame_h):
        clf = GestureClassifier()
        coords = [(np.random.uniform(0, frame_w), np.random.uniform(0, frame_h))
                  for _ in range(n_coords)]
        result = clf.classify(coords, frame_w=frame_w, frame_h=frame_h)
        assert isinstance(result, GestureResult)
        assert 0.0 <= result.intent_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
