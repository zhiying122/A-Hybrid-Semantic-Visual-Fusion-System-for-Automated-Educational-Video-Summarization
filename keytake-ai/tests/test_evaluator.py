"""
Evaluator Property-Based Tests

**Validates: Requirements 4.1, 4.4**

Property 4: 誤報率計算正確性
- For any selected 片段列表、ground_truth 列表和 total_duration，
  Evaluator.compute_false_alarm_rate() 應回傳 [0.0, 1.0] 範圍內的值，
  且 evaluate_all() 的回傳結果應包含 'false_alarm_rate' 鍵。
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fusion.evaluator import compute_false_alarm_rate, compute_recall


# ============================================================================
# Strategies
# ============================================================================

def segment_strategy():
    """Generate a valid segment dict with start < end."""
    return st.builds(
        lambda s, length: {"start": s, "end": s + length},
        s=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        length=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )


def gt_strategy():
    """Generate a valid ground truth tuple (start, end) with start < end."""
    return st.builds(
        lambda s, length: (s, s + length),
        s=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        length=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )


# ============================================================================
# Property Tests for compute_false_alarm_rate
# ============================================================================

class TestFalseAlarmRateProperties:
    """
    Feature: keytake-ai-completion, Property 4: 誤報率計算正確性

    **Validates: Requirements 4.1, 4.4**
    """

    @settings(max_examples=10)
    @given(
        selected=st.lists(segment_strategy(), min_size=0, max_size=5),
        ground_truth=st.lists(gt_strategy(), min_size=0, max_size=5),
        total_duration=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_false_alarm_rate_always_in_valid_range(self, selected, ground_truth, total_duration):
        """
        Property: compute_false_alarm_rate() 回傳值永遠在 [0.0, 1.0] 範圍內

        **Validates: Requirements 4.1**
        """
        far = compute_false_alarm_rate(selected, ground_truth, total_duration)

        assert isinstance(far, float), f"FAR should be float, got {type(far)}"
        assert 0.0 <= far <= 1.0, f"FAR {far} out of range [0.0, 1.0]"

    @settings(max_examples=10)
    @given(
        ground_truth=st.lists(gt_strategy(), min_size=1, max_size=5),
        total_duration=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_no_selected_segments_gives_zero_far(self, ground_truth, total_duration):
        """
        Property: 沒有選中片段時，誤報率應為 0.0

        **Validates: Requirements 4.1**
        """
        far = compute_false_alarm_rate([], ground_truth, total_duration)
        assert far == 0.0, f"No selected segments should give FAR=0.0, got {far}"

    @settings(max_examples=10)
    @given(
        selected=st.lists(segment_strategy(), min_size=1, max_size=5),
        total_duration=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_no_ground_truth_all_selected_are_false_alarms(self, selected, total_duration):
        """
        Property: 沒有 ground truth 時，所有選中片段都是誤報，
        FAR = sum(selected durations) / total_duration (clamped to 1.0)

        **Validates: Requirements 4.1**
        """
        far = compute_false_alarm_rate(selected, [], total_duration)
        expected_sum = sum(seg["end"] - seg["start"] for seg in selected)
        expected = min(1.0, max(0.0, expected_sum / total_duration))

        assert abs(far - expected) < 1e-9, \
            f"FAR {far} != expected {expected}"

    @settings(max_examples=10)
    @given(
        total_duration=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_full_overlap_gives_zero_far(self, total_duration):
        """
        Property: 當所有選中片段完全與 ground truth 重疊時，FAR 應為 0.0

        **Validates: Requirements 4.1**
        """
        # Create selected segments that exactly match ground truth
        selected = [{"start": 10.0, "end": 20.0}, {"start": 30.0, "end": 40.0}]
        ground_truth = [(10.0, 20.0), (30.0, 40.0)]

        far = compute_false_alarm_rate(selected, ground_truth, total_duration)
        assert far == 0.0, f"Full overlap should give FAR=0.0, got {far}"

    def test_zero_total_duration_returns_zero(self):
        """
        Property: total_duration 為 0 時應回傳 0.0

        **Validates: Requirements 4.1**
        """
        selected = [{"start": 0, "end": 10}]
        ground_truth = [(20, 30)]

        far = compute_false_alarm_rate(selected, ground_truth, 0.0)
        assert far == 0.0

    def test_negative_total_duration_returns_zero(self):
        """
        Property: total_duration 為負數時應回傳 0.0

        **Validates: Requirements 4.1**
        """
        selected = [{"start": 0, "end": 10}]
        ground_truth = [(20, 30)]

        far = compute_false_alarm_rate(selected, ground_truth, -5.0)
        assert far == 0.0


# ============================================================================
# Property Tests for evaluate_all (mocking bert_score)
# ============================================================================

class TestEvaluateAllProperties:
    """
    Feature: keytake-ai-completion, Property 4: evaluate_all 回傳結果包含 false_alarm_rate

    **Validates: Requirements 4.4**
    """

    @settings(max_examples=10)
    @given(
        selected=st.lists(segment_strategy(), min_size=0, max_size=3),
        ground_truth=st.lists(gt_strategy(), min_size=0, max_size=3),
        total_duration=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_evaluate_all_contains_false_alarm_rate_key(self, selected, ground_truth, total_duration):
        """
        Property: evaluate_all() 回傳結果應包含 'false_alarm_rate' 鍵

        **Validates: Requirements 4.4**
        """
        # 正確 mock module-level bert_score_fn（v2 修復：evaluator 改為 module-level import）
        mock_f1 = MagicMock()
        mock_f1.mean.return_value = 0.75

        with patch('src.fusion.evaluator.bert_score_fn', return_value=(None, None, mock_f1)):
            from src.fusion.evaluator import evaluate_all
            result = evaluate_all(
                selected, ground_truth, total_duration,
                ["summary"], ["reference"]
            )

        assert 'false_alarm_rate' in result, \
            f"evaluate_all result missing 'false_alarm_rate' key: {result.keys()}"
        assert 0.0 <= result['false_alarm_rate'] <= 1.0, \
            f"false_alarm_rate {result['false_alarm_rate']} out of range [0.0, 1.0]"
    
    @settings(max_examples=10)
    @given(
        selected=st.lists(segment_strategy(), min_size=0, max_size=3),
        ground_truth=st.lists(gt_strategy(), min_size=0, max_size=3),
        total_duration=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_evaluate_all_contains_f1_score_key(self, selected, ground_truth, total_duration):
        """
        Property: evaluate_all() 回傳結果應包含 'f1_score' 鍵（v2 新增）

        **Validates: Requirements 4.4**
        """
        mock_f1 = MagicMock()
        mock_f1.mean.return_value = 0.75

        with patch('src.fusion.evaluator.bert_score_fn', return_value=(None, None, mock_f1)):
            from src.fusion.evaluator import evaluate_all
            result = evaluate_all(
                selected, ground_truth, total_duration,
                ["summary"], ["reference"]
            )

        assert 'f1_score' in result, \
            f"evaluate_all result missing 'f1_score' key: {result.keys()}"
        assert 0.0 <= result['f1_score'] <= 1.0, \
            f"f1_score {result['f1_score']} out of range [0.0, 1.0]"

    def test_evaluate_all_returns_all_expected_keys(self):
        """
        Unit test: evaluate_all() 應回傳所有六個指標鍵（v2 新增 precision、f1_score）

        **Validates: Requirements 4.4**
        """
        mock_f1 = MagicMock()
        mock_f1.mean.return_value = 0.8

        with patch('src.fusion.evaluator.bert_score_fn', return_value=(None, None, mock_f1)):
            from src.fusion.evaluator import evaluate_all
            result = evaluate_all(
                [{"start": 0, "end": 10}],
                [(0, 10)],
                100.0,
                ["summary text"],
                ["reference text"]
            )

        expected_keys = {'recall', 'precision', 'f1_score', 'false_alarm_rate', 'bert_score', 'time_saving_rate'}
        assert expected_keys == set(result.keys()), \
            f"Expected keys {expected_keys}, got {set(result.keys())}"


# ============================================================================
# Unit Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Unit tests for edge cases and boundary conditions"""

    def test_false_alarm_rate_no_overlap(self):
        """Test FAR when selected segments don't overlap with ground truth"""
        selected = [{"start": 0, "end": 10}]
        ground_truth = [(20, 30)]
        total_duration = 100.0

        far = compute_false_alarm_rate(selected, ground_truth, total_duration)
        assert abs(far - 0.1) < 1e-9

    def test_false_alarm_rate_partial_overlap(self):
        """Test FAR with partial overlap — overlapping segment is NOT a false alarm"""
        selected = [{"start": 0, "end": 15}]
        ground_truth = [(10, 20)]
        total_duration = 100.0

        # The segment overlaps with GT, so it's not counted as false alarm
        far = compute_false_alarm_rate(selected, ground_truth, total_duration)
        assert far == 0.0

    def test_recall_empty_ground_truth(self):
        """Test recall with empty ground truth returns 0.0"""
        recall = compute_recall([{"start": 0, "end": 10}], [])
        assert recall == 0.0

    def test_recall_perfect_match(self):
        """Test recall with perfect overlap"""
        selected = [{"start": 0, "end": 10}]
        ground_truth = [(0, 10)]

        recall = compute_recall(selected, ground_truth)
        assert recall == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
