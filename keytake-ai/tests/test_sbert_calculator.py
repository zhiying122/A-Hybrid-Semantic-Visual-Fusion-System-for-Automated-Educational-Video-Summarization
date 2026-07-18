"""
SBERT_Calculator Property-Based Tests

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: SBERT 語意計算與降級備援
- For any OCR 辨識結果和 ASR 關鍵字列表，當 OCR 成功（ocr_text 非空）時，
  SBERT_Calculator.compute_similarity() 應被呼叫且回傳值在 [0.0, 1.0] 範圍內。
- 當 OCR 失敗時，應使用 IoU 計算作為降級備援，且 VisualScoreResult.fallback_used 應為 True。
"""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visual.sbert_calculator import (
    SBERTCalculator,
    compute_char_overlap,
    SBERT_AVAILABLE
)


# ============================================================================
# Strategies for Property-Based Testing
# ============================================================================

# 產生非空文字字串的策略
text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'S'),
        blacklist_characters='\x00'
    ),
    min_size=1,
    max_size=200
).filter(lambda x: x.strip() != '')

# 產生可能為空的文字字串策略
optional_text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'S'),
        blacklist_characters='\x00'
    ),
    min_size=0,
    max_size=200
)

# 產生關鍵字列表的策略
keywords_strategy = st.lists(
    text_strategy,
    min_size=0,
    max_size=10
)

# 產生非空關鍵字列表的策略
non_empty_keywords_strategy = st.lists(
    text_strategy,
    min_size=1,
    max_size=10
)


# ============================================================================
# Property Tests for compute_similarity()
# ============================================================================

class TestSBERTCalculatorProperties:
    """
    Feature: keytake-ai-completion, Property 2: SBERT 語意計算與降級備援
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ocr_text=text_strategy,
        transcript_text=text_strategy
    )
    def test_similarity_always_in_valid_range(self, ocr_text, transcript_text):
        """
        Property: compute_similarity() 回傳值永遠在 [0.0, 1.0] 範圍內
        
        **Validates: Requirements 2.1, 2.2**
        
        For any non-empty OCR text and transcript text,
        the similarity score should always be in [0.0, 1.0].
        """
        calculator = SBERTCalculator()
        similarity = calculator.compute_similarity(ocr_text, transcript_text)
        
        assert isinstance(similarity, float), \
            f"Similarity should be float, got {type(similarity)}"
        assert 0.0 <= similarity <= 1.0, \
            f"Similarity {similarity} out of range [0.0, 1.0]"

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(text=text_strategy)
    def test_identical_text_high_similarity(self, text):
        """
        Property: 相同文字的相似度應為 1.0（或接近 1.0）
        
        **Validates: Requirements 2.1, 2.2**
        
        For any non-empty text, computing similarity with itself
        should return a value close to 1.0.
        """
        calculator = SBERTCalculator()
        similarity = calculator.compute_similarity(text, text)
        
        # 相同文字的相似度應該非常高（接近 1.0）
        # 使用 0.9 作為閾值，因為某些特殊字元可能導致輕微差異
        assert similarity >= 0.9, \
            f"Identical text similarity {similarity} should be >= 0.9"

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ocr_text=optional_text_strategy,
        transcript_text=optional_text_strategy
    )
    def test_empty_input_returns_zero(self, ocr_text, transcript_text):
        """
        Property: 空字串輸入應回傳 0.0
        
        **Validates: Requirements 2.3, 2.4**
        
        When either input is empty or whitespace-only,
        the similarity should be 0.0.
        """
        # 只測試至少有一個輸入為空的情況
        assume(not ocr_text.strip() or not transcript_text.strip())
        
        calculator = SBERTCalculator()
        similarity = calculator.compute_similarity(ocr_text, transcript_text)
        
        assert similarity == 0.0, \
            f"Empty input should return 0.0, got {similarity}"

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ocr_text=text_strategy,
        asr_keywords=non_empty_keywords_strategy
    )
    def test_keywords_similarity_in_valid_range(self, ocr_text, asr_keywords):
        """
        Property: compute_similarity_with_keywords() 回傳值在 [0.0, 1.0] 範圍內
        
        **Validates: Requirements 2.1, 2.2**
        
        For any OCR text and non-empty keyword list,
        the similarity score should be in [0.0, 1.0].
        """
        calculator = SBERTCalculator()
        similarity = calculator.compute_similarity_with_keywords(ocr_text, asr_keywords)
        
        assert isinstance(similarity, float), \
            f"Similarity should be float, got {type(similarity)}"
        assert 0.0 <= similarity <= 1.0, \
            f"Similarity {similarity} out of range [0.0, 1.0]"

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(ocr_text=text_strategy)
    def test_empty_keywords_returns_zero(self, ocr_text):
        """
        Property: 空關鍵字列表應回傳 0.0
        
        **Validates: Requirements 2.3, 2.4**
        
        When the keyword list is empty, similarity should be 0.0.
        """
        calculator = SBERTCalculator()
        similarity = calculator.compute_similarity_with_keywords(ocr_text, [])
        
        assert similarity == 0.0, \
            f"Empty keywords should return 0.0, got {similarity}"


# ============================================================================
# Property Tests for Fallback Mode
# ============================================================================

class TestFallbackModeProperties:
    """
    Feature: keytake-ai-completion, Property 2: 降級備援模式正確運作
    
    **Validates: Requirements 2.3, 2.4**
    """

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        text1=text_strategy,
        text2=text_strategy
    )
    def test_char_overlap_in_valid_range(self, text1, text2):
        """
        Property: 字元重疊率（降級備援）回傳值在 [0.0, 1.0] 範圍內
        
        **Validates: Requirements 2.4**
        
        For any two non-empty strings, character overlap ratio
        should be in [0.0, 1.0].
        """
        overlap = compute_char_overlap(text1, text2)
        
        assert isinstance(overlap, float), \
            f"Overlap should be float, got {type(overlap)}"
        assert 0.0 <= overlap <= 1.0, \
            f"Overlap {overlap} out of range [0.0, 1.0]"

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(text=text_strategy)
    def test_char_overlap_identical_text_is_one(self, text):
        """
        Property: 相同文字的字元重疊率應為 1.0
        
        **Validates: Requirements 2.4**
        
        For any non-empty text, character overlap with itself should be 1.0.
        """
        overlap = compute_char_overlap(text, text)
        
        assert overlap == 1.0, \
            f"Identical text overlap should be 1.0, got {overlap}"

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(text=text_strategy)
    def test_char_overlap_empty_input_returns_zero(self, text):
        """
        Property: 空字串的字元重疊率應為 0.0
        
        **Validates: Requirements 2.4**
        
        When either input is empty, character overlap should be 0.0.
        """
        assert compute_char_overlap("", text) == 0.0
        assert compute_char_overlap(text, "") == 0.0
        assert compute_char_overlap("", "") == 0.0

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ocr_text=text_strategy,
        transcript_text=text_strategy
    )
    def test_fallback_mode_works_correctly(self, ocr_text, transcript_text):
        """
        Property: 降級備援模式正確運作
        
        **Validates: Requirements 2.3, 2.4**
        
        When SBERT model is unavailable, the calculator should
        fall back to character overlap and still return valid results.
        """
        # 強制使用降級備援模式
        with patch.object(SBERTCalculator, '_load_model') as mock_load:
            calculator = SBERTCalculator()
            calculator.fallback_mode = True
            calculator.model = None
            
            similarity = calculator.compute_similarity(ocr_text, transcript_text)
            
            # 驗證回傳值在有效範圍內
            assert 0.0 <= similarity <= 1.0, \
                f"Fallback similarity {similarity} out of range"
            
            # 驗證降級模式標記
            assert calculator.is_fallback_mode is True

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ocr_text=text_strategy,
        transcript_text=text_strategy
    )
    def test_fallback_on_sbert_exception(self, ocr_text, transcript_text):
        """
        Property: SBERT 計算失敗時應自動降級
        
        **Validates: Requirements 2.3, 2.4**
        
        When SBERT computation fails, the calculator should
        automatically fall back to character overlap.
        """
        calculator = SBERTCalculator()
        
        # 模擬 SBERT 計算失敗
        if calculator.model is not None:
            with patch.object(calculator.model, 'encode', side_effect=Exception("SBERT error")):
                similarity = calculator.compute_similarity(ocr_text, transcript_text)
                
                # 應該回傳有效的降級結果
                assert 0.0 <= similarity <= 1.0, \
                    f"Fallback similarity {similarity} out of range"


# ============================================================================
# Property Tests for Cosine Similarity Normalization
# ============================================================================

class TestCosineSimilarityProperties:
    """
    Feature: keytake-ai-completion, Property 2: 餘弦相似度正規化
    
    **Validates: Requirements 2.2**
    """

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        vec1=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=10, max_size=10),
        vec2=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=10, max_size=10)
    )
    def test_cosine_similarity_normalized_range(self, vec1, vec2):
        """
        Property: 餘弦相似度正規化後應在 [0.0, 1.0] 範圍內
        
        **Validates: Requirements 2.2**
        
        For any two embedding vectors, the normalized cosine similarity
        should be in [0.0, 1.0].
        """
        calculator = SBERTCalculator()
        
        embedding1 = np.array(vec1, dtype=np.float32)
        embedding2 = np.array(vec2, dtype=np.float32)
        
        similarity = calculator._compute_cosine_similarity(embedding1, embedding2)
        
        assert 0.0 <= similarity <= 1.0, \
            f"Normalized cosine similarity {similarity} out of range [0.0, 1.0]"

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        vec=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=10, max_size=10)
    )
    def test_cosine_similarity_with_self_is_one(self, vec):
        """
        Property: 向量與自身的餘弦相似度應為 1.0
        
        **Validates: Requirements 2.2**
        
        For any non-zero vector, cosine similarity with itself should be 1.0.
        """
        calculator = SBERTCalculator()
        
        embedding = np.array(vec, dtype=np.float32)
        
        # 跳過零向量
        if np.linalg.norm(embedding) == 0:
            return
        
        similarity = calculator._compute_cosine_similarity(embedding, embedding)
        
        assert abs(similarity - 1.0) < 1e-5, \
            f"Self-similarity should be 1.0, got {similarity}"

    def test_cosine_similarity_zero_vector_returns_zero(self):
        """
        Property: 零向量的餘弦相似度應為 0.0
        
        **Validates: Requirements 2.2**
        
        When either vector is zero, cosine similarity should be 0.0.
        """
        calculator = SBERTCalculator()
        
        zero_vec = np.zeros(10, dtype=np.float32)
        non_zero_vec = np.ones(10, dtype=np.float32)
        
        assert calculator._compute_cosine_similarity(zero_vec, non_zero_vec) == 0.0
        assert calculator._compute_cosine_similarity(non_zero_vec, zero_vec) == 0.0
        assert calculator._compute_cosine_similarity(zero_vec, zero_vec) == 0.0


# ============================================================================
# Unit Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Unit tests for edge cases and boundary conditions"""

    def test_whitespace_only_input(self):
        """Test that whitespace-only input returns 0.0"""
        calculator = SBERTCalculator()
        
        assert calculator.compute_similarity("   ", "hello") == 0.0
        assert calculator.compute_similarity("hello", "   ") == 0.0
        assert calculator.compute_similarity("\t\n", "\t\n") == 0.0

    def test_special_characters(self):
        """Test handling of special characters"""
        calculator = SBERTCalculator()
        
        similarity = calculator.compute_similarity("!@#$%", "!@#$%")
        assert 0.0 <= similarity <= 1.0

    def test_unicode_text(self):
        """Test handling of Unicode text (Chinese, etc.)"""
        calculator = SBERTCalculator()
        
        # 中文文字測試
        similarity = calculator.compute_similarity("機器學習", "深度學習")
        assert 0.0 <= similarity <= 1.0
        
        # 相同中文文字
        similarity = calculator.compute_similarity("人工智慧", "人工智慧")
        assert similarity >= 0.9

    def test_mixed_language_text(self):
        """Test handling of mixed language text"""
        calculator = SBERTCalculator()
        
        similarity = calculator.compute_similarity(
            "Machine Learning 機器學習",
            "Deep Learning 深度學習"
        )
        assert 0.0 <= similarity <= 1.0

    def test_very_long_text(self):
        """Test handling of very long text"""
        calculator = SBERTCalculator()
        
        long_text = "word " * 1000
        similarity = calculator.compute_similarity(long_text, long_text)
        assert 0.0 <= similarity <= 1.0

    def test_is_fallback_mode_property(self):
        """Test is_fallback_mode property"""
        calculator = SBERTCalculator()
        
        # 屬性應該是布林值
        assert isinstance(calculator.is_fallback_mode, bool)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for SBERT Calculator"""

    def test_full_workflow_with_keywords(self):
        """Test complete workflow with keyword list"""
        calculator = SBERTCalculator()
        
        ocr_text = "機器學習演算法"
        keywords = ["機器學習", "深度學習", "神經網路"]
        
        similarity = calculator.compute_similarity_with_keywords(ocr_text, keywords)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_calculator_initialization(self):
        """Test calculator initialization"""
        calculator = SBERTCalculator()
        
        # 應該成功初始化
        assert calculator is not None
        assert isinstance(calculator.fallback_mode, bool)

    def test_custom_model_name(self):
        """Test initialization with custom model name"""
        # 使用無效的模型名稱應該觸發降級模式
        calculator = SBERTCalculator(model_name="invalid-model-name")
        
        # 應該進入降級模式
        assert calculator.is_fallback_mode is True
        
        # 但仍然可以計算相似度
        similarity = calculator.compute_similarity("test", "test")
        assert 0.0 <= similarity <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
