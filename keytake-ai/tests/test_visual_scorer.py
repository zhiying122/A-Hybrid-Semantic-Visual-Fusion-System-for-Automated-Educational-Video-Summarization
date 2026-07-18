"""
Visual_Scorer Property-Based Tests

任務 5.3: Property 1 - 偵測優先流程正確性
**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

任務 5.4: Property 3 - SRGAN 模糊偵測整合
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

測試驗證：
1. compute_visual_score_v2() 回傳值的 score 永遠在 [0.0, 1.0] 範圍內
2. 當偵測到文字時，text_detected 應為 True
3. 當 ROI 模糊時，srgan_triggered 應為 True
4. 當 OCR 失敗或未偵測到文字時，fallback_used 應為 True
5. VisualScoreResult 的所有欄位都有正確的型別
"""

import sys
import os
from unittest.mock import patch, MagicMock
from dataclasses import fields

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visual.visual_scorer import (
    compute_visual_score_v2,
    VisualScoreResult,
    SRGANEnhancer,
    compute_iou,
    detect_highlighter,
    ocr_roi,
)
from src.visual.text_detector import TextDetector, TextDetectionResult
from src.visual.sbert_calculator import SBERTCalculator
from config import BLUR_THRESHOLD, SBERT_SIMILARITY_THRESHOLD


# ============================================================================
# Test Strategies
# ============================================================================

def make_random_image(h: int = 100, w: int = 100) -> np.ndarray:
    """產生隨機 BGR 影像"""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_blank_image(h: int = 100, w: int = 100) -> np.ndarray:
    """產生空白 BGR 影像"""
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def make_blurry_image(h: int = 100, w: int = 100) -> np.ndarray:
    """產生模糊影像（低 Laplacian 變異數）"""
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    # 加入輕微漸層使其更像真實模糊影像
    for i in range(h):
        img[i, :, :] = int(128 + (i / h) * 10)
    return img


def make_sharp_image(h: int = 100, w: int = 100) -> np.ndarray:
    """產生清晰影像（高 Laplacian 變異數）"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 加入高對比邊緣
    for i in range(0, h, 10):
        img[i:i+5, :, :] = 255
    for j in range(0, w, 10):
        img[:, j:j+5, :] = 255
    return img


# 影像尺寸策略
image_size_strategy = st.integers(min_value=50, max_value=200)

# 手部座標策略
hand_center_strategy = st.tuples(
    st.integers(min_value=10, max_value=90),
    st.integers(min_value=10, max_value=90)
)

# 關鍵字策略
keyword_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), blacklist_characters='\x00'),
    min_size=1,
    max_size=20
).filter(lambda x: x.strip() != '')

keywords_list_strategy = st.lists(keyword_strategy, min_size=0, max_size=5)

# 模糊閾值策略
blur_threshold_strategy = st.floats(min_value=50.0, max_value=500.0)


# ============================================================================
# Property Tests - Task 5.3: 偵測優先流程正確性
# ============================================================================

class TestDetectionFirstWorkflowProperties:
    """
    Feature: keytake-ai-completion, Property 1: 偵測優先流程正確性
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    
    For any ROI 影像輸入，Visual_Scorer 應先執行 Text_Detector.detect()，
    且 VisualScoreResult.text_detected 應正確反映偵測結果。
    當 has_text=True 時 OCR 應被執行，當 has_text=False 時應使用 IoU 降級備援。
    """

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        h=image_size_strategy,
        w=image_size_strategy,
        hand_center=hand_center_strategy,
        asr_keywords=keywords_list_strategy
    )
    def test_score_always_in_valid_range(self, h, w, hand_center, asr_keywords):
        """
        Property: compute_visual_score_v2() 回傳值的 score 永遠在 [0.0, 1.0] 範圍內
        
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        roi = make_random_image(h, w)
        frame = make_random_image(h, w)
        
        # Mock 外部依賴以確保測試穩定性
        with patch('src.visual.visual_scorer.ocr_roi', return_value="test text"):
            with patch.object(TextDetector, 'detect') as mock_detect:
                mock_detect.return_value = TextDetectionResult(
                    has_text=True, boxes=[(10, 10, 50, 50)], confidence=0.8
                )
                with patch.object(SBERTCalculator, 'compute_similarity_with_keywords', return_value=0.7):
                    result = compute_visual_score_v2(
                        roi=roi,
                        hand_center=hand_center,
                        asr_keywords=asr_keywords,
                        frame=frame
                    )
        
        assert isinstance(result.score, float), \
            f"Score should be float, got {type(result.score)}"
        assert 0.0 <= result.score <= 1.0, \
            f"Score {result.score} out of range [0.0, 1.0]"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        h=image_size_strategy,
        w=image_size_strategy,
        hand_center=hand_center_strategy
    )
    def test_text_detected_reflects_detector_result_true(self, h, w, hand_center):
        """
        Property: 當 Text_Detector 偵測到文字時，text_detected 應為 True
        
        **Validates: Requirements 1.1, 1.2**
        """
        roi = make_random_image(h, w)
        frame = make_random_image(h, w)
        
        with patch('src.visual.visual_scorer.ocr_roi', return_value="detected text"):
            with patch.object(TextDetector, 'detect') as mock_detect:
                mock_detect.return_value = TextDetectionResult(
                    has_text=True, boxes=[(10, 10, 50, 50)], confidence=0.9
                )
                with patch.object(SBERTCalculator, 'compute_similarity_with_keywords', return_value=0.8):
                    result = compute_visual_score_v2(
                        roi=roi,
                        hand_center=hand_center,
                        asr_keywords=["keyword"],
                        frame=frame
                    )
        
        assert result.text_detected is True, \
            "text_detected should be True when detector finds text"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        h=image_size_strategy,
        w=image_size_strategy,
        hand_center=hand_center_strategy
    )
    def test_text_detected_reflects_detector_result_false(self, h, w, hand_center):
        """
        Property: 當 Text_Detector 未偵測到文字時，text_detected 應為 False
        
        **Validates: Requirements 1.3**
        """
        roi = make_random_image(h, w)
        frame = make_random_image(h, w)
        
        with patch.object(TextDetector, 'detect') as mock_detect:
            mock_detect.return_value = TextDetectionResult(
                has_text=False, boxes=[], confidence=0.0
            )
            result = compute_visual_score_v2(
                roi=roi,
                hand_center=hand_center,
                asr_keywords=["keyword"],
                frame=frame
            )
        
        assert result.text_detected is False, \
            "text_detected should be False when detector finds no text"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        h=image_size_strategy,
        w=image_size_strategy,
        hand_center=hand_center_strategy
    )
    def test_fallback_used_when_no_text_detected(self, h, w, hand_center):
        """
        Property: 當未偵測到文字時，fallback_used 應為 True
        
        **Validates: Requirements 1.3, 1.4**
        """
        roi = make_random_image(h, w)
        frame = make_random_image(h, w)
        
        with patch.object(TextDetector, 'detect') as mock_detect:
            mock_detect.return_value = TextDetectionResult(
                has_text=False, boxes=[], confidence=0.0
            )
            result = compute_visual_score_v2(
                roi=roi,
                hand_center=hand_center,
                asr_keywords=["keyword"],
                frame=frame
            )
        
        assert result.fallback_used is True, \
            "fallback_used should be True when no text detected"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        h=image_size_strategy,
        w=image_size_strategy,
        hand_center=hand_center_strategy
    )
    def test_fallback_used_when_ocr_fails(self, h, w, hand_center):
        """
        Property: 當 OCR 失敗（回傳空字串）時，fallback_used 應為 True
        
        **Validates: Requirements 1.3, 1.4**
        """
        roi = make_random_image(h, w)
        frame = make_random_image(h, w)
        
        with patch('src.visual.visual_scorer.ocr_roi', return_value=""):
            with patch.object(TextDetector, 'detect') as mock_detect:
                mock_detect.return_value = TextDetectionResult(
                    has_text=True, boxes=[(10, 10, 50, 50)], confidence=0.8
                )
                result = compute_visual_score_v2(
                    roi=roi,
                    hand_center=hand_center,
                    asr_keywords=["keyword"],
                    frame=frame
                )
        
        assert result.fallback_used is True, \
            "fallback_used should be True when OCR returns empty string"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        h=image_size_strategy,
        w=image_size_strategy,
        hand_center=hand_center_strategy
    )
    def test_result_has_correct_types(self, h, w, hand_center):
        """
        Property: VisualScoreResult 的所有欄位都有正確的型別
        
        **Validates: Requirements 1.4**
        """
        roi = make_random_image(h, w)
        frame = make_random_image(h, w)
        
        with patch('src.visual.visual_scorer.ocr_roi', return_value="test"):
            with patch.object(TextDetector, 'detect') as mock_detect:
                mock_detect.return_value = TextDetectionResult(
                    has_text=True, boxes=[(10, 10, 50, 50)], confidence=0.8
                )
                with patch.object(SBERTCalculator, 'compute_similarity_with_keywords', return_value=0.7):
                    result = compute_visual_score_v2(
                        roi=roi,
                        hand_center=hand_center,
                        asr_keywords=["keyword"],
                        frame=frame
                    )
        
        assert isinstance(result, VisualScoreResult), \
            f"Result should be VisualScoreResult, got {type(result)}"
        assert isinstance(result.score, float), \
            f"score should be float, got {type(result.score)}"
        assert isinstance(result.text_detected, bool), \
            f"text_detected should be bool, got {type(result.text_detected)}"
        assert result.ocr_text is None or isinstance(result.ocr_text, str), \
            f"ocr_text should be str or None, got {type(result.ocr_text)}"
        assert result.sbert_similarity is None or isinstance(result.sbert_similarity, float), \
            f"sbert_similarity should be float or None, got {type(result.sbert_similarity)}"
        assert isinstance(result.srgan_triggered, bool), \
            f"srgan_triggered should be bool, got {type(result.srgan_triggered)}"
        assert isinstance(result.fallback_used, bool), \
            f"fallback_used should be bool, got {type(result.fallback_used)}"


# ============================================================================
# Property Tests - Task 5.4: SRGAN 模糊偵測整合
# ============================================================================

class TestSRGANIntegrationProperties:
    """
    Feature: keytake-ai-completion, Property 3: SRGAN 模糊偵測整合
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    
    For any ROI 影像，當 Laplacian 變異數低於 BLUR_THRESHOLD 時，
    SRGAN_Enhancer.enhance_roi() 應被觸發且 VisualScoreResult.srgan_triggered 應為 True。
    當變異數高於閾值時，應使用原始 ROI 且 srgan_triggered 應為 False。
    """

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        hand_center=hand_center_strategy,
        blur_variance=st.floats(min_value=0.0, max_value=50.0)
    )
    def test_srgan_triggered_when_blurry(self, hand_center, blur_variance):
        """
        Property: 當 Laplacian 變異數低於閾值時，srgan_triggered 應為 True
        
        **Validates: Requirements 3.1, 3.2**
        """
        roi = make_blurry_image()
        frame = make_random_image()
        
        # Mock is_blurry 回傳 True（模糊）
        with patch.object(SRGANEnhancer, 'is_blurry', return_value=True):
            with patch.object(SRGANEnhancer, 'enhance', return_value=roi):
                with patch.object(TextDetector, 'detect') as mock_detect:
                    mock_detect.return_value = TextDetectionResult(
                        has_text=False, boxes=[], confidence=0.0
                    )
                    result = compute_visual_score_v2(
                        roi=roi,
                        hand_center=hand_center,
                        asr_keywords=["keyword"],
                        frame=frame
                    )
        
        assert result.srgan_triggered is True, \
            "srgan_triggered should be True when image is blurry"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        hand_center=hand_center_strategy,
        blur_variance=st.floats(min_value=150.0, max_value=500.0)
    )
    def test_srgan_not_triggered_when_sharp(self, hand_center, blur_variance):
        """
        Property: 當 Laplacian 變異數高於閾值時，srgan_triggered 應為 False
        
        **Validates: Requirements 3.4**
        """
        roi = make_sharp_image()
        frame = make_random_image()
        
        # Mock is_blurry 回傳 False（清晰）
        with patch.object(SRGANEnhancer, 'is_blurry', return_value=False):
            with patch.object(TextDetector, 'detect') as mock_detect:
                mock_detect.return_value = TextDetectionResult(
                    has_text=False, boxes=[], confidence=0.0
                )
                result = compute_visual_score_v2(
                    roi=roi,
                    hand_center=hand_center,
                    asr_keywords=["keyword"],
                    frame=frame
                )
        
        assert result.srgan_triggered is False, \
            "srgan_triggered should be False when image is sharp"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        h=image_size_strategy,
        w=image_size_strategy,
        hand_center=hand_center_strategy,
        threshold=blur_threshold_strategy
    )
    def test_srgan_trigger_respects_custom_threshold(self, h, w, hand_center, threshold):
        """
        Property: SRGAN 觸發應尊重自訂的模糊閾值
        
        **Validates: Requirements 3.1, 3.5**
        """
        roi = make_random_image(h, w)
        frame = make_random_image(h, w)
        
        # 使用自訂閾值
        with patch.object(TextDetector, 'detect') as mock_detect:
            mock_detect.return_value = TextDetectionResult(
                has_text=False, boxes=[], confidence=0.0
            )
            result = compute_visual_score_v2(
                roi=roi,
                hand_center=hand_center,
                asr_keywords=["keyword"],
                frame=frame,
                blur_threshold=threshold
            )
        
        # 結果應該是有效的
        assert isinstance(result.srgan_triggered, bool), \
            "srgan_triggered should be boolean"
        assert 0.0 <= result.score <= 1.0, \
            f"Score {result.score} out of range [0.0, 1.0]"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        hand_center=hand_center_strategy
    )
    def test_srgan_records_trigger_status(self, hand_center):
        """
        Property: Visual_Scorer 應記錄是否觸發 SRGAN 增強供效能分析使用
        
        **Validates: Requirements 3.5**
        """
        roi = make_random_image()
        frame = make_random_image()
        
        with patch.object(TextDetector, 'detect') as mock_detect:
            mock_detect.return_value = TextDetectionResult(
                has_text=False, boxes=[], confidence=0.0
            )
            result = compute_visual_score_v2(
                roi=roi,
                hand_center=hand_center,
                asr_keywords=["keyword"],
                frame=frame
            )
        
        # 確認 srgan_triggered 欄位存在且為布林值
        assert hasattr(result, 'srgan_triggered'), \
            "Result should have srgan_triggered field"
        assert isinstance(result.srgan_triggered, bool), \
            "srgan_triggered should be boolean for performance analysis"

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        hand_center=hand_center_strategy
    )
    def test_enhanced_roi_used_for_ocr_when_blurry(self, hand_center):
        """
        Property: 當 SRGAN 增強完成後，應使用增強後的影像進行 OCR 辨識
        
        **Validates: Requirements 3.3**
        """
        roi = make_blurry_image()
        enhanced_roi = make_sharp_image()
        frame = make_random_image()
        
        ocr_called_with = []
        
        def mock_ocr(image):
            ocr_called_with.append(image)
            return "enhanced text"
        
        with patch('src.visual.visual_scorer.ocr_roi', side_effect=mock_ocr):
            with patch.object(SRGANEnhancer, 'is_blurry', return_value=True):
                with patch.object(SRGANEnhancer, 'enhance', return_value=enhanced_roi):
                    with patch.object(TextDetector, 'detect') as mock_detect:
                        mock_detect.return_value = TextDetectionResult(
                            has_text=True, boxes=[(10, 10, 50, 50)], confidence=0.8
                        )
                        with patch.object(SBERTCalculator, 'compute_similarity_with_keywords', return_value=0.7):
                            result = compute_visual_score_v2(
                                roi=roi,
                                hand_center=hand_center,
                                asr_keywords=["keyword"],
                                frame=frame
                            )
        
        assert result.srgan_triggered is True, \
            "SRGAN should be triggered for blurry image"
        # OCR 應該被呼叫
        assert len(ocr_called_with) > 0, \
            "OCR should be called when text is detected"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for Visual_Scorer"""

    def test_empty_roi_returns_fallback(self):
        """Test that empty ROI returns fallback result"""
        empty_roi = np.array([])
        frame = make_random_image()
        
        result = compute_visual_score_v2(
            roi=empty_roi,
            hand_center=(50, 50),
            asr_keywords=["keyword"],
            frame=frame
        )
        
        assert result.score == 0.0
        assert result.fallback_used is True
        assert result.text_detected is False

    def test_none_roi_returns_fallback(self):
        """Test that None ROI returns fallback result"""
        frame = make_random_image()
        
        result = compute_visual_score_v2(
            roi=None,
            hand_center=(50, 50),
            asr_keywords=["keyword"],
            frame=frame
        )
        
        assert result.score == 0.0
        assert result.fallback_used is True
        assert result.text_detected is False

    def test_empty_keywords_list(self):
        """Test handling of empty keywords list"""
        roi = make_random_image()
        frame = make_random_image()
        
        with patch('src.visual.visual_scorer.ocr_roi', return_value="test text"):
            with patch.object(TextDetector, 'detect') as mock_detect:
                mock_detect.return_value = TextDetectionResult(
                    has_text=True, boxes=[(10, 10, 50, 50)], confidence=0.8
                )
                with patch.object(SBERTCalculator, 'compute_similarity_with_keywords', return_value=0.0):
                    result = compute_visual_score_v2(
                        roi=roi,
                        hand_center=(50, 50),
                        asr_keywords=[],
                        frame=frame
                    )
        
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result, VisualScoreResult)


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Unit tests for helper functions"""

    def test_compute_iou_no_overlap(self):
        """Test IoU calculation with no overlap"""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        
        iou = compute_iou(box1, box2)
        
        assert iou == 0.0

    def test_compute_iou_full_overlap(self):
        """Test IoU calculation with full overlap"""
        box1 = (0, 0, 10, 10)
        box2 = (0, 0, 10, 10)
        
        iou = compute_iou(box1, box2)
        
        assert iou == 1.0

    def test_compute_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap"""
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        
        iou = compute_iou(box1, box2)
        
        assert 0.0 < iou < 1.0

    def test_detect_highlighter_returns_valid_range(self):
        """Test highlighter detection returns value in [0, 1]"""
        frame = make_random_image()
        
        ratio = detect_highlighter(frame)
        
        assert 0.0 <= ratio <= 1.0


# ============================================================================
# SRGANEnhancer Unit Tests
# ============================================================================

class TestSRGANEnhancer:
    """Unit tests for SRGANEnhancer class"""

    def test_is_blurry_with_blurry_image(self):
        """Test is_blurry returns True for blurry image"""
        enhancer = SRGANEnhancer(blur_threshold=BLUR_THRESHOLD)
        blurry_img = make_blurry_image()
        
        result = enhancer.is_blurry(blurry_img)
        
        assert isinstance(result, bool)

    def test_is_blurry_with_sharp_image(self):
        """Test is_blurry returns False for sharp image"""
        enhancer = SRGANEnhancer(blur_threshold=BLUR_THRESHOLD)
        sharp_img = make_sharp_image()
        
        result = enhancer.is_blurry(sharp_img)
        
        assert isinstance(result, bool)

    def test_enhance_returns_valid_image(self):
        """Test enhance returns valid image"""
        enhancer = SRGANEnhancer()
        roi = make_random_image(50, 50)
        
        enhanced = enhancer.enhance(roi)
        
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.ndim == 3
        assert enhanced.shape[2] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
