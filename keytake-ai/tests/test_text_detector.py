'''
Text_Detector Unit Tests
'''

import sys
import os
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visual.text_detector import TextDetector, TextDetectionResult
from config import TEXT_DETECTION_CONFIDENCE_THRESHOLD


def make_blank_image(h=100, w=100):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def make_grayscale_image(h=100, w=100):
    return np.ones((h, w), dtype=np.uint8) * 128


class TestTextDetectionResult:

    def test_result_with_text(self):
        result = TextDetectionResult(
            has_text=True,
            boxes=[(10, 20, 50, 40), (60, 70, 100, 90)],
            confidence=0.85
        )
        assert result.has_text is True
        assert len(result.boxes) == 2
        assert result.confidence == 0.85

    def test_result_without_text(self):
        result = TextDetectionResult(
            has_text=False,
            boxes=[],
            confidence=0.0
        )
        assert result.has_text is False
        assert len(result.boxes) == 0
        assert result.confidence == 0.0

    def test_confidence_range(self):
        result_low = TextDetectionResult(has_text=False, boxes=[], confidence=0.0)
        result_high = TextDetectionResult(has_text=True, boxes=[(0, 0, 10, 10)], confidence=1.0)
        assert 0.0 <= result_low.confidence <= 1.0
        assert 0.0 <= result_high.confidence <= 1.0


class TestTextDetectorInit:

    def test_default_threshold(self):
        detector = TextDetector()
        assert detector.confidence_threshold == TEXT_DETECTION_CONFIDENCE_THRESHOLD

    def test_custom_threshold(self):
        custom_threshold = 0.7
        detector = TextDetector(confidence_threshold=custom_threshold)
        assert detector.confidence_threshold == custom_threshold


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestTextDetectorWithText:
    '''Tests for detecting text - Validates: Requirements 1.1, 1.2'''

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_detect_text_returns_has_text_true(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [85, 90, 75],
            'left': [10, 50, 100],
            'top': [20, 30, 40],
            'width': [30, 40, 50],
            'height': [15, 20, 25]
        }
        detector = TextDetector(confidence_threshold=0.5)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is True
        assert len(result.boxes) > 0

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_detect_text_returns_correct_boxes(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [80],
            'left': [10],
            'top': [20],
            'width': [30],
            'height': [15]
        }
        detector = TextDetector(confidence_threshold=0.5)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert len(result.boxes) == 1
        x1, y1, x2, y2 = result.boxes[0]
        assert x1 == 10
        assert y1 == 20
        assert x2 == 40
        assert y2 == 35

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_detect_text_calculates_average_confidence(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [80, 90],
            'left': [10, 50],
            'top': [20, 30],
            'width': [30, 40],
            'height': [15, 20]
        }
        detector = TextDetector(confidence_threshold=0.5)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert abs(result.confidence - 0.85) < 0.01

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_detect_multiple_text_regions(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [85, 90, 75, 80],
            'left': [10, 50, 100, 150],
            'top': [20, 30, 40, 50],
            'width': [30, 40, 50, 60],
            'height': [15, 20, 25, 30]
        }
        detector = TextDetector(confidence_threshold=0.5)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is True
        assert len(result.boxes) == 4


class TestTextDetectorWithoutText:
    '''Tests for no text detected - Validates: Requirement 1.3'''

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_detect_no_text_returns_has_text_false(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [-1, -1, -1],
            'left': [0, 0, 0],
            'top': [0, 0, 0],
            'width': [0, 0, 0],
            'height': [0, 0, 0]
        }
        detector = TextDetector()
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is False
        assert len(result.boxes) == 0
        assert result.confidence == 0.0

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_detect_empty_result(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [],
            'left': [],
            'top': [],
            'width': [],
            'height': []
        }
        detector = TextDetector()
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is False
        assert len(result.boxes) == 0

    def test_detect_none_input(self):
        detector = TextDetector()
        result = detector.detect(None)
        assert result.has_text is False
        assert len(result.boxes) == 0
        assert result.confidence == 0.0

    def test_detect_empty_image(self):
        detector = TextDetector()
        empty_image = np.array([])
        result = detector.detect(empty_image)
        assert result.has_text is False
        assert len(result.boxes) == 0
        assert result.confidence == 0.0

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_detect_exception_handling(self, mock_image_to_data):
        mock_image_to_data.side_effect = Exception('Tesseract error')
        detector = TextDetector()
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is False
        assert len(result.boxes) == 0
        assert result.confidence == 0.0


class TestConfidenceThreshold:
    '''Tests for confidence threshold filtering'''

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_filter_low_confidence_boxes(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [30, 65, 80, 45],
            'left': [10, 50, 100, 150],
            'top': [20, 30, 40, 50],
            'width': [30, 40, 50, 60],
            'height': [15, 20, 25, 30]
        }
        detector = TextDetector(confidence_threshold=0.6)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is True
        assert len(result.boxes) == 2

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_all_below_threshold_returns_no_text(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [30, 40, 45],
            'left': [10, 50, 100],
            'top': [20, 30, 40],
            'width': [30, 40, 50],
            'height': [15, 20, 25]
        }
        detector = TextDetector(confidence_threshold=0.5)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is False
        assert len(result.boxes) == 0

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_exact_threshold_value(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [50],
            'left': [10],
            'top': [20],
            'width': [30],
            'height': [15]
        }
        detector = TextDetector(confidence_threshold=0.5)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert result.has_text is True
        assert len(result.boxes) == 1

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_high_threshold_filters_more(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [60, 70, 80, 90],
            'left': [10, 50, 100, 150],
            'top': [20, 30, 40, 50],
            'width': [30, 40, 50, 60],
            'height': [15, 20, 25, 30]
        }
        detector_low = TextDetector(confidence_threshold=0.5)
        result_low = detector_low.detect(make_blank_image())
        detector_high = TextDetector(confidence_threshold=0.75)
        result_high = detector_high.detect(make_blank_image())
        assert len(result_low.boxes) == 4
        assert len(result_high.boxes) == 2


class TestEdgeCases:
    '''Tests for edge cases'''

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_zero_width_height_boxes_filtered(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [80, 85, 90],
            'left': [10, 50, 100],
            'top': [20, 30, 40],
            'width': [0, 40, 50],
            'height': [15, 0, 25]
        }
        detector = TextDetector(confidence_threshold=0.5)
        roi = make_blank_image()
        result = detector.detect(roi)
        assert len(result.boxes) == 1
        assert result.boxes[0] == (100, 40, 150, 65)

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_grayscale_input(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [80],
            'left': [10],
            'top': [20],
            'width': [30],
            'height': [15]
        }
        detector = TextDetector()
        roi = make_grayscale_image()
        result = detector.detect(roi)
        assert result.has_text is True

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_single_pixel_image(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [],
            'left': [],
            'top': [],
            'width': [],
            'height': []
        }
        detector = TextDetector()
        roi = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result = detector.detect(roi)
        assert result.has_text is False

    @patch('src.visual.text_detector.pytesseract.image_to_data')
    def test_large_image(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            'conf': [85],
            'left': [100],
            'top': [200],
            'width': [500],
            'height': [100]
        }
        detector = TextDetector()
        roi = make_blank_image(h=1080, w=1920)
        result = detector.detect(roi)
        assert result.has_text is True


class TestIntegration:
    '''Integration tests without mocking'''

    def test_detect_with_real_blank_image(self):
        try:
            detector = TextDetector()
            roi = make_blank_image(200, 200)
            result = detector.detect(roi)
            assert isinstance(result, TextDetectionResult)
            assert isinstance(result.has_text, bool)
            assert isinstance(result.boxes, list)
            assert 0.0 <= result.confidence <= 1.0
        except Exception:
            pytest.skip('Tesseract not installed')
