"""
視覺特徵提取模組
對應計畫書 4.2 步驟三
"""

from .text_detector import TextDetector, TextDetectionResult
from .sbert_calculator import SBERTCalculator, compute_char_overlap
from .visual_scorer import (
    VisualScoreResult,
    SRGANEnhancer,
    compute_visual_score_v2,
    compute_visual_score,
    compute_iou,
    detect_highlighter,
    ocr_roi,
)

__all__ = [
    "TextDetector",
    "TextDetectionResult",
    "SBERTCalculator",
    "compute_char_overlap",
    "VisualScoreResult",
    "SRGANEnhancer",
    "compute_visual_score_v2",
    "compute_visual_score",
    "compute_iou",
    "detect_highlighter",
    "ocr_roi",
]
