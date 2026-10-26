"""
視覺特徵提取模組

包含手部追蹤、手勢分類、板書 OCR、CLIP 對齊、視覺可靠度估測等子模組。
使用 lazy import 避免 import 時觸發模型下載。
"""

# 使用 lazy import 避免 import 時觸發模型下載
def __getattr__(name):
    if name in ("TextDetector", "TextDetectionResult"):
        from .text_detector import TextDetector, TextDetectionResult
        return locals()[name]
    if name in ("SBERTCalculator", "compute_char_overlap"):
        from .sbert_calculator import SBERTCalculator, compute_char_overlap
        return locals()[name]
    if name in ("VisualScoreResult", "SRGANEnhancer", "compute_visual_score_v2",
                "compute_visual_score", "compute_iou", "detect_highlighter", "ocr_roi"):
        from .visual_scorer import (
            VisualScoreResult, SRGANEnhancer, compute_visual_score_v2,
            compute_visual_score, compute_iou, detect_highlighter, ocr_roi,
        )
        return locals()[name]
    if name in ("VisualReliabilityEstimator", "ReliabilitySignals", "ReliabilityResult",
                "build_signals_from_tracker_result"):
        from .visual_reliability import (
            VisualReliabilityEstimator, ReliabilitySignals, ReliabilityResult,
            build_signals_from_tracker_result,
        )
        return locals()[name]
    if name == "CLIPScorer":
        from .clip_scorer import CLIPScorer
        return CLIPScorer
    if name in ("HandTracker",):
        from .hand_tracker import HandTracker
        return HandTracker
    if name in ("GestureClassifier", "GestureIntent", "GestureResult"):
        from .gesture_classifier import GestureClassifier, GestureIntent, GestureResult
        return locals()[name]
    raise AttributeError(f"module 'src.visual' has no attribute {name!r}")
