"""
步驟三（後半）：視覺分數計算
- OCR 辨識板書內容並與 ASR 關鍵字計算語意相關性
- 降級備援：字跡模糊時改用手部座標與文字框 IoU
- 輔助特徵：HSV 色彩標記（螢光筆）
- v2：整合文字偵測優先、SBERT 語意計算、SRGAN 模糊增強
對應計畫書 4.2 步驟三
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pytesseract
from config import (
    IOU_FALLBACK_THRESHOLD,
    BLUR_THRESHOLD,
    SBERT_SIMILARITY_THRESHOLD,
    SBERT_OCR_WEIGHT,
)

from .text_detector import TextDetector, TextDetectionResult
from .sbert_calculator import SBERTCalculator
from .srgan import is_blurry, enhance_roi


class SRGANEnhancer:
    """SRGAN 增強器封裝類別"""
    
    def __init__(self, blur_threshold: float = BLUR_THRESHOLD):
        self.blur_threshold = blur_threshold
    
    def is_blurry(self, image: np.ndarray) -> bool:
        return is_blurry(image, self.blur_threshold)
    
    def enhance(self, roi: np.ndarray) -> np.ndarray:
        return enhance_roi(roi)


@dataclass
class VisualScoreResult:
    """視覺分數計算結果"""
    score: float
    text_detected: bool
    ocr_text: Optional[str]
    sbert_similarity: Optional[float]
    srgan_triggered: bool
    fallback_used: bool


def compute_iou(box1: tuple, box2: tuple) -> float:
    """計算兩個矩形框的 IoU"""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def detect_highlighter(frame: np.ndarray) -> float:
    """HSV 色彩空間偵測螢光筆標記"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 100, 100])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    ratio = mask.sum() / (255 * frame.shape[0] * frame.shape[1])
    return float(ratio)


def ocr_roi(roi: np.ndarray) -> str:
    """對 ROI 執行 OCR"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, lang="chi_tra+eng")
    return text.strip()


def compute_visual_score(
    roi: np.ndarray,
    hand_center: tuple[int, int],
    asr_keywords: list[str],
    frame: np.ndarray
) -> float:
    """計算最終視覺分數 S_visual（舊版）"""
    ocr_text = ocr_roi(roi)
    if ocr_text:
        matched = sum(1 for kw in asr_keywords if kw in ocr_text)
        ocr_score = matched / max(len(asr_keywords), 1)
    else:
        ocr_score = 0.0
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
        for i, conf in enumerate(data["conf"]):
            if int(conf) > 0:
                bx, by = data["left"][i], data["top"][i]
                bw, bh = data["width"][i], data["height"][i]
                text_box = (bx, by, bx + bw, by + bh)
                hx, hy = hand_center
                hand_box = (hx - 10, hy - 10, hx + 10, hy + 10)
                iou = compute_iou(hand_box, text_box)
                ocr_score = max(ocr_score, iou)
    highlight_score = detect_highlighter(frame)
    return min(1.0, ocr_score + highlight_score * 0.2)


def _compute_iou_fallback(
    roi: np.ndarray,
    hand_center: tuple[int, int],
    text_boxes: list[tuple[int, int, int, int]]
) -> float:
    """計算手部座標與文字框的 IoU 作為降級備援分數"""
    hx, hy = hand_center
    hand_box = (hx - 10, hy - 10, hx + 10, hy + 10)
    max_iou = 0.0
    if text_boxes:
        for text_box in text_boxes:
            iou = compute_iou(hand_box, text_box)
            max_iou = max(max_iou, iou)
    else:
        try:
            data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
            for i, conf in enumerate(data["conf"]):
                if int(conf) > 0:
                    bx, by = data["left"][i], data["top"][i]
                    bw, bh = data["width"][i], data["height"][i]
                    if bw > 0 and bh > 0:
                        text_box = (bx, by, bx + bw, by + bh)
                        iou = compute_iou(hand_box, text_box)
                        max_iou = max(max_iou, iou)
        except Exception:
            pass
    return max_iou


def compute_visual_score_v2(
    roi: np.ndarray,
    hand_center: tuple[int, int],
    asr_keywords: list[str],
    frame: np.ndarray,
    text_detector: Optional[TextDetector] = None,
    sbert_calculator: Optional[SBERTCalculator] = None,
    srgan_enhancer: Optional[SRGANEnhancer] = None,
    blur_threshold: float = BLUR_THRESHOLD,
    sbert_threshold: float = SBERT_SIMILARITY_THRESHOLD
) -> VisualScoreResult:
    """
    計算視覺分數（v2：偵測優先 + SBERT + SRGAN 整合）
    
    流程：
    1. 模糊偵測 → 觸發 SRGAN 增強（若需要）
    2. 文字偵測 → 判斷是否執行 OCR
    3. OCR 辨識 → SBERT 語意相似度計算
    4. 降級備援 → IoU 計算（若 OCR 失敗或未偵測到文字）
    """
    if text_detector is None:
        text_detector = TextDetector()
    if sbert_calculator is None:
        sbert_calculator = SBERTCalculator()
    if srgan_enhancer is None:
        srgan_enhancer = SRGANEnhancer(blur_threshold=blur_threshold)
    
    srgan_triggered = False
    text_detected = False
    ocr_text: Optional[str] = None
    sbert_similarity: Optional[float] = None
    fallback_used = False
    score = 0.0
    
    if roi is None or roi.size == 0:
        return VisualScoreResult(
            score=0.0, text_detected=False, ocr_text=None,
            sbert_similarity=None, srgan_triggered=False, fallback_used=True
        )
    
    # 步驟 1：模糊偵測 → SRGAN 增強
    processed_roi = roi.copy()
    if srgan_enhancer.is_blurry(roi):
        processed_roi = srgan_enhancer.enhance(roi)
        srgan_triggered = True
    
    # 步驟 2：文字偵測
    detection_result: TextDetectionResult = text_detector.detect(processed_roi)
    text_detected = detection_result.has_text
    
    # 步驟 3：OCR 辨識 + SBERT 語意計算
    if text_detected:
        ocr_text = ocr_roi(processed_roi)
        if ocr_text:
            sbert_similarity = sbert_calculator.compute_similarity_with_keywords(
                ocr_text, asr_keywords
            )
            if sbert_similarity >= sbert_threshold:
                score = sbert_similarity * SBERT_OCR_WEIGHT
            else:
                matched = sum(1 for kw in asr_keywords if kw in ocr_text)
                char_overlap_score = matched / max(len(asr_keywords), 1)
                score = (sbert_similarity * 0.5 + char_overlap_score * 0.5) * SBERT_OCR_WEIGHT
        else:
            fallback_used = True
            score = _compute_iou_fallback(processed_roi, hand_center, detection_result.boxes)
    else:
        # 步驟 4：未偵測到文字，使用 IoU 降級備援
        fallback_used = True
        score = _compute_iou_fallback(processed_roi, hand_center, [])
    
    # 加入螢光筆輔助特徵
    highlight_score = detect_highlighter(frame)
    score = min(1.0, score + highlight_score * 0.2)
    
    return VisualScoreResult(
        score=score,
        text_detected=text_detected,
        ocr_text=ocr_text,
        sbert_similarity=sbert_similarity,
        srgan_triggered=srgan_triggered,
        fallback_used=fallback_used
    )
