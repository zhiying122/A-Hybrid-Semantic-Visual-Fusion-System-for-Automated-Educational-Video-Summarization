"""文字區域偵測模組

以 pytesseract image_to_data 偵測 ROI 中的文字框，
作為 OCR 的前置過濾，避免對無文字區域執行完整 OCR。
"""

from dataclasses import dataclass
import cv2
import numpy as np
import pytesseract

from config import TEXT_DETECTION_CONFIDENCE_THRESHOLD


@dataclass
class TextDetectionResult:
    has_text: bool
    boxes: list[tuple[int, int, int, int]]  # [(x1, y1, x2, y2), ...]
    confidence: float                        # 偵測到的文字框平均信心度 [0, 1]


class TextDetector:
    def __init__(self, confidence_threshold: float = TEXT_DETECTION_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def detect(self, roi: np.ndarray) -> TextDetectionResult:
        """偵測 ROI 中的文字框，回傳 TextDetectionResult。"""
        if roi is None or roi.size == 0:
            return TextDetectionResult(has_text=False, boxes=[], confidence=0.0)

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            data = pytesseract.image_to_data(
                thresh,
                output_type=pytesseract.Output.DICT,
                lang="chi_tra+eng",
            )
        except Exception:
            return TextDetectionResult(has_text=False, boxes=[], confidence=0.0)

        boxes = []
        confidences = []

        for i, conf in enumerate(data["conf"]):
            # pytesseract 以 -1（或 -1.0）標記整列資訊而非詞語
            if int(conf) < 0:
                continue
            conf_normalized = float(conf) / 100.0
            if conf_normalized < self.confidence_threshold:
                continue
            w, h = data["width"][i], data["height"][i]
            if w <= 0 or h <= 0:
                continue
            x, y = data["left"][i], data["top"][i]
            boxes.append((x, y, x + w, y + h))
            confidences.append(conf_normalized)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return TextDetectionResult(
            has_text=len(boxes) > 0,
            boxes=boxes,
            confidence=avg_confidence,
        )
