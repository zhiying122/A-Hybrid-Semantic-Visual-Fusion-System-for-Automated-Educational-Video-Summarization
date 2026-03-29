"""
文字區域偵測模組
- 使用 pytesseract image_to_data 進行文字區域偵測
- 作為 OCR 的前置判斷，減少不必要的 OCR 運算
對應計畫書 4.2 步驟三：視覺特徵提取
"""

from dataclasses import dataclass
import cv2
import numpy as np
import pytesseract

from config import TEXT_DETECTION_CONFIDENCE_THRESHOLD


@dataclass
class TextDetectionResult:
    """文字偵測結果"""
    has_text: bool                              # 是否偵測到文字
    boxes: list[tuple[int, int, int, int]]      # 文字框 [(x1, y1, x2, y2), ...]
    confidence: float                           # 偵測信心度 [0.0, 1.0]


class TextDetector:
    """文字區域偵測器"""
    
    def __init__(self, confidence_threshold: float = TEXT_DETECTION_CONFIDENCE_THRESHOLD):
        """
        初始化文字偵測器
        
        Args:
            confidence_threshold: 文字偵測信心度閾值，預設使用 config 設定
        """
        self.confidence_threshold = confidence_threshold
    
    def detect(self, roi: np.ndarray) -> TextDetectionResult:
        """
        偵測 ROI 中的文字區域
        
        Args:
            roi: BGR 格式的 ROI 影像 (H, W, 3)
            
        Returns:
            TextDetectionResult 包含：
            - has_text: bool - 是否偵測到文字
            - boxes: list[tuple] - 文字框座標 [(x1, y1, x2, y2), ...]
            - confidence: float - 偵測信心度 [0.0, 1.0]
        """
        # 驗證輸入
        if roi is None or roi.size == 0:
            return TextDetectionResult(has_text=False, boxes=[], confidence=0.0)
        
        # 轉換為灰階並進行預處理
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # 使用 OTSU 二值化增強文字對比
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 使用 pytesseract image_to_data 進行文字區域偵測
        try:
            data = pytesseract.image_to_data(
                thresh, 
                output_type=pytesseract.Output.DICT,
                lang="chi_tra+eng"
            )
        except Exception:
            # 偵測失敗時回傳無文字結果
            return TextDetectionResult(has_text=False, boxes=[], confidence=0.0)
        
        # 解析偵測結果
        boxes = []
        confidences = []
        
        for i, conf in enumerate(data["conf"]):
            # pytesseract 回傳的 conf 為 -1 表示無效
            if conf == -1:
                continue
            
            conf_normalized = float(conf) / 100.0  # 轉換為 [0.0, 1.0]
            
            # 只保留信心度高於閾值的文字框
            if conf_normalized >= self.confidence_threshold:
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                
                # 過濾無效的文字框（寬高為 0）
                if w > 0 and h > 0:
                    boxes.append((x, y, x + w, y + h))
                    confidences.append(conf_normalized)
        
        # 計算整體信心度（取平均）
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.0
        
        has_text = len(boxes) > 0
        
        return TextDetectionResult(
            has_text=has_text,
            boxes=boxes,
            confidence=avg_confidence
        )
