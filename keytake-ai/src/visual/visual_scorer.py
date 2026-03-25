"""
步驟三（後半）：視覺分數計算
- OCR 辨識板書內容並與 ASR 關鍵字計算語意相關性
- 降級備援：字跡模糊時改用手部座標與文字框 IoU
- 輔助特徵：HSV 色彩標記（螢光筆）
對應計畫書 4.2 步驟三
"""

import cv2
import numpy as np
import pytesseract
from config import IOU_FALLBACK_THRESHOLD


def compute_iou(box1: tuple, box2: tuple) -> float:
    """計算兩個矩形框的 IoU（降級備援用）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def detect_highlighter(frame: np.ndarray) -> float:
    """
    HSV 色彩空間偵測螢光筆標記（輔助特徵）
    回傳標記像素佔比作為輔助視覺分數
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 螢光黃範圍
    lower = np.array([25, 100, 100])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    ratio = mask.sum() / (255 * frame.shape[0] * frame.shape[1])
    return float(ratio)


def ocr_roi(roi: np.ndarray) -> str:
    """對 ROI 執行 OCR，回傳辨識文字"""
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
    """
    計算最終視覺分數 S_visual
    策略：OCR 優先，字跡模糊時降級為 IoU
    """
    ocr_text = ocr_roi(roi)

    if ocr_text:
        # OCR 成功：計算與 ASR 關鍵字的字元重疊率
        matched = sum(1 for kw in asr_keywords if kw in ocr_text)
        ocr_score = matched / max(len(asr_keywords), 1)
    else:
        # 降級備援：使用手部座標與文字框 IoU
        ocr_score = 0.0
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
        for i, conf in enumerate(data["conf"]):
            if int(conf) > 0:
                bx = data["left"][i]
                by = data["top"][i]
                bw = data["width"][i]
                bh = data["height"][i]
                text_box = (bx, by, bx + bw, by + bh)
                hx, hy = hand_center
                hand_box = (hx - 10, hy - 10, hx + 10, hy + 10)
                iou = compute_iou(hand_box, text_box)
                ocr_score = max(ocr_score, iou)

    # 加入螢光筆輔助特徵
    highlight_score = detect_highlighter(frame)
    return min(1.0, ocr_score + highlight_score * 0.2)
