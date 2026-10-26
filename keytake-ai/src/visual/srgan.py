"""影像超解析度增強模組

偵測 ROI 是否模糊（Laplacian 變異數），模糊時以 bicubic 插值放大。
若未來需要 Real-ESRGAN，可在此替換 enhance_roi 實作。
"""

import cv2
import numpy as np
from config import SRGAN_SCALE


def enhance_roi(roi: np.ndarray, scale: int = SRGAN_SCALE) -> np.ndarray:
    """以 bicubic 插值對 ROI 進行超解析度增強。"""
    if roi is None or roi.size == 0:
        return roi
    h, w = roi.shape[:2]
    if roi.dtype != np.uint8:
        roi = np.clip(roi, 0, 255).astype(np.uint8)
    return cv2.resize(roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """以 Laplacian 變異數判斷影像是否模糊（低於閾值即視為模糊）。"""
    if image is None or image.size == 0:
        return False
    # 支援灰階（ndim=2）與彩色（ndim=3）輸入
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(variance < threshold)


def enhance_if_blurry(roi: np.ndarray, threshold: float = 100.0) -> tuple[np.ndarray, bool]:
    """只在偵測到模糊時才觸發增強，回傳 (增強後影像, 是否觸發)。"""
    if is_blurry(roi, threshold):
        return enhance_roi(roi), True
    return roi, False
