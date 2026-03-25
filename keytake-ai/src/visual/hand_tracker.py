"""
步驟三：視覺特徵提取 - 手部軌跡追蹤
- 使用 MediaPipe Hands 進行手部骨架追蹤
- 卡爾曼濾波平滑座標
- 計算位移變異數判斷「滯留/指引」意圖
- 觸發後裁切 ROI
對應計畫書 4.2 步驟三
"""

import cv2
import numpy as np
import mediapipe as mp
from config import HAND_VARIANCE_THRESHOLD, ROI_SIZE


class KalmanSmoother:
    """簡易 1D 卡爾曼濾波，用於平滑手部座標抖動"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                              [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def update(self, x: float, y: float) -> tuple[float, float]:
        measurement = np.array([[x], [y]], np.float32)
        self.kf.correct(measurement)
        predicted = self.kf.predict()
        return float(predicted[0]), float(predicted[1])


class HandTracker:
    def __init__(self, window_size: int = 15):
        """
        window_size: 計算位移變異數的滑動視窗幀數
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.smoother = KalmanSmoother()
        self.window_size = window_size
        self.coord_history: list[tuple[float, float]] = []

    def _is_pointing_gesture(self, hand_landmarks) -> bool:
        """
        判斷是否為伸展/指引姿態：食指伸直且其他手指彎曲
        使用指尖與指根的 y 座標差判斷
        """
        lm = hand_landmarks.landmark
        # 食指伸直：指尖(8) 高於 第二關節(6)
        index_extended = lm[8].y < lm[6].y
        # 中指彎曲：指尖(12) 低於 第二關節(10)
        middle_bent = lm[12].y > lm[10].y
        return index_extended and middle_bent

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        處理單一幀，回傳視覺事件資訊
        回傳格式：{"triggered": bool, "roi_center": (x, y) or None, "s_visual_raw": float}
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            self.coord_history.clear()
            return {"triggered": False, "roi_center": None, "s_visual_raw": 0.0}

        # 取第一隻手的食指指尖座標（landmark 8）
        hand = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        raw_x = hand.landmark[8].x * w
        raw_y = hand.landmark[8].y * h
        sx, sy = self.smoother.update(raw_x, raw_y)

        self.coord_history.append((sx, sy))
        if len(self.coord_history) > self.window_size:
            self.coord_history.pop(0)

        # 計算位移變異數
        if len(self.coord_history) < 3:
            return {"triggered": False, "roi_center": None, "s_visual_raw": 0.0}

        xs = [c[0] for c in self.coord_history]
        ys = [c[1] for c in self.coord_history]
        variance = np.var(xs) + np.var(ys)

        # 判斷滯留 + 指引姿態 → 觸發 ROI 擷取
        is_pointing = self._is_pointing_gesture(hand)
        triggered = (variance < HAND_VARIANCE_THRESHOLD) and is_pointing

        # 視覺分數：變異數越低（越穩定）分數越高，正規化到 [0,1]
        s_visual_raw = max(0.0, 1.0 - variance / (HAND_VARIANCE_THRESHOLD * 2))

        return {
            "triggered": triggered,
            "roi_center": (int(sx), int(sy)) if triggered else None,
            "s_visual_raw": s_visual_raw
        }

    def extract_roi(self, frame: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        """以食指座標為中心裁切 ROI_SIZE × ROI_SIZE 區域"""
        x, y = center
        h, w = frame.shape[:2]
        half = ROI_SIZE // 2
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half), min(h, y + half)
        return frame[y1:y2, x1:x2]
