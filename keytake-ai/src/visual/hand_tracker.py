"""
步驟三：視覺特徵提取 - 手部軌跡追蹤
- 使用 MediaPipe Hands Tasks API（0.10.x 新版）進行手部骨架追蹤
- 卡爾曼濾波平滑座標
- 計算位移變異數判斷「滯留/指引」意圖
- 觸發後裁切 ROI
對應計畫書 4.2 步驟三
"""

import cv2
import numpy as np
from config import HAND_VARIANCE_THRESHOLD, ROI_SIZE

# 嘗試載入新版 MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _USE_TASKS_API = True
except (ImportError, AttributeError):
    _USE_TASKS_API = False

# 降級備援：嘗試舊版 solutions API
try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
    _USE_LEGACY_API = True
except AttributeError:
    _USE_LEGACY_API = False


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
        return float(predicted[0][0]), float(predicted[1][0])


class HandTracker:
    def __init__(self, window_size: int = 15):
        self.smoother = KalmanSmoother()
        self.window_size = window_size
        self.coord_history: list[tuple[float, float]] = []
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        """初始化手部偵測器，自動選擇可用的 API"""
        if _USE_TASKS_API:
            try:
                self._init_tasks_api()
                self._api_mode = "tasks"
                return
            except Exception:
                pass

        if _USE_LEGACY_API:
            try:
                self._hands_legacy = _mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                )
                self._api_mode = "legacy"
                return
            except Exception:
                pass

        # 完全降級：不使用 MediaPipe
        self._api_mode = "none"
        print("[HandTracker] 警告：MediaPipe 無法初始化，視覺分數將為 0")

    def _init_tasks_api(self):
        """初始化新版 Tasks API"""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request
        import os

        # 下載模型（若不存在）
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print("[HandTracker] 下載 MediaPipe 手部模型...")
            urllib.request.urlretrieve(url, model_path)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def _detect_landmarks_tasks(self, frame: np.ndarray):
        """使用新版 Tasks API 偵測手部關鍵點"""
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        return result.hand_landmarks if result.hand_landmarks else None

    def _detect_landmarks_legacy(self, frame: np.ndarray):
        """使用舊版 solutions API 偵測手部關鍵點"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands_legacy.process(rgb)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else None

    def _is_pointing_gesture(self, landmarks, h: int, w: int) -> bool:
        """判斷是否為伸展/指引姿態"""
        if self._api_mode == "tasks":
            lm = landmarks[0]  # Tasks API 回傳 list of NormalizedLandmark
            index_extended = lm[8].y < lm[6].y
            middle_bent = lm[12].y > lm[10].y
        else:
            lm = landmarks[0].landmark
            index_extended = lm[8].y < lm[6].y
            middle_bent = lm[12].y > lm[10].y
        return index_extended and middle_bent

    def _get_index_tip(self, landmarks, h: int, w: int) -> tuple[float, float]:
        """取得食指指尖座標"""
        if self._api_mode == "tasks":
            lm = landmarks[0]
            return lm[8].x * w, lm[8].y * h
        else:
            lm = landmarks[0].landmark
            return lm[8].x * w, lm[8].y * h

    def process_frame(self, frame: np.ndarray) -> dict:
        """處理單一幀，回傳視覺事件資訊"""
        if self._api_mode == "none":
            return {"triggered": False, "roi_center": None, "s_visual_raw": 0.0}

        h, w = frame.shape[:2]

        try:
            if self._api_mode == "tasks":
                landmarks = self._detect_landmarks_tasks(frame)
            else:
                landmarks = self._detect_landmarks_legacy(frame)
        except Exception:
            return {"triggered": False, "roi_center": None, "s_visual_raw": 0.0}

        if not landmarks:
            self.coord_history.clear()
            return {"triggered": False, "roi_center": None, "s_visual_raw": 0.0}

        raw_x, raw_y = self._get_index_tip(landmarks, h, w)
        sx, sy = self.smoother.update(raw_x, raw_y)

        self.coord_history.append((sx, sy))
        if len(self.coord_history) > self.window_size:
            self.coord_history.pop(0)

        if len(self.coord_history) < 3:
            return {"triggered": False, "roi_center": None, "s_visual_raw": 0.0}

        xs = [c[0] for c in self.coord_history]
        ys = [c[1] for c in self.coord_history]
        variance = np.var(xs) + np.var(ys)

        is_pointing = self._is_pointing_gesture(landmarks, h, w)
        triggered = (variance < HAND_VARIANCE_THRESHOLD) and is_pointing
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
