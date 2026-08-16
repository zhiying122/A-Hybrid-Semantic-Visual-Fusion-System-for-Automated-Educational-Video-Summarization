"""
步驟三：視覺特徵提取 - 手部軌跡追蹤 (v3)
─────────────────────────────────────────────────────────────────────
v3 改進：整合 GestureClassifier，把「手是否靜止」升級為「手在做什麼」

原始做法：只看食指位移變異數（靜止 = 重要，移動 = 不重要）
v3 做法：用 GRU 模型對食指軌跡序列分類為 5 種手勢意圖，
          每種意圖對應不同的視覺重要性貢獻分數：
            POINTING   → 1.00（最重要）
            EMPHASIS   → 0.90（強調）
            WRITING    → 0.85（書寫）
            TRANSITION → 0.20（換場）
            IDLE       → 0.10（無手）

模型不存在時自動降級為規則型分類器（不影響主流程）。

- 使用 MediaPipe Hands Tasks API（0.10.x 新版）進行手部骨架追蹤
- 卡爾曼濾波平滑座標
- 觸發後裁切 ROI
對應計畫書 4.2 步驟三
"""

import cv2
import numpy as np
from config import HAND_VARIANCE_THRESHOLD, ROI_SIZE
from src.visual.gesture_classifier import GestureClassifier, GestureIntent, GestureResult

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
        # v3：初始化手勢意圖分類器
        self.gesture_classifier = GestureClassifier()
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

        self._api_mode = "none"
        print("[HandTracker] 警告：MediaPipe 無法初始化，視覺分數將為 0")

    def _init_tasks_api(self):
        """初始化新版 Tasks API"""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request
        import os

        # 模型搜尋順序：專案 models/ 目錄 → 當前目錄 → 自動下載
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _model_candidates = [
            os.path.join(_project_root, "models", "hand_landmarker.task"),
            os.path.join(_project_root, "weights", "hand_landmarker.task"),
            "hand_landmarker.task",
        ]

        model_path = None
        for candidate in _model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if model_path is None:
            # 預設下載至 models/ 目錄
            model_path = _model_candidates[0]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print(f"[HandTracker] 下載 MediaPipe 手部模型至 {model_path}...")
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
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        return result.hand_landmarks if result.hand_landmarks else None

    def _detect_landmarks_legacy(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands_legacy.process(rgb)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else None

    def _is_pointing_gesture(self, landmarks, h: int, w: int) -> bool:
        """判斷是否為伸展/指引姿態（保留相容舊邏輯）"""
        if self._api_mode == "tasks":
            lm = landmarks[0]
            index_extended = lm[8].y < lm[6].y
            middle_bent = lm[12].y > lm[10].y
        else:
            lm = landmarks[0].landmark
            index_extended = lm[8].y < lm[6].y
            middle_bent = lm[12].y > lm[10].y
        return index_extended and middle_bent

    def _get_index_tip(self, landmarks, h: int, w: int) -> tuple[float, float]:
        if self._api_mode == "tasks":
            lm = landmarks[0]
            return lm[8].x * w, lm[8].y * h
        else:
            lm = landmarks[0].landmark
            return lm[8].x * w, lm[8].y * h

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        處理單一幀，回傳視覺事件資訊（v3 新增 gesture_result）

        回傳：{
            "triggered": bool,
            "roi_center": tuple or None,
            "s_visual_raw": float,        # 基於變異數的原始分數（向後兼容）
            "gesture_result": GestureResult,  # v3 新增：手勢意圖分類結果
            "intent_score": float,            # v3 新增：意圖重要性分數
        }
        """
        if self._api_mode == "none":
            from src.visual.gesture_classifier import _make_result
            idle = _make_result(GestureIntent.IDLE, 0.95)
            return {
                "triggered": False,
                "roi_center": None,
                "s_visual_raw": 0.0,
                "gesture_result": idle,
                "intent_score": idle.intent_score,
            }

        h, w = frame.shape[:2]

        try:
            if self._api_mode == "tasks":
                landmarks = self._detect_landmarks_tasks(frame)
            else:
                landmarks = self._detect_landmarks_legacy(frame)
        except Exception:
            from src.visual.gesture_classifier import _make_result
            idle = _make_result(GestureIntent.IDLE, 0.9)
            return {
                "triggered": False,
                "roi_center": None,
                "s_visual_raw": 0.0,
                "gesture_result": idle,
                "intent_score": idle.intent_score,
            }

        if not landmarks:
            self.coord_history.clear()
            from src.visual.gesture_classifier import _make_result
            idle = _make_result(GestureIntent.IDLE, 0.9)
            return {
                "triggered": False,
                "roi_center": None,
                "s_visual_raw": 0.0,
                "gesture_result": idle,
                "intent_score": idle.intent_score,
            }

        raw_x, raw_y = self._get_index_tip(landmarks, h, w)
        sx, sy = self.smoother.update(raw_x, raw_y)

        self.coord_history.append((sx, sy))
        if len(self.coord_history) > self.window_size:
            self.coord_history.pop(0)

        if len(self.coord_history) < 3:
            from src.visual.gesture_classifier import _make_result
            idle = _make_result(GestureIntent.IDLE, 0.8)
            return {
                "triggered": False,
                "roi_center": None,
                "s_visual_raw": 0.0,
                "gesture_result": idle,
                "intent_score": idle.intent_score,
            }

        xs = [c[0] for c in self.coord_history]
        ys = [c[1] for c in self.coord_history]
        variance = np.var(xs) + np.var(ys)

        # v3：用手勢分類器取代純變異數判斷
        gesture_result = self.gesture_classifier.classify(
            self.coord_history,
            frame_w=w,
            frame_h=h,
            hand_detected=True,
        )

        # triggered 條件：指引或書寫或強調（意圖分數 > 0.7）
        triggered = gesture_result.intent_score >= 0.7
        # s_visual_raw 改為用意圖分數（而非純變異數）
        s_visual_raw = gesture_result.intent_score

        return {
            "triggered": triggered,
            "roi_center": (int(sx), int(sy)) if triggered else None,
            "s_visual_raw": s_visual_raw,
            "gesture_result": gesture_result,
            "intent_score": gesture_result.intent_score,
        }

    def extract_roi(self, frame: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        """以食指座標為中心裁切 ROI_SIZE × ROI_SIZE 區域"""
        x, y = center
        h, w = frame.shape[:2]
        half = ROI_SIZE // 2
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half), min(h, y + half)
        return frame[y1:y2, x1:x2]
