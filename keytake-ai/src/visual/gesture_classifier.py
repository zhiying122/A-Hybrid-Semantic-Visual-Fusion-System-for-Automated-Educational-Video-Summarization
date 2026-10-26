"""手勢意圖分類器

分析食指軌跡序列，將手部動作分為五類：
  POINTING   — 靜止指向（重要性 1.00）
  WRITING    — 手腕小幅規律移動（重要性 0.85）
  EMPHASIS   — 局部反覆來回強調（重要性 0.90）
  TRANSITION — 大幅移動換區（重要性 0.20）
  IDLE       — 未偵測到手（重要性 0.10）

模型輸入：過去 N 幀食指位置序列，shape = (seq_len, 4)，
特徵為 [x_norm, y_norm, dx_norm, dy_norm]。
採用雙層 GRU，約 50 KB，模型不存在時自動降級為規則型分類器。
"""

from __future__ import annotations

import os
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


class GestureIntent(IntEnum):
    POINTING   = 0
    WRITING    = 1
    EMPHASIS   = 2
    TRANSITION = 3
    IDLE       = 4


# 各意圖對應的視覺重要性貢獻分數，乘上 s_visual 得到加權視覺分數
INTENT_SCORE_MAP = {
    GestureIntent.POINTING:   1.0,
    GestureIntent.WRITING:    0.85,
    GestureIntent.EMPHASIS:   0.90,
    GestureIntent.TRANSITION: 0.20,
    GestureIntent.IDLE:       0.10,
}

INTENT_LABELS = {
    GestureIntent.POINTING:   "指引",
    GestureIntent.WRITING:    "書寫",
    GestureIntent.EMPHASIS:   "強調",
    GestureIntent.TRANSITION: "過渡",
    GestureIntent.IDLE:       "無手部",
}


@dataclass
class GestureResult:
    intent: GestureIntent
    confidence: float           # 最高類別的機率
    intent_score: float         # 對應的重要性分數
    label: str
    probabilities: list[float]  # 全部 5 類的機率分布


# ── GRU 模型架構（僅在 TORCH_AVAILABLE 時定義）─────────────────────

if TORCH_AVAILABLE:
    class GestureGRU(nn.Module):
        """雙層 GRU 手勢分類器。

        輸入：(batch, seq_len, 4)  輸出：(batch, 5)
        """

        def __init__(
            self,
            input_size: int = 4,
            hidden_size: int = 64,
            num_layers: int = 2,
            num_classes: int = 5,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.gru(x)
            return self.classifier(out[:, -1, :])
else:
    GestureGRU = None  # type: ignore[assignment]


# ── 特徵萃取 ──────────────────────────────────────────────────────

def extract_trajectory_features(
    coords: list[tuple[float, float]],
    frame_w: int = 640,
    frame_h: int = 480,
) -> np.ndarray:
    """把食指座標歷史轉成 (N, 4) 特徵矩陣 [x_norm, y_norm, dx_norm, dy_norm]。

    dx/dy 是相鄰幀位移；速度以全域最大值正規化到 [-1, 1]。
    靜止片段中 max_speed 趨近 0 時，clip 後速度全被放大—此屬預期行為，
    因靜止段本身就應被 POINTING 分類器捕捉。
    """
    if not coords:
        return np.zeros((1, 4), dtype=np.float32)

    coords_arr = np.array(coords, dtype=np.float32)
    x_norm = coords_arr[:, 0] / max(frame_w, 1)
    y_norm = coords_arr[:, 1] / max(frame_h, 1)

    if len(coords) > 1:
        dx = np.diff(x_norm, prepend=x_norm[0])
        dy = np.diff(y_norm, prepend=y_norm[0])
    else:
        dx = np.zeros_like(x_norm)
        dy = np.zeros_like(y_norm)

    max_speed = max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
    dx = np.clip(dx / max_speed, -1.0, 1.0)
    dy = np.clip(dy / max_speed, -1.0, 1.0)

    return np.stack([x_norm, y_norm, dx, dy], axis=1).astype(np.float32)


def pad_or_truncate(features: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """截斷或前置零填充特徵序列至 seq_len。"""
    n = features.shape[0]
    if n >= seq_len:
        return features[-seq_len:]
    pad = np.zeros((seq_len - n, features.shape[1]), dtype=np.float32)
    return np.vstack([pad, features])


# ── 規則型後備分類器 ──────────────────────────────────────────────

def rule_based_classify(
    coords: list[tuple[float, float]],
    variance_threshold: float = 50.0,
) -> GestureResult:
    """無訓練模型時的規則型分類（基於位移變異數與方向反轉次數）。

    注意：variance_threshold 以像素平方為單位，對不同解析度的影片
    閾值敏感度會有差異，建議搭配正規化解析度的輸入使用。
    """
    if len(coords) < 3:
        return _make_result(GestureIntent.IDLE, 0.9)

    coords_arr = np.array(coords, dtype=np.float32)
    xs, ys = coords_arr[:, 0], coords_arr[:, 1]
    var = float(np.var(xs) + np.var(ys))

    dxs = np.diff(xs)
    reversal_count = int(np.sum(dxs[:-1] * dxs[1:] < 0)) if len(coords) > 2 else 0

    dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    path_length = float(dists.sum()) if len(coords) > 1 else 0.0

    displacement = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))

    if var < variance_threshold * 0.5:
        return _make_result(
            GestureIntent.POINTING,
            min(1.0, 1.0 - var / (variance_threshold * 0.5))
        )
    elif reversal_count >= 3 and path_length > 20:
        return _make_result(GestureIntent.EMPHASIS, 0.75)
    elif var < variance_threshold * 2 and path_length < 80:
        return _make_result(GestureIntent.WRITING, 0.70)
    elif displacement > 100:
        return _make_result(GestureIntent.TRANSITION, 0.80)
    else:
        return _make_result(GestureIntent.POINTING, 0.50)


def _make_result(intent: GestureIntent, confidence: float) -> GestureResult:
    rest = (1.0 - confidence) / 4
    probs = [rest] * 5
    probs[int(intent)] = confidence
    return GestureResult(
        intent=intent,
        confidence=confidence,
        intent_score=INTENT_SCORE_MAP[intent],
        label=INTENT_LABELS[intent],
        probabilities=probs,
    )


# ── 主分類器介面 ──────────────────────────────────────────────────

class GestureClassifier:
    """手勢意圖分類器。

    優先使用 GRU 模型（weights/gesture_classifier.pth）；
    模型不存在或 PyTorch 未安裝時自動降級為規則型分類器。

    使用方式：
        classifier = GestureClassifier()
        result = classifier.classify(coord_history, frame_w=1280, frame_h=720)
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../weights/gesture_classifier.pth")
    SEQ_LEN = 20
    INPUT_SIZE = 4

    def __init__(self, device: str = "auto"):
        self._mode = "rule"
        self.model = None
        self.device = None

        if not TORCH_AVAILABLE:
            print("[GestureClassifier] PyTorch 未安裝，使用規則型分類器")
            return

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self) -> None:
        model_path = os.path.abspath(self.MODEL_PATH)
        if not os.path.exists(model_path):
            print(f"[GestureClassifier] 模型未找到（{model_path}），使用規則型分類器")
            print("[GestureClassifier] 執行 python tools/train_gesture_classifier.py 可訓練模型")
            return

        try:
            self.model = GestureGRU(
                input_size=self.INPUT_SIZE,
                hidden_size=64,
                num_layers=2,
                num_classes=5,
            ).to(self.device)
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self._mode = "gru"
            print(f"[GestureClassifier] 載入 GRU 模型：{model_path}")
        except Exception as e:
            print(f"[GestureClassifier] 模型載入失敗（{e}），使用規則型分類器")

    def classify(
        self,
        coord_history: list[tuple[float, float]],
        frame_w: int = 640,
        frame_h: int = 480,
        hand_detected: bool = True,
    ) -> GestureResult:
        if not hand_detected or len(coord_history) < 2:
            return _make_result(GestureIntent.IDLE, 0.95)

        if self._mode == "gru":
            return self._classify_gru(coord_history, frame_w, frame_h)
        return rule_based_classify(coord_history)

    def _classify_gru(
        self,
        coord_history: list[tuple[float, float]],
        frame_w: int,
        frame_h: int,
    ) -> GestureResult:
        features = extract_trajectory_features(coord_history, frame_w, frame_h)
        padded = pad_or_truncate(features, self.SEQ_LEN)

        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        intent_idx = int(np.argmax(probs))
        intent = GestureIntent(intent_idx)
        return GestureResult(
            intent=intent,
            confidence=float(probs[intent_idx]),
            intent_score=INTENT_SCORE_MAP[intent],
            label=INTENT_LABELS[intent],
            probabilities=probs.tolist(),
        )

    @property
    def mode(self) -> str:
        return self._mode
