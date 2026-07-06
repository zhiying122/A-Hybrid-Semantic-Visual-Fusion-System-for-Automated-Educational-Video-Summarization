"""
手勢意圖分類器 (Gesture Intent Classifier)
──────────────────────────────────────────────────────────────────
這是 KeyTake AI v3 的核心創新模組。

過去的做法只看「手部是否靜止」（位移變異數閾值），這太粗糙：
  - 教師走動時手也可能是靜止的（但不是在指板書）
  - 書寫時手是移動的（但這是重要時刻）

本模組改為分析食指軌跡的時序序列，把手部動作分成 5 類：

  POINTING  (0) — 食指靜止指向某個位置，最強的「我要你看這裡」信號
  WRITING   (1) — 手腕小幅規律移動，正在板書，通常是重要內容
  EMPHASIS  (2) — 在某區域反覆來回（下劃線、畫圈），強調動作
  TRANSITION(3) — 大幅度橫向或縱向移動，換話題或換區域
  IDLE      (4) — 沒有偵測到手，或手部不在畫面中

每種意圖對應不同的視覺重要性權重，直接影響 S_visual 的計算。

架構選擇：
  輸入：過去 N 幀的食指位置序列，shape = (seq_len, 4)
        特徵：[x_norm, y_norm, dx_norm, dy_norm]（歸一化座標 + 速度）
  模型：雙層 GRU（替代 LSTM，參數更少、速度更快，在短序列效果相當）
  輸出：5 類手勢意圖的機率分布

為什麼用 GRU 而不是 Transformer：
  序列長度只有 15~30 幀（0.5~1 秒），Transformer 的 self-attention
  在這個長度上沒有優勢，GRU 更快更輕量，適合嵌入即時 pipeline。

模型大小：~50KB（不含訓練資料），可以跟著 repo 一起提交。
"""

from __future__ import annotations

import os
import json
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# 手勢意圖類別定義
# ──────────────────────────────────────────────────────────────────

class GestureIntent(IntEnum):
    POINTING   = 0   # 靜止指引
    WRITING    = 1   # 書寫板書
    EMPHASIS   = 2   # 強調（來回移動）
    TRANSITION = 3   # 過渡換區
    IDLE       = 4   # 無手部

# 每種意圖對應的「視覺重要性貢獻分數」
# 這個分數會乘上 S_visual，讓不同手勢對最終分數有不同貢獻
INTENT_SCORE_MAP = {
    GestureIntent.POINTING:   1.0,   # 最重要：直接指向
    GestureIntent.WRITING:    0.85,  # 很重要：正在寫
    GestureIntent.EMPHASIS:   0.90,  # 很重要：強調
    GestureIntent.TRANSITION: 0.20,  # 不重要：換場
    GestureIntent.IDLE:       0.10,  # 不重要：無手部
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
    """分類結果"""
    intent: GestureIntent
    confidence: float          # 最高類別的機率
    intent_score: float        # 對應的重要性分數
    label: str                 # 中文標籤
    probabilities: list[float] # 全部 5 類的機率分布


# ──────────────────────────────────────────────────────────────────
# GRU 模型架構
# ──────────────────────────────────────────────────────────────────

class GestureGRU(nn.Module):
    """
    雙層 GRU 手勢分類器

    輸入：(batch, seq_len, input_size=4)
    輸出：(batch, num_classes=5)
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 雙層 GRU，第一層 → 第二層
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 分類頭：GRU 最後一個時步的輸出 → 5 類
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 4)
        out, _ = self.gru(x)           # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]           # 取最後一個時步
        logits = self.classifier(last) # (batch, num_classes)
        return logits


# ──────────────────────────────────────────────────────────────────
# 特徵萃取
# ──────────────────────────────────────────────────────────────────

def extract_trajectory_features(
    coords: list[tuple[float, float]],
    frame_w: int = 640,
    frame_h: int = 480,
) -> np.ndarray:
    """
    把食指座標歷史轉換成模型輸入特徵

    輸入：coords = [(x0,y0), (x1,y1), ...]，長度任意
    輸出：shape (len(coords), 4)，每行是 [x_norm, y_norm, dx_norm, dy_norm]

    dx/dy 是相鄰幀的位移，用來捕捉速度資訊。
    所有值都歸一化到 [-1, 1] 或 [0, 1]。
    """
    if not coords:
        return np.zeros((1, 4), dtype=np.float32)

    coords_arr = np.array(coords, dtype=np.float32)

    # 位置歸一化到 [0, 1]
    x_norm = coords_arr[:, 0] / max(frame_w, 1)
    y_norm = coords_arr[:, 1] / max(frame_h, 1)

    # 速度（相鄰幀位移）
    if len(coords) > 1:
        dx = np.diff(x_norm, prepend=x_norm[0])
        dy = np.diff(y_norm, prepend=y_norm[0])
    else:
        dx = np.zeros_like(x_norm)
        dy = np.zeros_like(y_norm)

    # 速度歸一化（用全局最大值壓縮到 [-1, 1]）
    max_speed = max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
    dx = np.clip(dx / max_speed, -1.0, 1.0)
    dy = np.clip(dy / max_speed, -1.0, 1.0)

    features = np.stack([x_norm, y_norm, dx, dy], axis=1).astype(np.float32)
    return features


def pad_or_truncate(features: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """把特徵序列統一截斷或零填充到 seq_len"""
    n = features.shape[0]
    if n >= seq_len:
        return features[-seq_len:]   # 取最近的 seq_len 幀
    else:
        pad = np.zeros((seq_len - n, features.shape[1]), dtype=np.float32)
        return np.vstack([pad, features])  # 前面補零


# ──────────────────────────────────────────────────────────────────
# 規則型後備分類器（無模型時使用）
# ──────────────────────────────────────────────────────────────────

def rule_based_classify(
    coords: list[tuple[float, float]],
    variance_threshold: float = 50.0,
) -> GestureResult:
    """
    當訓練好的模型不存在時，使用手工規則做分類。
    這個函式的邏輯比原本的「只看變異數」更細緻。

    規則：
      1. 沒有座標 → IDLE
      2. 整體位移很小（var < threshold*0.5）→ POINTING
      3. 整體位移中等但路徑規律（自相關高）→ WRITING
      4. 在局部區域反覆來回（displacement reversal 多）→ EMPHASIS
      5. 整體位移大且單向 → TRANSITION
    """
    if len(coords) < 3:
        return _make_result(GestureIntent.IDLE, 0.9)

    coords_arr = np.array(coords, dtype=np.float32)
    xs, ys = coords_arr[:, 0], coords_arr[:, 1]
    var = float(np.var(xs) + np.var(ys))

    # 計算速度方向反轉次數（EMPHASIS 的特徵）
    if len(coords) > 2:
        dxs = np.diff(xs)
        reversal_count = int(np.sum(dxs[:-1] * dxs[1:] < 0))
    else:
        reversal_count = 0

    # 計算路徑總長度
    if len(coords) > 1:
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        path_length = float(dists.sum())
    else:
        path_length = 0.0

    # 端點位移（整體移動量）
    displacement = float(np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2))

    if var < variance_threshold * 0.5:
        return _make_result(GestureIntent.POINTING, min(1.0, 1.0 - var / (variance_threshold * 0.5)))
    elif reversal_count >= 3 and path_length > 20:
        return _make_result(GestureIntent.EMPHASIS, 0.75)
    elif var < variance_threshold * 2 and path_length < 80:
        return _make_result(GestureIntent.WRITING, 0.70)
    elif displacement > 100:
        return _make_result(GestureIntent.TRANSITION, 0.80)
    else:
        return _make_result(GestureIntent.POINTING, 0.50)


def _make_result(intent: GestureIntent, confidence: float) -> GestureResult:
    probs = [0.1 / 4] * 5
    probs[int(intent)] = confidence
    # 讓其他類別機率加總等於 1
    rest = (1.0 - confidence) / 4
    for i in range(5):
        if i != int(intent):
            probs[i] = rest
    return GestureResult(
        intent=intent,
        confidence=confidence,
        intent_score=INTENT_SCORE_MAP[intent],
        label=INTENT_LABELS[intent],
        probabilities=probs,
    )


# ──────────────────────────────────────────────────────────────────
# 主分類器介面
# ──────────────────────────────────────────────────────────────────

class GestureClassifier:
    """
    手勢意圖分類器主介面

    優先使用訓練好的 GRU 模型（weights/gesture_classifier.pth）；
    若模型不存在，自動降級為規則型分類器（不影響主流程運作）。

    使用方式：
        classifier = GestureClassifier()
        result = classifier.classify(coord_history, frame_w=1280, frame_h=720)
        print(result.label, result.intent_score)
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../weights/gesture_classifier.pth")
    SEQ_LEN = 20      # 輸入序列長度（幀數）
    INPUT_SIZE = 4    # 特徵維度

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[GestureGRU] = None
        self._mode = "rule"  # "gru" or "rule"
        self._load_model()

    def _load_model(self):
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
        """
        對一段座標歷史進行手勢意圖分類。

        Args:
            coord_history:  HandTracker 的 coord_history（食指座標序列）
            frame_w/h:      畫面尺寸（用於特徵歸一化）
            hand_detected:  本幀是否偵測到手部

        Returns:
            GestureResult，含意圖類別、信心度、重要性分數
        """
        if not hand_detected or len(coord_history) < 2:
            return _make_result(GestureIntent.IDLE, 0.95)

        if self._mode == "gru":
            return self._classify_gru(coord_history, frame_w, frame_h)
        else:
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
        confidence = float(probs[intent_idx])

        return GestureResult(
            intent=intent,
            confidence=confidence,
            intent_score=INTENT_SCORE_MAP[intent],
            label=INTENT_LABELS[intent],
            probabilities=probs.tolist(),
        )

    @property
    def mode(self) -> str:
        return self._mode
