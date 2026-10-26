"""
視覺可靠度估測模組

對每個視覺觀測估計可靠度 R_visual ∈ [0,1]，
用以動態折扣融合公式中的視覺權重 β：

  β_eff = β × R_visual
  α_eff = 1 − β_eff
  Score = α_eff × s_text + β_eff × s_visual

R_visual 由四項可觀測訊號加權估計：
  手部穩定度（0.35）、偵測信心度（0.25）、
  板書文字密度（0.25）、手板接近度（0.15）

加上指數移動平均（EMA）平滑，避免單幀跳動干擾穩定性。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import (
    HAND_VARIANCE_THRESHOLD,
    IOU_FALLBACK_THRESHOLD,
    TEXT_DETECTION_CONFIDENCE_THRESHOLD,
)


@dataclass
class ReliabilitySignals:
    """視覺可靠度的原始訊號"""
    hand_variance: float = 0.0           # 手部位移變異數（越低越穩定）
    hand_confidence: float = 0.0         # MediaPipe 偵測信心度 [0, 1]
    text_density: float = 0.0            # 畫面板書文字框密度 [0, 1]
    hand_board_proximity: float = 0.0    # 手部與板書區域接近度 [0, 1]
    hand_detected: bool = False          # 本幀是否偵測到手部


@dataclass
class ReliabilityResult:
    """視覺可靠度估測結果"""
    reliability: float                   # 整體可靠度 R_visual ∈ [0, 1]
    signals: ReliabilitySignals
    dominant_factor: str                 # 主要影響因素說明
    effective_beta: float = 0.5          # 動態調整後的視覺權重 β_eff


class VisualReliabilityEstimator:
    """
    視覺可靠度估測器

    多維度訊號融合，以 EMA 平滑，輸出 R_visual ∈ [0,1]。
    可直接插入現有 pipeline，不影響其他模組。
    """

    # 各訊號的組合權重（加總為 1）
    SIGNAL_WEIGHTS = {
        "hand_stability": 0.35,     # 手部穩定性（低變異數）
        "hand_confidence": 0.25,    # MediaPipe 偵測信心度
        "text_density": 0.25,       # 板書文字密度
        "proximity": 0.15,          # 手部與板書接近度
    }

    def __init__(
        self,
        ema_alpha: float = 0.3,
        min_reliability: float = 0.05,
    ):
        """
        Args:
            ema_alpha:        EMA 平滑係數（越大越敏感，越小越穩定）
            min_reliability:  可靠度下限（避免視覺權重歸零導致退化為純語意）
        """
        self.ema_alpha = ema_alpha
        self.min_reliability = min_reliability
        self._ema_reliability: Optional[float] = None  # 初始未校準

    # ─────────────────────────────────────────────────────────────
    # 公開介面
    # ─────────────────────────────────────────────────────────────

    def estimate(
        self,
        signals: ReliabilitySignals,
        base_beta: float = 0.5,
    ) -> ReliabilityResult:
        """
        根據訊號計算本幀可靠度，並動態調整視覺權重 β_eff。

        Args:
            signals:    由 hand_tracker / visual_scorer 收集的訊號
            base_beta:  config.py 的靜態 β（Grid Search 最佳化結果）

        Returns:
            ReliabilityResult，含最終 β_eff 與解釋
        """
        raw_reliability, dominant = self._compute_raw_reliability(signals)

        # EMA 平滑
        if self._ema_reliability is None:
            self._ema_reliability = raw_reliability
        else:
            self._ema_reliability = (
                self.ema_alpha * raw_reliability
                + (1 - self.ema_alpha) * self._ema_reliability
            )

        # 夾在 [min_reliability, 1.0]
        r = max(self.min_reliability, min(1.0, self._ema_reliability))

        # 動態 β_eff：以可靠度線性縮放靜態 β（用 min 確保不超過 base_beta）
        beta_eff = round(min(base_beta, base_beta * r), 4)

        return ReliabilityResult(
            reliability=round(r, 4),
            signals=signals,
            dominant_factor=dominant,
            effective_beta=beta_eff,
        )

    def reset(self):
        """重置 EMA（開始處理新影片時呼叫）"""
        self._ema_reliability = None

    # ─────────────────────────────────────────────────────────────
    # 訊號→分數轉換（各子函式皆回傳 [0, 1]）
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _hand_stability_score(variance: float) -> float:
        """手部位移變異數轉穩定性分數（指數衰減映射）。"""
        if variance <= 0:
            return 1.0
        threshold = HAND_VARIANCE_THRESHOLD
        # 指數衰減映射
        score = np.exp(-variance / threshold)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _text_density_score(
        text_boxes: list[tuple[int, int, int, int]],
        frame_area: int,
    ) -> float:
        """板書文字框面積佔畫面比例（對數壓縮）。"""
        if frame_area <= 0 or not text_boxes:
            return 0.0
        total_text_area = sum(
            max(0, (x2 - x1)) * max(0, (y2 - y1))
            for x1, y1, x2, y2 in text_boxes
        )
        raw_ratio = total_text_area / frame_area
        # 對數壓縮：避免文字非常密時分數過高
        score = min(1.0, np.log1p(raw_ratio * 20) / np.log1p(20))
        return float(score)

    @staticmethod
    def _proximity_score(
        hand_center: Optional[tuple[int, int]],
        text_boxes: list[tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
    ) -> float:
        """食指中心點與最近文字框的距離轉接近度分數（指數衰減）。"""
        if hand_center is None or not text_boxes:
            return 0.0

        hx, hy = hand_center
        diag = np.sqrt(frame_w ** 2 + frame_h ** 2)
        if diag <= 0:
            return 0.0

        min_dist = float("inf")
        for x1, y1, x2, y2 in text_boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
            min_dist = min(min_dist, dist)

        normalized_dist = min_dist / diag
        # 距離 0 → 1.0；距離 diag → 0.0，指數衰減
        score = np.exp(-normalized_dist * 4)
        return float(np.clip(score, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────
    # 聚合
    # ─────────────────────────────────────────────────────────────

    def _compute_raw_reliability(
        self, signals: ReliabilitySignals
    ) -> tuple[float, str]:
        """將各子分數依 SIGNAL_WEIGHTS 加權合併，回傳（原始可靠度, 主要影響因素）。"""
        if not signals.hand_detected:
            return 0.0, "no_hand_detected"

        stability = self._hand_stability_score(signals.hand_variance)
        confidence = float(np.clip(signals.hand_confidence, 0.0, 1.0))
        density = float(np.clip(signals.text_density, 0.0, 1.0))
        proximity = float(np.clip(signals.hand_board_proximity, 0.0, 1.0))

        w = self.SIGNAL_WEIGHTS
        raw = (
            w["hand_stability"] * stability
            + w["hand_confidence"] * confidence
            + w["text_density"] * density
            + w["proximity"] * proximity
        )

        # 找出主要影響因素（最低的子分數決定瓶頸）
        sub_scores = {
            "手部穩定性": stability,
            "偵測信心度": confidence,
            "板書文字密度": density,
            "手板接近度": proximity,
        }
        dominant = min(sub_scores, key=sub_scores.get)

        return float(raw), dominant


def build_signals_from_tracker_result(
    tracker_result: dict,
    text_boxes: list[tuple[int, int, int, int]] = None,
    frame_w: int = 640,
    frame_h: int = 480,
) -> ReliabilitySignals:
    """
    從 HandTracker.process_frame() 回傳的 dict 和文字框列表
    建構 ReliabilitySignals，供 VisualReliabilityEstimator 使用。

    Args:
        tracker_result: HandTracker.process_frame() 的回傳值
        text_boxes:     TextDetector 偵測到的文字框列表
        frame_w/h:      畫面尺寸（計算接近度距離用）
    """
    text_boxes = text_boxes or []
    triggered = tracker_result.get("triggered", False)
    roi_center = tracker_result.get("roi_center")

    # 計算文字密度
    estimator = VisualReliabilityEstimator()
    text_density = estimator._text_density_score(text_boxes, frame_w * frame_h)

    # 計算接近度
    proximity = estimator._proximity_score(
        roi_center, text_boxes, frame_w, frame_h
    )

    # 從 s_visual_raw 反推 variance（近似）
    s_raw = tracker_result.get("s_visual_raw", 0.0)
    # s_visual_raw = max(0, 1 - variance / (2 * threshold))
    # → variance ≈ (1 - s_raw) * 2 * threshold
    approx_variance = (1.0 - min(s_raw, 1.0)) * 2.0 * HAND_VARIANCE_THRESHOLD

    return ReliabilitySignals(
        hand_variance=approx_variance,
        hand_confidence=s_raw,        # 用 s_visual_raw 作為信心度代理
        text_density=text_density,
        hand_board_proximity=proximity,
        hand_detected=triggered or (s_raw > 0),
    )
