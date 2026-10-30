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
    hand_variance: Optional[float] = None  # 手部位移變異數；None 表示沒有足夠追蹤資料
    hand_confidence: Optional[float] = None  # MediaPipe 偵測信心度 [0, 1]
    text_density: float = 0.0            # 畫面板書文字框密度 [0, 1]
    hand_board_proximity: float = 0.0    # 手部與板書區域接近度 [0, 1]
    hand_detected: bool = False          # 本幀是否偵測到手部
    visual_available: bool = True        # 本幀是否有對應的有效視覺分數


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
        if not np.isfinite(ema_alpha) or not 0.0 <= ema_alpha <= 1.0:
            raise ValueError("ema_alpha 必須介於 0 與 1")
        if (
            not np.isfinite(min_reliability)
            or not 0.0 <= min_reliability <= 1.0
        ):
            raise ValueError("min_reliability 必須介於 0 與 1")
        self.ema_alpha = float(ema_alpha)
        self.min_reliability = float(min_reliability)
        self._ema_reliability: Optional[float] = None  # 初始未校準

    # ─────────────────────────────────────────────────────────────
    # 公開介面
    # ─────────────────────────────────────────────────────────────

    def estimate(
        self,
        signals: ReliabilitySignals,
        base_beta: float = 0.5,
    ) -> ReliabilityResult:
        """計算本幀可靠度，並動態調整視覺權重 β_eff。

        base_beta 由呼叫端提供（主流程使用 config.py 的目前基準值）；
        它不是已驗證的最佳值。β_eff = base_beta × R_visual，
        可靠度低時自動縮小視覺權重。
        """
        if not np.isfinite(base_beta):
            raise ValueError("base_beta 必須是有限數值")
        base_beta = float(np.clip(base_beta, 0.0, 1.0))
        raw_reliability, dominant = self._compute_raw_reliability(signals)

        if raw_reliability == 0.0:
            # Missing observations are not low-quality observations; never carry
            # a previous frame's confidence into a frame with no usable score.
            self._ema_reliability = None
            r = 0.0
        else:
            if self._ema_reliability is None:
                self._ema_reliability = raw_reliability
            else:
                self._ema_reliability = (
                    self.ema_alpha * raw_reliability
                    + (1 - self.ema_alpha) * self._ema_reliability
                )
            r = max(self.min_reliability, min(1.0, self._ema_reliability))

        beta_eff = round(base_beta * r, 4)

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
    def _hand_stability_score(variance: Optional[float]) -> float:
        """手部位移變異數轉穩定性分數（指數衰減映射）。"""
        if variance is None or not np.isfinite(variance) or variance < 0:
            return 0.0
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
        """板書文字框聯集面積佔畫面比例，避免重疊框重複計算。"""
        if frame_area <= 0 or not text_boxes:
            return 0.0
        boxes = [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2 in text_boxes
            if x2 > x1 and y2 > y1
        ]
        if not boxes:
            return 0.0

        x_edges = sorted({edge for x1, _, x2, _ in boxes for edge in (x1, x2)})
        total_text_area = 0.0
        for left, right in zip(x_edges, x_edges[1:]):
            if right <= left:
                continue
            intervals = sorted(
                (y1, y2)
                for x1, y1, x2, y2 in boxes
                if x1 < right and x2 > left
            )
            covered_height = 0
            if intervals:
                interval_start, interval_end = intervals[0]
                for start, end in intervals[1:]:
                    if start <= interval_end:
                        interval_end = max(interval_end, end)
                    else:
                        covered_height += interval_end - interval_start
                        interval_start, interval_end = start, end
                covered_height += interval_end - interval_start
            total_text_area += (right - left) * covered_height

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
        if not signals.visual_available:
            return 0.0, "visual_score_unavailable"
        if not signals.hand_detected and signals.text_density <= 0:
            return 0.0, "no_hand_detected"

        sub_scores = {}
        if signals.hand_detected:
            if (
                signals.hand_variance is not None
                and np.isfinite(signals.hand_variance)
                and signals.hand_variance >= 0
            ):
                sub_scores["手部穩定性"] = self._hand_stability_score(
                    signals.hand_variance
                )
            if (
                signals.hand_confidence is not None
                and np.isfinite(signals.hand_confidence)
            ):
                sub_scores["偵測信心度"] = float(
                    np.clip(signals.hand_confidence, 0.0, 1.0)
                )
        if np.isfinite(signals.text_density):
            sub_scores["板書文字密度"] = float(
                np.clip(signals.text_density, 0.0, 1.0)
            )
        if signals.hand_detected and np.isfinite(signals.hand_board_proximity):
            sub_scores["手板接近度"] = float(
                np.clip(signals.hand_board_proximity, 0.0, 1.0)
            )
        if not sub_scores:
            return 0.0, "no_valid_signals"

        weights = {
            "手部穩定性": self.SIGNAL_WEIGHTS["hand_stability"],
            "偵測信心度": self.SIGNAL_WEIGHTS["hand_confidence"],
            "板書文字密度": self.SIGNAL_WEIGHTS["text_density"],
            "手板接近度": self.SIGNAL_WEIGHTS["proximity"],
        }
        # 缺少的訊號不當作 0；只對可用訊號重整權重，避免缺測被誤判為低品質。
        available_weight = sum(weights[key] for key in sub_scores)
        raw = sum(weights[key] * value for key, value in sub_scores.items())
        raw /= available_weight
        dominant = min(sub_scores, key=sub_scores.get)
        return float(np.clip(raw, 0.0, 1.0)), dominant


def build_signals_from_tracker_result(
    tracker_result: dict,
    text_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    frame_w: int = 640,
    frame_h: int = 480,
) -> ReliabilitySignals:
    """建構訊號；手勢意圖不作為追蹤信心度或位移穩定性的替代值。"""
    text_boxes = text_boxes or []
    triggered = tracker_result.get("triggered", False)
    hand_detected = tracker_result.get("hand_detected", triggered)
    hand_center = tracker_result.get("hand_center", tracker_result.get("roi_center"))

    # 直接呼叫靜態方法，不需建立完整的 VisualReliabilityEstimator 物件
    text_density = VisualReliabilityEstimator._text_density_score(
        text_boxes, frame_w * frame_h
    )
    proximity = VisualReliabilityEstimator._proximity_score(
        hand_center if hand_detected else None, text_boxes, frame_w, frame_h
    )

    # HandTracker 目前未輸出 landmark confidence，因此該訊號會缺席並由上述規則重整權重。
    return ReliabilitySignals(
        hand_variance=tracker_result.get("hand_variance"),
        hand_confidence=tracker_result.get("hand_confidence"),
        text_density=text_density,
        hand_board_proximity=proximity,
        hand_detected=bool(hand_detected),
        visual_available=bool(
            tracker_result.get(
                "visual_available",
                triggered and tracker_result.get("roi_center"),
            )
        ),
    )
