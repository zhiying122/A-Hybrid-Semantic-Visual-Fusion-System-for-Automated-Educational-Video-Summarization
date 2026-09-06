"""
步驟四：多模態融合與動態剪輯
─────────────────────────────────────────────────────────────────────
v2 改進（回應評審意見）：新增「自適應置信度加權晚期融合」機制

舊版（靜態加權）：
    Score = α * S_text + β * S_visual

v2（自適應置信度加權）：
    β_eff  = β * R_visual          ← R_visual 由 VisualReliabilityEstimator 估算
    α_eff  = 1 - β_eff             ← 語意權重自動補足，確保加總為 1
    Score  = α_eff * S_text + β_eff * S_visual

核心創新：
  - 當手勢不在板書前（R_visual 低）→ 自動提高語意權重，避免雜訊污染摘要
  - 當板書密度高且手部穩定（R_visual 高）→ 視覺權重自動提升，充分利用視覺資訊
  - 理論依據：Uncertainty-aware Multimodal Fusion，以可觀測信號作為不確定性代理
  - 與靜態加權完全相容：R_visual = 1.0 時退化為原始 α/β

其他功能（同 v1）：
- Grid Search 找最佳 α/β（在驗證集上）
- 留一交叉驗證（Leave-One-Out）確保泛化性
- 語意感知滑動視窗保持片段完整性

對應計畫書 4.2 步驟四
"""

import time
import numpy as np
from itertools import product
from config import ALPHA, BETA, FUSION_SCORE_THRESHOLD, SLIDING_WINDOW_MIN_SEC

try:
    from config import MAX_SUMMARY_RATIO, MIN_SUMMARY_SEC
except ImportError:
    MAX_SUMMARY_RATIO = 0.6
    MIN_SUMMARY_SEC = 2.0

# 全域進度回調（由 api.py 注入）
_progress_callback = None

def set_progress_callback(fn):
    global _progress_callback
    _progress_callback = fn

def _report(step: int, total: int, message: str):
    if _progress_callback:
        _progress_callback(step, total, message)


def fuse_scores(
    s_text: float,
    s_visual: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    visual_reliability: float = 1.0,
) -> float:
    """
    自適應置信度加權晚期融合（Adaptive Confidence-Weighted Late Fusion）

    Args:
        s_text:             語意分數 S_text ∈ [0, 1]
        s_visual:           視覺分數 S_visual ∈ [0, 1]
        alpha:              靜態語意權重 α（由 Grid Search 取得）
        beta:               靜態視覺權重 β（由 Grid Search 取得）
        visual_reliability: 視覺可靠度 R_visual ∈ [0, 1]
                            （由 VisualReliabilityEstimator 估算；預設 1.0 = 靜態模式）

    Returns:
        融合分數 ∈ [0, 1]

    融合公式：
        β_eff  = β × R_visual
        α_eff  = 1 − β_eff
        Score  = α_eff × S_text + β_eff × S_visual
    """
    visual_reliability = float(np.clip(visual_reliability, 0.0, 1.0))
    beta_eff = beta * visual_reliability
    alpha_eff = 1.0 - beta_eff
    return float(alpha_eff * s_text + beta_eff * s_visual)


def grid_search_weights(
    segments: list[dict],
    ground_truth: list[tuple[float, float]],
    step: float = 0.1
) -> tuple[float, float]:
    """
    在驗證集上以 Grid Search 找最佳 α/β 組合（最大化 Recall）
    ground_truth: [(start, end), ...] 人工標註的重點片段
    """
    best_alpha, best_beta, best_recall = 0.5, 0.5, 0.0

    for a in np.arange(0.0, 1.0 + step, step):
        b = round(1.0 - a, 1)
        scores = [fuse_scores(seg["s_text"], seg.get("s_visual", 0.0), a, b) for seg in segments]
        selected = [seg for seg, sc in zip(segments, scores) if sc >= FUSION_SCORE_THRESHOLD]
        recall = _compute_recall(selected, ground_truth)
        if recall > best_recall:
            best_recall, best_alpha, best_beta = recall, a, b

    print(f"[GridSearch] 最佳 α={best_alpha:.1f}, β={best_beta:.1f}, Recall={best_recall:.3f}")
    return best_alpha, best_beta


def _compute_recall(selected: list[dict], ground_truth: list[tuple[float, float]]) -> float:
    """計算重點召回率：被選中的 GT 片段數 / 全部 GT 片段數"""
    if not ground_truth:
        return 0.0
    hit = 0
    for gt_start, gt_end in ground_truth:
        for seg in selected:
            # 時間重疊超過 50% 視為命中
            overlap = min(seg["end"], gt_end) - max(seg["start"], gt_start)
            gt_len = gt_end - gt_start
            if gt_len > 0 and overlap / gt_len >= 0.5:
                hit += 1
                break
    return hit / len(ground_truth)


def leave_one_out_cv(all_videos: list[dict]) -> list[dict]:
    """
    留一交叉驗證：輪流以一部影片為測試集，其餘為訓練集
    all_videos: [{"segments": [...], "ground_truth": [...]}]
    回傳每折的最佳參數與 Recall
    """
    results = []
    for i, test_video in enumerate(all_videos):
        train_videos = [v for j, v in enumerate(all_videos) if j != i]
        # 合併訓練集片段
        train_segs = [seg for v in train_videos for seg in v["segments"]]
        train_gt = [gt for v in train_videos for gt in v["ground_truth"]]
        alpha, beta = grid_search_weights(train_segs, train_gt)
        # 在測試集評估
        test_scores = [fuse_scores(s["s_text"], s.get("s_visual", 0.0), alpha, beta)
                       for s in test_video["segments"]]
        selected = [s for s, sc in zip(test_video["segments"], test_scores)
                    if sc >= FUSION_SCORE_THRESHOLD]
        recall = _compute_recall(selected, test_video["ground_truth"])
        results.append({"fold": i, "alpha": alpha, "beta": beta, "recall": recall})
        print(f"[LOOCV] Fold {i}: α={alpha}, β={beta}, Recall={recall:.3f}")
    return results


def semantic_sliding_window(
    segments: list[dict],
    scores: list[float],
    min_sec: float = SLIDING_WINDOW_MIN_SEC,
    max_summary_ratio: float = MAX_SUMMARY_RATIO,
    min_summary_sec: float = MIN_SUMMARY_SEC,
) -> list[dict]:
    """
    語意感知滑動視窗（含「保證濃縮」機制）：
    鎖定重點時間點後，依 Whisper 斷句向前後延伸，確保板書推導與語句完整性。

    保證濃縮（不論影片長度，輸出必定比原片短）：
      1. 短影片保護：若「原片 × max_summary_ratio」比 min_sec 還短，
         自動縮小最小片段長度，避免單段就吃掉整部片。
      2. 總長度上限：摘要總時長不得超過「原片 × max_summary_ratio」。
         若門檻選段超出，依融合分數由高到低保留片段直到符合上限。
      3. 保底輸出：若門檻導致一段都沒選到，至少保留分數最高的一段
         （長度不超過上限，且不短於 min_summary_sec），確保永遠有輸出。

    Args:
        segments:          片段列表（需含 start/end），需與 scores 等長且時間遞增
        scores:            各片段融合分數
        min_sec:           滑動視窗最小保留秒數
        max_summary_ratio: 摘要總時長上限比例（相對原片），∈ (0, 1)
        min_summary_sec:   摘要最短輸出秒數（保底）

    Returns:
        選中片段列表（依時間排序），總時長 ≤ 原片 × max_summary_ratio
    """
    if not segments:
        return []

    # 原片時長（以片段時間範圍估算）
    original_duration = max(s["end"] for s in segments) - min(s["start"] for s in segments)
    max_summary_ratio = float(np.clip(max_summary_ratio, 0.05, 0.95))
    budget = original_duration * max_summary_ratio  # 允許的最大摘要時長

    # 短影片保護：預算比最小片段還短時，縮小最小片段長度
    effective_min_sec = min_sec
    if budget < min_sec:
        effective_min_sec = max(min_summary_sec, budget * 0.5)

    def _make(idx_start: int, seg_start: float, seg_end: float) -> dict:
        return {
            "start": seg_start,
            "end": seg_end,
            "s_text": segments[idx_start].get("s_text", 0.0),
            "s_visual": segments[idx_start].get("s_visual", 0.0),
            "text": segments[idx_start].get("text", ""),
            "score": scores[idx_start],
        }

    # ── 第一階段：門檻選段 + 語意延伸 ──────────────────
    selected = []
    i = 0
    while i < len(segments):
        if scores[i] >= FUSION_SCORE_THRESHOLD:
            start = segments[i]["start"]
            end = segments[i]["end"]
            j = i + 1
            while j < len(segments):
                candidate_end = segments[j]["end"]
                duration = candidate_end - start
                if duration >= effective_min_sec and scores[j] < FUSION_SCORE_THRESHOLD:
                    break
                end = candidate_end
                j += 1
            selected.append(_make(i, start, end))
            i = j
        else:
            i += 1

    # ── 保底：一段都沒選到時，取分數最高的單段 ─────────
    if not selected:
        best_idx = int(np.argmax(scores))
        start = segments[best_idx]["start"]
        end = segments[best_idx]["end"]
        # 目標長度：不超過預算，且保證嚴格短於原片（極短影片時取預算值）
        target_len = min(max(min_summary_sec, end - start), budget)
        target_len = min(target_len, original_duration * max_summary_ratio)
        # 若單一原始片段就 ≥ 目標長度，直接截斷；否則向後補
        if (end - start) >= target_len:
            end = start + target_len
        else:
            k = best_idx + 1
            while (end - start) < target_len and k < len(segments):
                end = segments[k]["end"]
                k += 1
            end = min(end, start + target_len)
        selected.append(_make(best_idx, start, end))
        return selected

    # ── 第二階段：總長度上限（保證濃縮）─────────────────
    total = sum(s["end"] - s["start"] for s in selected)
    if total <= budget:
        return selected

    # 超出預算：依分數由高到低貪婪保留，直到逼近上限
    ranked = sorted(selected, key=lambda s: s.get("score", 0.0), reverse=True)
    kept, used = [], 0.0
    for s in ranked:
        seg_len = s["end"] - s["start"]
        if used + seg_len <= budget:
            kept.append(s)
            used += seg_len
        if used >= budget:
            break

    # 極端情況：連最高分單段都超過預算 → 截斷最高分片段至預算長度
    # （預算優先，確保嚴格短於原片；極短影片時可能小於 min_summary_sec）
    if not kept:
        top = ranked[0]
        capped_len = min(budget, top["end"] - top["start"])
        kept = [{**top, "end": top["start"] + capped_len}]

    # 依時間排序回傳，維持播放順序
    kept.sort(key=lambda s: s["start"])
    return kept
