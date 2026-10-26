"""
多模態融合與動態剪輯模組

處理文本與視覺分數的自適應置信度晚期融合 (Adaptive Confidence-Weighted Late Fusion)，
並透過語意滑動視窗進行片段篩選，確保輸出符合長度限制與語意完整性。
"""

import numpy as np
from typing import Optional, Callable

from config import ALPHA, BETA, FUSION_SCORE_THRESHOLD, SLIDING_WINDOW_MIN_SEC

try:
    from config import MAX_SUMMARY_RATIO, MIN_SUMMARY_SEC
except ImportError:
    MAX_SUMMARY_RATIO = 0.6
    MIN_SUMMARY_SEC = 2.0

# 全域進度回調
_progress_callback: Optional[Callable[[int, int, str], None]] = None

def set_progress_callback(fn: Callable[[int, int, str], None]) -> None:
    global _progress_callback
    _progress_callback = fn

def _report(step: int, total: int, message: str) -> None:
    if _progress_callback:
        _progress_callback(step, total, message)


def fuse_scores(
    s_text: float,
    s_visual: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    visual_reliability: float = 1.0,
) -> float:
    """依據視覺可靠度動態調整視覺與文本分數的融合權重。"""
    # 對單一數值使用原生的 min/max 比 np.clip 快
    reliability = max(0.0, min(visual_reliability, 1.0))
    beta_eff = beta * reliability
    alpha_eff = 1.0 - beta_eff
    return alpha_eff * s_text + beta_eff * s_visual


def grid_search_weights(
    segments: list[dict],
    ground_truth: list[tuple[float, float]],
    step: float = 0.1
) -> tuple[float, float]:
    """在驗證集上搜尋最大化 Recall 的最佳 α/β 組合。"""
    best_alpha, best_beta, best_recall = 0.5, 0.5, 0.0

    for a in np.arange(0.0, 1.0 + step, step):
        b = round(1.0 - a, 1)
        scores = [fuse_scores(seg["s_text"], seg.get("s_visual", 0.0), a, b) for seg in segments]
        selected = [seg for seg, sc in zip(segments, scores) if sc >= FUSION_SCORE_THRESHOLD]
        recall = _compute_recall(selected, ground_truth)
        
        if recall > best_recall:
            best_recall, best_alpha, best_beta = recall, a, b

    print(f"[GridSearch] Best α={best_alpha:.1f}, β={best_beta:.1f}, Recall={best_recall:.3f}")
    return best_alpha, best_beta


def _compute_recall(selected: list[dict], ground_truth: list[tuple[float, float]]) -> float:
    """計算被選中的 Ground Truth 比例（時間重疊 ≥ 50% 視為命中）。"""
    if not ground_truth:
        return 0.0
    
    hit = 0
    for gt_start, gt_end in ground_truth:
        gt_len = gt_end - gt_start
        if gt_len <= 0:
            continue
            
        for seg in selected:
            overlap = min(seg["end"], gt_end) - max(seg["start"], gt_start)
            if overlap / gt_len >= 0.5:
                hit += 1
                break
    return hit / len(ground_truth)


def leave_one_out_cv(all_videos: list[dict]) -> list[dict]:
    """留一交叉驗證 (LOOCV) 評估模型泛化能力。"""
    results = []
    for i, test_video in enumerate(all_videos):
        train_videos = [v for j, v in enumerate(all_videos) if j != i]
        
        train_segs = [seg for v in train_videos for seg in v["segments"]]
        train_gt = [gt for v in train_videos for gt in v["ground_truth"]]
        
        alpha, beta = grid_search_weights(train_segs, train_gt)
        
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
    基於分數門檻與語意邊界的滑動視窗選段。
    確保總時長不超過比例上限，並具備極端情況的保底輸出機制。
    """
    if not segments:
        return []

    original_duration = max(s["end"] for s in segments) - min(s["start"] for s in segments)
    max_ratio = max(0.05, min(max_summary_ratio, 0.95))
    budget = original_duration * max_ratio

    # 處理短影片邊界條件
    effective_min_sec = min_sec if budget >= min_sec else max(min_summary_sec, budget * 0.5)
    
    # 取固定門檻與分數中位數之較大者，避免整體分數偏高時選入過多片段
    select_threshold = FUSION_SCORE_THRESHOLD
    if len(scores) >= 5:
        select_threshold = max(select_threshold, float(np.median(scores)))

    max_segment_sec = max(effective_min_sec * 2.0, min_summary_sec)

    def _make_segment(idx_start: int, seg_start: float, seg_end: float) -> dict:
        return {
            "start": seg_start,
            "end": seg_end,
            "s_text": segments[idx_start].get("s_text", 0.0),
            "s_visual": segments[idx_start].get("s_visual", 0.0),
            "text": segments[idx_start].get("text", ""),
            "score": scores[idx_start],
        }

    # 1. 篩選高分片段並沿語意向後延伸
    selected = []
    i = 0
    while i < len(segments):
        if scores[i] >= select_threshold:
            start, end = segments[i]["start"], segments[i]["end"]
            j = i + 1
            
            while j < len(segments):
                candidate_end = segments[j]["end"]
                duration = candidate_end - start
                
                if (duration >= effective_min_sec and scores[j] < select_threshold) or duration > max_segment_sec:
                    break
                
                end = candidate_end
                j += 1
                
            selected.append(_make_segment(i, start, end))
            i = j
        else:
            i += 1

    # 2. 保底機制：若無片段達標，強制保留最高分單段
    if not selected:
        best_idx = int(np.argmax(scores))
        start, end = segments[best_idx]["start"], segments[best_idx]["end"]
        
        target_len = min(max(min_summary_sec, end - start), budget)
        
        if (end - start) < target_len:
            k = best_idx + 1
            while (end - start) < target_len and k < len(segments):
                end = segments[k]["end"]
                k += 1
                
        end = min(end, start + target_len)
        return [_make_segment(best_idx, start, end)]

    # 3. 預算控管：若超出總時長上限，依分數貪婪保留
    total_len = sum(s["end"] - s["start"] for s in selected)
    if total_len <= budget:
        return selected

    ranked = sorted(selected, key=lambda s: s.get("score", 0.0), reverse=True)
    kept, used = [], 0.0
    
    for s in ranked:
        seg_len = s["end"] - s["start"]
        if used + seg_len <= budget:
            kept.append(s)
            used += seg_len
            
    if not kept:
        top = ranked[0]
        capped_len = min(budget, top["end"] - top["start"])
        kept = [{**top, "end": top["start"] + capped_len}]

    kept.sort(key=lambda s: s["start"])
    return kept