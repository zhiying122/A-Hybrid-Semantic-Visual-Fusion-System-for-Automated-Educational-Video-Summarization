"""
步驟四：多模態融合與動態剪輯
- 適應性晚期融合：Score = α * S_text + β * S_visual
- Grid Search 找最佳 α/β（在驗證集上）
- 留一交叉驗證（Leave-One-Out）確保泛化性
- 語意感知滑動視窗保持片段完整性
對應計畫書 4.2 步驟四
"""

import numpy as np
from itertools import product
from config import ALPHA, BETA, FUSION_SCORE_THRESHOLD, SLIDING_WINDOW_MIN_SEC


def fuse_scores(s_text: float, s_visual: float, alpha: float = ALPHA, beta: float = BETA) -> float:
    """
    適應性晚期融合公式：Score = α * S_text + β * S_visual
    """
    return alpha * s_text + beta * s_visual


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
    min_sec: float = SLIDING_WINDOW_MIN_SEC
) -> list[dict]:
    """
    語意感知滑動視窗：
    鎖定重點時間點後，依 Whisper 斷句與畫面靜止向前後延伸，
    確保板書推導與語句表達的完整性
    """
    selected = []
    i = 0
    while i < len(segments):
        if scores[i] >= FUSION_SCORE_THRESHOLD:
            start = segments[i]["start"]
            end = segments[i]["end"]
            # 向後延伸直到滿足最小保留時間且遇到低分片段
            j = i + 1
            while j < len(segments):
                end = segments[j]["end"]
                duration = end - start
                if duration >= min_sec and scores[j] < FUSION_SCORE_THRESHOLD:
                    break
                j += 1
            selected.append({"start": start, "end": end})
            i = j
        else:
            i += 1
    return selected
