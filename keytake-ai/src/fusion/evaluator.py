"""
成效評估模組
─────────────────────────────────────────────────────────────────────
對應計畫書 4.4 量化指標：
1. 重點召回率 (Recall)
2. 誤報率 (False Alarm Rate)
3. 語意相似度 (BERTScore)
4. 時間節省率 (Time Saving Rate)

v2 改進：
  - 修復 bert_score_fn 測試 mock bug（改為 module-level import）
  - 新增精確度 (Precision) 與 F1-Score
"""

from config import TARGET_RECALL, TARGET_BERT_SCORE, TARGET_TIME_SAVING_RATE

# Module-level import 讓 mock 可以正確攔截
try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None


def compute_recall(selected: list[dict], ground_truth: list[tuple[float, float]]) -> float:
    """重點召回率：系統選中的 GT 片段比例"""
    if not ground_truth:
        return 0.0
    hit = 0
    for gt_start, gt_end in ground_truth:
        for seg in selected:
            overlap = min(seg["end"], gt_end) - max(seg["start"], gt_start)
            if (gt_end - gt_start) > 0 and overlap / (gt_end - gt_start) >= 0.5:
                hit += 1
                break
    return hit / len(ground_truth)


def compute_false_alarm_rate(selected: list[dict], ground_truth: list[tuple[float, float]],
                              total_duration: float) -> float:
    """誤報率：被選中但非重點的時間佔總時長比例"""
    false_alarm_sec = 0.0
    for seg in selected:
        is_gt = any(
            min(seg["end"], gt_end) - max(seg["start"], gt_start) > 0
            for gt_start, gt_end in ground_truth
        )
        if not is_gt:
            false_alarm_sec += seg["end"] - seg["start"]
    if total_duration <= 0:
        return 0.0
    return min(1.0, max(0.0, false_alarm_sec / total_duration))


def compute_bert_score(summary_texts: list[str], reference_texts: list[str]) -> float:
    """
    語意相似度：使用 BERTScore 計算摘要與原始重點的向量相似度
    
    Note: bert_score_fn 現為 module-level import，測試時可正確 mock
    """
    if bert_score_fn is None:
        raise ImportError("bert-score 未安裝，請執行：pip install bert-score")
    _, _, f1 = bert_score_fn(summary_texts, reference_texts, lang="zh", verbose=False)
    return float(f1.mean())


def compute_time_saving_rate(selected: list[dict], total_duration: float) -> float:
    """時間節省率：(原始時長 - 摘要時長) / 原始時長"""
    summary_duration = sum(seg["end"] - seg["start"] for seg in selected)
    return 1.0 - summary_duration / total_duration if total_duration > 0 else 0.0


def compute_precision(selected: list[dict], ground_truth: list[tuple[float, float]]) -> float:
    """
    精確度 (Precision)：選中片段與 GT 重疊的比例
    
    計算方式：選中片段中有多少比例真正命中 GT
    """
    if not selected:
        return 0.0
    
    hit = 0
    for seg in selected:
        for gt_start, gt_end in ground_truth:
            overlap = min(seg["end"], gt_end) - max(seg["start"], gt_start)
            if (gt_end - gt_start) > 0 and overlap / (gt_end - gt_start) >= 0.5:
                hit += 1
                break
    
    return hit / len(selected)


def compute_f1_score(recall: float, precision: float) -> float:
    """F1-Score：Recall 與 Precision 的調和平均"""
    if recall + precision <= 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def evaluate_all(selected, ground_truth, total_duration, summary_texts, reference_texts) -> dict:
    """
    一次計算所有指標並與目標對比
    
    v2 新增：Precision、F1-Score
    """
    recall = compute_recall(selected, ground_truth)
    precision = compute_precision(selected, ground_truth)
    f1 = compute_f1_score(recall, precision)
    far = compute_false_alarm_rate(selected, ground_truth, total_duration)
    bs = compute_bert_score(summary_texts, reference_texts)
    tsr = compute_time_saving_rate(selected, total_duration)

    results = {
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "false_alarm_rate": far,
        "bert_score": bs,
        "time_saving_rate": tsr,
    }

    print("\n===== 成效評估結果 =====")
    print(f"重點召回率 (Recall)     : {recall:.3f}  (目標 > {TARGET_RECALL})")
    print(f"精確度 (Precision)       : {precision:.3f}")
    print(f"F1-Score                 : {f1:.3f}")
    print(f"誤報率 (FAR)             : {far:.3f}  (目標 < 0.25)")
    print(f"語意相似度 (BERTScore)   : {bs:.3f}  (目標 > {TARGET_BERT_SCORE})")
    print(f"時間節省率 (TSR)         : {tsr:.3f}  (目標 > {TARGET_TIME_SAVING_RATE})")
    return results
