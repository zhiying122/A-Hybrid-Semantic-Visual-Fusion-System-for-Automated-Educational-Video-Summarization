"""
效能評估模組

計算系統量化指標，包含 Recall, Precision, F1, FAR, TSR 與 BERTScore。
"""

from typing import List, Tuple, Dict
from config import TARGET_RECALL, TARGET_BERT_SCORE, TARGET_TIME_SAVING_RATE

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None


def _is_hit(seg: dict, gt_start: float, gt_end: float, threshold: float = 0.5) -> bool:
    """判斷單片段與 Ground Truth 是否達到重疊門檻"""
    gt_len = gt_end - gt_start
    if gt_len <= 0:
        return False
    overlap = min(seg["end"], gt_end) - max(seg["start"], gt_start)
    return (overlap / gt_len) >= threshold


def compute_recall(selected: list[dict], ground_truth: list[tuple[float, float]]) -> float:
    if not ground_truth:
        return 0.0
    
    # 只要 GT 區間內有任何一個 selected 片段符合重疊條件，即算命中
    hit = sum(1 for gt_start, gt_end in ground_truth 
              if any(_is_hit(seg, gt_start, gt_end) for seg in selected))
    
    return hit / len(ground_truth)


def compute_precision(selected: list[dict], ground_truth: list[tuple[float, float]]) -> float:
    if not selected:
        return 0.0
    
    # 只要 selected 片段有打中任何一個 GT 區間，即算命中
    hit = sum(1 for seg in selected 
              if any(_is_hit(seg, gt_start, gt_end) for gt_start, gt_end in ground_truth))
    
    return hit / len(selected)


def compute_f1_score(recall: float, precision: float) -> float:
    if recall + precision <= 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def compute_false_alarm_rate(
    selected: list[dict], 
    ground_truth: list[tuple[float, float]],
    total_duration: float
) -> float:
    if total_duration <= 0:
        return 0.0
        
    false_alarm_sec = 0.0
    for seg in selected:
        # 只要有交集就算涵蓋 (FAR 的判定通常比 hit 寬鬆，不要求 50%)
        is_gt = any(
            min(seg["end"], gt_end) - max(seg["start"], gt_start) > 0
            for gt_start, gt_end in ground_truth
        )
        if not is_gt:
            false_alarm_sec += seg["end"] - seg["start"]
            
    return min(1.0, max(0.0, false_alarm_sec / total_duration))


def compute_time_saving_rate(selected: list[dict], total_duration: float) -> float:
    if total_duration <= 0:
        return 0.0
    summary_duration = sum(seg["end"] - seg["start"] for seg in selected)
    return 1.0 - (summary_duration / total_duration)


def compute_bert_score(summary_texts: list[str], reference_texts: list[str]) -> float | None:
    """計算 BERTScore F1 均值；bert-score 未安裝時回傳 None 而非拋出例外。"""
    if bert_score_fn is None:
        import warnings
        warnings.warn("bert-score 未安裝，BERTScore 評估已跳過。請執行：pip install bert-score")
        return None

    _, _, f1 = bert_score_fn(summary_texts, reference_texts, lang="zh", verbose=False)
    return float(f1.mean())


def evaluate_all(
    selected: list[dict],
    ground_truth: list[tuple[float, float]],
    total_duration: float,
    summary_texts: list[str],
    reference_texts: list[str]
) -> dict:
    results = {
        "recall": compute_recall(selected, ground_truth),
        "precision": compute_precision(selected, ground_truth),
        "false_alarm_rate": compute_false_alarm_rate(selected, ground_truth, total_duration),
        "time_saving_rate": compute_time_saving_rate(selected, total_duration),
    }
    results["f1_score"] = compute_f1_score(results["recall"], results["precision"])
    results["bert_score"] = compute_bert_score(summary_texts, reference_texts)

    print("\n===== 成效評估結果 =====")
    print(f"重點召回率 (Recall)     : {results['recall']:.3f}  (目標 > {TARGET_RECALL})")
    print(f"精確度 (Precision)       : {results['precision']:.3f}")
    print(f"F1-Score                 : {results['f1_score']:.3f}")
    print(f"誤報率 (FAR)             : {results['false_alarm_rate']:.3f}  (目標 < 0.25)")
    bert = results["bert_score"]
    bert_display = f"{bert:.3f}" if bert is not None else "N/A（bert-score 未安裝）"
    print(f"語意相似度 (BERTScore)   : {bert_display}  (目標 > {TARGET_BERT_SCORE})")
    print(f"時間節省率 (TSR)         : {results['time_saving_rate']:.3f}  (目標 > {TARGET_TIME_SAVING_RATE})")

    return results