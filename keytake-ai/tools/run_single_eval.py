"""
單影片評估腳本（支援評估範圍限定）
─────────────────────────────────────────────────────────────────────
用於在只標註了影片「部分範圍」的情況下，做公平的量化評估。

與 run_batch_eval.py 的差異：
  - 針對單一影片 + 單一 Ground Truth 檔
  - 支援 --range-end：只評估影片前 N 秒（與部分標註對齊）
    只計算落在評估範圍內的選中片段，避免範圍外的選段被誤算為誤報。
  - 輸出 Recall / Precision / F1 / FAR / TSR 與完整明細，供論文引用

用法：
  cd keytake-ai
  python tools/run_single_eval.py \
      --video data/videos/test_10min.mp4 \
      --gt data/annotations/gt_test_10min.json \
      --output results/eval_test_10min_real.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run_pipeline
from src.fusion.evaluator import (
    compute_recall, compute_precision, compute_f1_score,
    compute_false_alarm_rate, compute_time_saving_rate,
)


def _load_gt(gt_path: str):
    """讀取 GT，支援 start_sec/end_sec 或 start/end 兩種欄位。"""
    with open(gt_path, encoding="utf-8") as f:
        raw = json.load(f)
    gt_list = raw.get("ground_truth", raw if isinstance(raw, list) else [])
    gt = []
    for g in gt_list:
        if isinstance(g, dict):
            s = g.get("start_sec", g.get("start"))
            e = g.get("end_sec", g.get("end"))
            gt.append((float(s), float(e)))
        else:
            gt.append((float(g[0]), float(g[1])))
    meta = raw if isinstance(raw, dict) else {}
    return gt, meta


def _clip_segments_to_range(segments, range_end):
    """只保留（並裁切）落在 [0, range_end] 內的選中片段。"""
    clipped = []
    for seg in segments:
        if seg["start"] >= range_end:
            continue
        clipped.append({**seg, "end": min(seg["end"], range_end)})
    return clipped


def main():
    parser = argparse.ArgumentParser(description="單影片評估（支援範圍限定）")
    parser.add_argument("--video", required=True, help="影片路徑")
    parser.add_argument("--gt", required=True, help="Ground Truth JSON 路徑")
    parser.add_argument("--output", default="results/single_eval.json")
    parser.add_argument("--range-end", type=float, default=None,
                        help="評估範圍結束秒數（預設讀 GT 的 eval_range_sec，或用全片）")
    args = parser.parse_args()

    ground_truth, meta = _load_gt(args.gt)

    # 決定評估範圍
    range_end = args.range_end
    if range_end is None and "eval_range_sec" in meta:
        range_end = float(meta["eval_range_sec"][1])

    print(f"影片：{args.video}")
    print(f"GT 區間數：{len(ground_truth)}")
    print(f"評估範圍：0 ~ {range_end if range_end else '全片'}s\n")

    t0 = time.time()
    result = run_pipeline(args.video, output_dir="output/eval_run")
    elapsed = time.time() - t0

    selected_full = result["segments"]
    original_duration = result["original_duration"]

    # 範圍對齊
    if range_end is not None:
        selected = _clip_segments_to_range(selected_full, range_end)
        eval_duration = range_end
    else:
        selected = selected_full
        eval_duration = original_duration

    recall = compute_recall(selected, ground_truth)
    precision = compute_precision(selected, ground_truth)
    f1 = compute_f1_score(recall, precision)
    far = compute_false_alarm_rate(selected, ground_truth, eval_duration)
    summary_dur = sum(s["end"] - s["start"] for s in selected)
    tsr = 1.0 - summary_dur / eval_duration if eval_duration > 0 else 0.0

    report = {
        "generated_at": datetime.now().isoformat(),
        "video": os.path.basename(args.video),
        "data_type": "REAL (真實影片 + 人工標註)",
        "annotator_count": meta.get("annotator_count", "unknown"),
        "cohens_kappa": meta.get("cohens_kappa"),
        "note": meta.get("note", ""),
        "eval_range_sec": [0.0, eval_duration],
        "original_duration_sec": round(original_duration, 2),
        "metrics": {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "false_alarm_rate": round(far, 4),
            "time_saving_rate": round(tsr, 4),
        },
        "ground_truth_count": len(ground_truth),
        "selected_in_range": len(selected),
        "selected_total": len(selected_full),
        "summary_duration_in_range_sec": round(summary_dur, 2),
        "processing_time_sec": round(elapsed, 1),
        "selected_segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2)}
            for s in selected
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("  真實影片評估結果（單人標註初步版）")
    print("=" * 50)
    print(f"  Recall (重點召回率) : {recall:.1%}")
    print(f"  Precision (精確率)  : {precision:.1%}")
    print(f"  F1-Score            : {f1:.3f}")
    print(f"  FAR (誤報率)        : {far:.1%}  (目標 < 25%)")
    print(f"  TSR (時間節省率)    : {tsr:.1%}  (目標 > 50%)")
    print(f"  範圍內選中片段      : {len(selected)} 個")
    print(f"  報告已存至          : {args.output}")


if __name__ == "__main__":
    main()
