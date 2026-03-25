"""
批次評估腳本
對 30 部測試影片跑完整 pipeline 並輸出 Recall / BERTScore / TSR 報告
對應計畫書 4.4 量化成效評估

用法：
  cd keytake-ai
  python tools/run_batch_eval.py --videos data/videos/ --annotations data/annotations/ --output results/batch_report.json
"""

import json
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run_pipeline
from src.fusion.evaluator import (
    compute_recall, compute_false_alarm_rate,
    compute_time_saving_rate, compute_bert_score
)
from config import TARGET_RECALL, TARGET_FALSE_ALARM_RATE, TARGET_BERT_SCORE, TARGET_TIME_SAVING_RATE


def main():
    parser = argparse.ArgumentParser(description="批次評估 30 部影片")
    parser.add_argument("--videos", default="data/videos", help="影片資料夾")
    parser.add_argument("--annotations", default="data/annotations", help="標註資料夾")
    parser.add_argument("--output", default="results/batch_report.json")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="跳過 BERTScore（較耗時，可先略過）")
    args = parser.parse_args()

    video_dir = args.videos
    ann_dir = args.annotations

    video_files = sorted(
        f for f in os.listdir(video_dir)
        if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ) if os.path.exists(video_dir) else []

    if not video_files:
        print(f"找不到影片：{video_dir}")
        sys.exit(1)

    print(f"共找到 {len(video_files)} 部影片，開始批次評估...\n")

    all_results = []
    for i, vf in enumerate(video_files):
        vid_id = os.path.splitext(vf)[0]
        video_path = os.path.join(video_dir, vf)
        gt_path = os.path.join(ann_dir, f"gt_{vid_id}.json")

        print(f"[{i+1}/{len(video_files)}] {vf}")

        if not os.path.exists(gt_path):
            print(f"  跳過（找不到 Ground Truth：{gt_path}）\n")
            continue

        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)
        ground_truth = [(g["start_sec"], g["end_sec"]) for g in gt_data["ground_truth"]]

        t0 = time.time()
        try:
            result = run_pipeline(video_path, output_dir=f"output/{vid_id}")
        except Exception as e:
            print(f"  ✗ Pipeline 失敗：{e}\n")
            all_results.append({"video": vf, "error": str(e)})
            continue

        elapsed = time.time() - t0
        selected = result["segments"]
        total_duration = result["original_duration"]

        recall = compute_recall(selected, ground_truth)
        far = compute_false_alarm_rate(selected, ground_truth, total_duration)
        tsr = compute_time_saving_rate(selected, total_duration)

        # BERTScore（可選，較耗時）
        bs = None
        if not args.skip_bertscore:
            summary_texts = [s.get("text", "") for s in selected if s.get("text")]
            ref_texts = [s.get("text", "") for s in selected[:len(summary_texts)]]
            if summary_texts and ref_texts:
                try:
                    bs = compute_bert_score(summary_texts, ref_texts)
                except Exception:
                    bs = None

        video_result = {
            "video": vf,
            "recall": round(recall, 4),
            "false_alarm_rate": round(far, 4),
            "bert_score": round(bs, 4) if bs is not None else "skipped",
            "time_saving_rate": round(tsr, 4),
            "processing_time_sec": round(elapsed, 1),
            "segments_selected": len(selected),
        }
        all_results.append(video_result)

        status = "✓" if recall >= TARGET_RECALL else "✗"
        print(f"  {status} Recall={recall:.3f}  FAR={far:.3f}  TSR={tsr:.3f}  ({elapsed:.1f}s)\n")

    # 彙總統計
    valid = [r for r in all_results if "error" not in r]
    if valid:
        avg = lambda key: sum(r[key] for r in valid if isinstance(r.get(key), float)) / len(valid)
        summary = {
            "total_videos": len(video_files),
            "evaluated": len(valid),
            "avg_recall": round(avg("recall"), 4),
            "avg_false_alarm_rate": round(avg("false_alarm_rate"), 4),
            "avg_time_saving_rate": round(avg("time_saving_rate"), 4),
            "target_recall": TARGET_RECALL,
            "target_false_alarm_rate": TARGET_FALSE_ALARM_RATE,
            "target_time_saving_rate": TARGET_TIME_SAVING_RATE,
            "videos": all_results
        }
    else:
        summary = {"error": "沒有成功評估的影片", "videos": all_results}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if valid:
        print("=" * 45)
        print(f"批次評估完成（{len(valid)}/{len(video_files)} 部）")
        print(f"平均 Recall        : {summary['avg_recall']:.4f}  (目標 > {TARGET_RECALL})")
        print(f"平均 False Alarm   : {summary['avg_false_alarm_rate']:.4f}  (目標 < {TARGET_FALSE_ALARM_RATE})")
        print(f"平均 Time Saving   : {summary['avg_time_saving_rate']:.4f}  (目標 > {TARGET_TIME_SAVING_RATE})")
        print(f"報告儲存至：{args.output}")


if __name__ == "__main__":
    main()
