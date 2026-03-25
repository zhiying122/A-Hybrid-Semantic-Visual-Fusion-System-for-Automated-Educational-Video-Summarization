"""
Leave-One-Out Cross-Validation 執行腳本
對應計畫書 4.2 步驟四：驗證模型在未見過課程上的泛化穩定性

用法：
  cd keytake-ai
  python tools/run_loocv.py --annotations data/annotations/ --output results/loocv.json
"""

import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import leave_one_out_cv


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out 交叉驗證")
    parser.add_argument("--annotations", default="data/annotations")
    parser.add_argument("--output", default="results/loocv.json")
    args = parser.parse_args()

    ann_dir = args.annotations
    if not os.path.exists(ann_dir):
        print(f"找不到標註資料夾：{ann_dir}")
        sys.exit(1)

    seg_files = sorted(f for f in os.listdir(ann_dir) if f.startswith("segments_"))
    all_videos = []
    for sf in seg_files:
        vid_id = sf.replace("segments_", "").replace(".json", "")
        gt_path = os.path.join(ann_dir, f"gt_{vid_id}.json")
        if not os.path.exists(gt_path):
            continue
        with open(os.path.join(ann_dir, sf), encoding="utf-8") as f:
            segments = json.load(f)
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)
        all_videos.append({
            "video_id": vid_id,
            "segments": segments,
            "ground_truth": [(g["start_sec"], g["end_sec"]) for g in gt_data["ground_truth"]]
        })

    if len(all_videos) < 2:
        print(f"LOOCV 至少需要 2 部影片，目前只有 {len(all_videos)} 部")
        sys.exit(1)

    print(f"開始 LOOCV，共 {len(all_videos)} 折...")
    results = leave_one_out_cv(all_videos)

    recalls = [r["recall"] for r in results]
    avg_recall = sum(recalls) / len(recalls)

    summary = {
        "folds": results,
        "avg_recall": round(avg_recall, 4),
        "min_recall": round(min(recalls), 4),
        "max_recall": round(max(recalls), 4),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLOOCV 完成")
    print(f"平均 Recall：{avg_recall:.4f}（min={min(recalls):.4f}, max={max(recalls):.4f}）")
    print(f"結果儲存至：{args.output}")


if __name__ == "__main__":
    main()
