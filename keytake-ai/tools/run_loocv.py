"""Leave-one-video-out evaluation for fusion-weight selection."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import leave_one_out_cv
from src.fusion.evaluation_data import load_annotated_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Video-Out fusion-weight evaluation"
    )
    parser.add_argument("--annotations", default="data/annotations")
    parser.add_argument("--features-dir", default="results/features_12d")
    parser.add_argument("--output", default="results/loocv.json")
    parser.add_argument(
        "--fusion-mode",
        choices=("adaptive", "static"),
        default="adaptive",
    )
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="LOOCV 影片 ID（不含 features_ 前綴；可重複指定）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.annotations):
        parser.error(f"找不到標註資料夾：{args.annotations}")
    if not os.path.isdir(args.features_dir):
        parser.error(f"找不到特徵資料夾：{args.features_dir}")
    if len(set(args.video)) != len(args.video):
        parser.error("--video 不可重複指定")

    videos = []
    for video_id in args.video:
        feature_path = os.path.join(
            args.features_dir,
            f"features_{video_id}.json",
        )
        gt_path = os.path.join(args.annotations, f"gt_{video_id}.json")
        if not os.path.isfile(feature_path):
            parser.error(f"找不到 LOOCV 特徵檔：{feature_path}")
        if not os.path.isfile(gt_path):
            parser.error(f"找不到 LOOCV Ground Truth：{gt_path}")
        videos.append(load_annotated_video(feature_path, gt_path))

    if len(videos) < 2:
        parser.error("LOOCV 至少需要 2 部特徵與標註配對影片")

    print(f"開始 LOOCV，共 {len(videos)} 折...")
    results = leave_one_out_cv(videos, fusion_mode=args.fusion_mode)
    metric_names = ("recall", "precision", "f1_score")
    averages = {
        metric: sum(result[metric] for result in results) / len(results)
        for metric in metric_names
    }
    summary = {
        "fusion_mode": args.fusion_mode,
        "video_ids": [video["video_id"] for video in videos],
        "folds": results,
        "macro_metrics": {
            metric: round(value, 4) for metric, value in averages.items()
        },
        "avg_recall": round(averages["recall"], 4),
        "min_recall": round(min(result["recall"] for result in results), 4),
        "max_recall": round(max(result["recall"] for result in results), 4),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(
        f"平均 Recall={averages['recall']:.4f}, "
        f"Precision={averages['precision']:.4f}, "
        f"F1={averages['f1_score']:.4f}"
    )
    print(f"結果儲存至：{args.output}")


if __name__ == "__main__":
    main()
