"""Grid search static late-fusion weights on annotated validation videos."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import grid_search_weights_by_video
from src.fusion.evaluation_data import load_annotated_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid Search on explicitly selected validation videos"
    )
    parser.add_argument(
        "--annotations",
        default="data/annotations",
        help="包含 gt_*.json 的 Ground Truth 資料夾",
    )
    parser.add_argument(
        "--features-dir",
        default="results/features_12d",
        help="包含 features_<video>.json 的多模態特徵資料夾",
    )
    parser.add_argument("--output", default="results/grid_search.json")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--fusion-mode",
        choices=("adaptive", "static"),
        default="adaptive",
    )
    parser.add_argument(
        "--validation-video",
        action="append",
        required=True,
        help="validation 影片 ID（不含 segments_ 前綴；可重複指定）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.annotations):
        parser.error(f"找不到標註資料夾：{args.annotations}")
    if not os.path.isdir(args.features_dir):
        parser.error(f"找不到特徵資料夾：{args.features_dir}")
    if len(set(args.validation_video)) != len(args.validation_video):
        parser.error("--validation-video 不可重複指定")

    videos = []
    for video_id in args.validation_video:
        features_path = os.path.join(
            args.features_dir,
            f"features_{video_id}.json",
        )
        gt_path = os.path.join(args.annotations, f"gt_{video_id}.json")
        if not os.path.isfile(features_path):
            parser.error(f"找不到 validation 特徵檔：{features_path}")
        if not os.path.isfile(gt_path):
            parser.error(f"找不到 validation Ground Truth：{gt_path}")
        videos.append(load_annotated_video(features_path, gt_path))

    print(
        f"載入 {len(videos)} 部影片、"
        f"{sum(len(video['segments']) for video in videos)} 個片段、"
        f"{sum(len(video['ground_truth']) for video in videos)} 個 Ground Truth"
    )
    best_alpha, best_beta, metrics = grid_search_weights_by_video(
        videos,
        step=args.step,
        fusion_mode=args.fusion_mode,
    )
    result = {
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "validation_metrics": metrics,
        "recall_on_validation": round(metrics["recall"], 4),
        "total_videos": len(videos),
        "total_segments": sum(len(video["segments"]) for video in videos),
        "total_gt": sum(len(video["ground_truth"]) for video in videos),
        "video_ids": [video["video_id"] for video in videos],
        "selection_policy": "semantic_sliding_window_per_video",
        "objective": "macro_f1",
        "fusion_mode": args.fusion_mode,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(
        f"\n最佳參數：alpha={best_alpha:.2f}, beta={best_beta:.2f}; "
        f"Recall={metrics['recall']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"F1={metrics['f1_score']:.4f}"
    )
    print(f"結果儲存至：{args.output}")


if __name__ == "__main__":
    main()
