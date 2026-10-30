"""Train the 12-feature MLP fusion model from extracted per-video features."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MLP_FUSION_MODEL_PATH
from src.fusion.mlp_fusion import (
    MLP_FEATURE_VERSION,
    MLPFusionTrainer,
    TEACHING_STAGES,
)
from src.fusion.adaptive_fusion import semantic_sliding_window
from src.fusion.evaluator import (
    compute_f1_score,
    compute_precision,
    compute_recall,
)
from src.fusion.evaluation_data import normalize_evaluation_data

REQUIRED_FEATURES = (
    "s_text",
    "s_visual",
    "visual_reliability",
    "gesture_intent_score",
    "s_prosodic",
    "prosodic_available",
    "teaching_stage",
    "feature_version",
)
NUMERIC_FEATURES = (
    "s_text",
    "s_visual",
    "visual_reliability",
    "gesture_intent_score",
    "s_prosodic",
)


def _load_ground_truth_data(
    path: Path,
) -> tuple[list[tuple[float, float]], list[float] | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("ground_truth", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError(f"Ground Truth 不可為空或格式錯誤：{path}")

    intervals = []
    for item in items:
        if isinstance(item, dict):
            start = item.get("start_sec", item.get("start"))
            end = item.get("end_sec", item.get("end"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[:2]
        else:
            raise ValueError(f"Ground Truth 區間格式錯誤：{path}")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Ground Truth 時間範圍無效：{path}")
        intervals.append((start, end))
    metadata = raw if isinstance(raw, dict) else {}
    return intervals, metadata.get("eval_range_sec")


def _load_feature_file(path: Path) -> list[dict]:
    segments = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"特徵檔必須是非空片段陣列：{path}")

    for index, segment in enumerate(segments):
        missing = [key for key in REQUIRED_FEATURES if key not in segment]
        if missing:
            raise ValueError(
                f"{path.name} 第 {index + 1} 段缺少 MLP 特徵：{', '.join(missing)}；"
                "請先使用 tools/extract_features.py 重新抽取"
            )
        if segment["feature_version"] != MLP_FEATURE_VERSION:
            raise ValueError(
                f"{path.name} 第 {index + 1} 段特徵版本不相容："
                f"{segment['feature_version']!r}；預期 {MLP_FEATURE_VERSION!r}，"
                "請重新抽取特徵"
            )
        for key in NUMERIC_FEATURES:
            try:
                value = float(segment[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name} 第 {index + 1} 段的 {key} 非數字") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{path.name} 第 {index + 1} 段的 {key} 必須介於 0 與 1"
                )
        if not isinstance(segment["teaching_stage"], str):
            raise ValueError(
                f"{path.name} 第 {index + 1} 段的 teaching_stage 必須是字串"
            )
        if not isinstance(segment["prosodic_available"], bool):
            raise ValueError(
                f"{path.name} 第 {index + 1} 段的 prosodic_available 必須是布林值"
            )
        if segment["teaching_stage"] not in (*TEACHING_STAGES, "unclassified"):
            raise ValueError(
                f"{path.name} 第 {index + 1} 段的 teaching_stage 不在支援類別中："
                f"{segment['teaching_stage']}"
            )
        start, end = float(segment.get("start", -1)), float(segment.get("end", -1))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"{path.name} 第 {index + 1} 段時間範圍無效")
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="訓練 12 維 MLP 融合模型")
    parser.add_argument(
        "--features",
        default="results",
        help="包含 features_<video>.json 的資料夾",
    )
    parser.add_argument(
        "--annotations",
        default="data/annotations",
        help="包含 gt_<video>.json 的 Ground Truth 資料夾",
    )
    parser.add_argument("--output", default=MLP_FUSION_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--test-features",
        action="append",
        required=True,
        help="獨立測試影片特徵 JSON；可重複指定多部影片",
    )
    parser.add_argument(
        "--test-annotations",
        action="append",
        required=True,
        help="與每個 --test-features 對應的 Ground Truth JSON",
    )
    parser.add_argument(
        "--label-overlap-basis",
        choices=("ground_truth", "segment"),
        default="ground_truth",
        help="正樣本重疊比例分母；預設與 evaluator 一致",
    )
    args = parser.parse_args()

    feature_dir = Path(args.features)
    annotation_dir = Path(args.annotations)
    feature_files = sorted(feature_dir.glob("features_*.json"))
    if not feature_files:
        parser.error(f"找不到特徵檔：{feature_dir}\\features_*.json")
    if len(args.test_features) != len(args.test_annotations):
        parser.error("--test-features 與 --test-annotations 必須一一對應")

    test_pairs = [
        (Path(feature_path), Path(annotation_path))
        for feature_path, annotation_path in zip(
            args.test_features,
            args.test_annotations,
        )
    ]
    test_video_ids = {
        feature_path.stem.removeprefix("features_")
        for feature_path, _ in test_pairs
    }
    if len(test_video_ids) != len(test_pairs):
        parser.error("獨立測試影片不可重複指定")
    test_sets = []
    for feature_path, annotation_path in test_pairs:
        if not feature_path.is_file():
            parser.error(f"找不到獨立測試特徵檔：{feature_path}")
        if not annotation_path.is_file():
            parser.error(f"找不到獨立測試 Ground Truth：{annotation_path}")
        try:
            test_segments = _load_feature_file(feature_path)
            test_ground_truth, eval_range = _load_ground_truth_data(
                annotation_path
            )
            # 獨立測試資料也必須裁至標註範圍，不能把未標註的尾段計入選段預算。
            test_segments, test_ground_truth, test_duration = (
                normalize_evaluation_data(
                    test_segments,
                    test_ground_truth,
                    eval_range,
                )
            )
            test_sets.append(
                (
                    feature_path,
                    test_segments,
                    test_ground_truth,
                    test_duration,
                )
            )
        except ValueError as exc:
            parser.error(str(exc))

    all_segments: list[dict] = []
    all_ground_truth: list[tuple[float, float]] = []
    all_groups: list[str] = []
    videos_loaded = 0
    time_offset = 0.0

    for feature_path in feature_files:
        video_id = feature_path.stem.removeprefix("features_")
        if video_id in test_video_ids:
            print(f"[隔離測試集] {feature_path.name}")
            continue
        gt_path = annotation_dir / f"gt_{video_id}.json"
        if not gt_path.is_file():
            print(f"[略過] 找不到配對的 Ground Truth：{gt_path}")
            continue

        try:
            segments = _load_feature_file(feature_path)
            ground_truth, eval_range = _load_ground_truth_data(gt_path)
            # 每部訓練影片先各自裁切到 eval_range，再平移到不重疊的合併時間軸。
            segments, ground_truth, duration = normalize_evaluation_data(
                segments,
                ground_truth,
                eval_range,
            )
        except ValueError as exc:
            parser.error(str(exc))
        shifted_segments = []
        for segment in segments:
            shifted = dict(segment)
            shifted["start"] = float(segment["start"]) + time_offset
            shifted["end"] = float(segment["end"]) + time_offset
            shifted_segments.append(shifted)
        all_segments.extend(shifted_segments)
        all_groups.extend([video_id] * len(shifted_segments))
        all_ground_truth.extend(
            (start + time_offset, end + time_offset)
            for start, end in ground_truth
        )

        time_offset += duration + 1.0
        videos_loaded += 1

    if not all_segments:
        parser.error("沒有找到同時具備完整特徵與配對 Ground Truth 的影片")
    if videos_loaded < 2:
        parser.error("至少需要兩部訓練影片，才能進行影片分組的 train/validation 切分")

    print(
        f"載入 {videos_loaded} 部影片、{len(all_segments)} 個片段、"
        f"{len(all_ground_truth)} 個 Ground Truth 區間"
    )
    print(f"訓練標籤規則：overlap / {args.label_overlap_basis} duration >= 0.5")
    trainer = MLPFusionTrainer(
        model_path=args.output,
        label_overlap_basis=args.label_overlap_basis,
    )
    stats = trainer.train(
        all_segments,
        all_ground_truth,
        epochs=args.epochs,
        lr=args.lr,
        groups=all_groups,
    )

    print(
        f"訓練完成：best_epoch={stats['best_epoch']}, "
        f"val_loss={stats['best_val_loss']:.4f}, "
        f"positive_ratio={stats['positive_ratio']:.2%}"
    )
    print(f"模型儲存至：{args.output}")
    print("獨立測試集（未參與訓練或 validation 選擇）：")
    for (
        feature_path,
        test_segments,
        test_ground_truth,
        test_duration,
    ) in test_sets:
        test_scores = trainer.predict(test_segments)
        selected = semantic_sliding_window(
            test_segments,
            test_scores,
            original_duration=test_duration,
        )
        recall = compute_recall(selected, test_ground_truth)
        precision = compute_precision(selected, test_ground_truth)
        f1 = compute_f1_score(recall, precision)
        print(
            f"  {feature_path.stem}: Recall={recall:.3f}, "
            f"Precision={precision:.3f}, F1={f1:.3f}"
        )

    print(
        "MLP 預設仍維持停用；需先檢視獨立測試結果，再於 config.py 明確啟用。"
    )


if __name__ == "__main__":
    main()
