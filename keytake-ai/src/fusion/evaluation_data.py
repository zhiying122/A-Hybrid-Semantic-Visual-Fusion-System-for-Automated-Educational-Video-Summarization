"""Load and normalize per-video features and annotations for evaluation."""

import json
import math
import os


def normalize_evaluation_data(
    segments: list[dict],
    ground_truth: list[tuple[float, float]],
    eval_range: list[float] | tuple[float, float] | None = None,
) -> tuple[list[dict], list[tuple[float, float]], float]:
    validated_segments = []
    for index, segment in enumerate(segments):
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"特徵片段時間範圍無效：第 {index + 1} 段"
            ) from error
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(
                f"特徵片段時間範圍無效：第 {index + 1} 段"
            )
        validated_segments.append({**segment, "start": start, "end": end})

    for index, interval in enumerate(ground_truth):
        if len(interval) != 2:
            raise ValueError(f"Ground Truth 區間格式錯誤：第 {index + 1} 段")
        start, end = map(float, interval)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"Ground Truth 時間範圍無效：第 {index + 1} 段")

    if eval_range is not None:
        if not isinstance(eval_range, (list, tuple)) or len(eval_range) < 2:
            raise ValueError("eval_range_sec 必須包含起訖秒數")
        range_start, range_end = float(eval_range[0]), float(eval_range[1])
        if (
            not math.isfinite(range_start)
            or not math.isfinite(range_end)
            or range_start < 0
            or range_end <= range_start
        ):
            raise ValueError("eval_range_sec 時間範圍無效")
    else:
        range_start = 0.0
        range_end = max(
            max((segment["end"] for segment in validated_segments), default=0.0),
            max((end for _, end in ground_truth), default=0.0),
        )

    clipped_segments = []
    for segment in validated_segments:
        start, end = segment["start"], segment["end"]
        clipped_start = max(start, range_start)
        clipped_end = min(end, range_end)
        if clipped_end <= clipped_start:
            continue
        clipped_segments.append(
            {
                **segment,
                "start": clipped_start - range_start,
                "end": clipped_end - range_start,
            }
        )

    clipped_ground_truth = []
    for start, end in ground_truth:
        clipped_start = max(start, range_start)
        clipped_end = min(end, range_end)
        if clipped_end > clipped_start:
            clipped_ground_truth.append(
                (clipped_start - range_start, clipped_end - range_start)
            )

    return clipped_segments, clipped_ground_truth, range_end - range_start


def load_annotated_video(feature_path: str, gt_path: str) -> dict:
    with open(feature_path, encoding="utf-8") as file:
        segments = json.load(file)
    with open(gt_path, encoding="utf-8") as file:
        gt_data = json.load(file)

    if not isinstance(segments, list):
        raise ValueError(f"特徵檔必須是片段清單：{feature_path}")

    raw_ground_truth = (
        gt_data.get("ground_truth", [])
        if isinstance(gt_data, dict)
        else gt_data
    )
    if not isinstance(raw_ground_truth, list):
        raise ValueError(f"Ground Truth 必須是區間清單：{gt_path}")

    ground_truth = []
    for item in raw_ground_truth:
        if isinstance(item, dict):
            start = item.get("start_sec", item.get("start"))
            end = item.get("end_sec", item.get("end"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[:2]
        else:
            raise ValueError(f"Ground Truth 格式錯誤：{gt_path}")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"Ground Truth 時間範圍無效：{gt_path}")
        ground_truth.append((start, end))

    metadata = gt_data if isinstance(gt_data, dict) else {}
    normalized_segments, normalized_gt, duration = normalize_evaluation_data(
        segments,
        ground_truth,
        metadata.get("eval_range_sec"),
    )
    video_id = os.path.splitext(os.path.basename(feature_path))[0].replace(
        "features_", "", 1
    )
    return {
        "video_id": video_id,
        "segments": normalized_segments,
        "ground_truth": normalized_gt,
        "duration": duration,
    }
