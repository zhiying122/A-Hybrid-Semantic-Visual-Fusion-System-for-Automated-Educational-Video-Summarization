"""Multimodal late fusion and budget-aware semantic segment selection."""

import numpy as np

from config import (
    ALPHA,
    BETA,
    FUSION_SCORE_THRESHOLD,
    MAX_SUMMARY_RATIO,
    MIN_SUMMARY_SEC,
    SLIDING_WINDOW_MIN_SEC,
)
from src.fusion.evaluator import compute_f1_score, compute_precision, compute_recall


def fuse_scores(
    s_text: float,
    s_visual: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    visual_reliability: float = 0.0,
) -> float:
    """Fuse text and visual scores using reliability-weighted late fusion.

    beta_eff = beta * visual_reliability
    alpha_eff = 1 - beta_eff
    """
    values = (s_text, s_visual, alpha, beta, visual_reliability)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("融合分數、權重與可靠度必須是有限數值")
    if not 0.0 <= s_text <= 1.0 or not 0.0 <= s_visual <= 1.0:
        raise ValueError("文字與視覺分數必須介於 0 與 1")
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError("融合權重必須介於 0 與 1")
    if not np.isclose(alpha + beta, 1.0, atol=1e-6):
        raise ValueError("融合權重必須符合 alpha + beta = 1")

    reliability = float(np.clip(visual_reliability, 0.0, 1.0))
    # 可靠度只折扣視覺權重；語意權重補足至 1，維持兩模態權重總和為 1。
    beta_eff = beta * reliability
    alpha_eff = 1.0 - beta_eff
    return float(alpha_eff * s_text + beta_eff * s_visual)


def _video_duration(video: dict) -> float | None:
    duration = video.get("duration")
    if duration is not None:
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("影片 duration 必須是有限的正數")
        return duration

    segments = video.get("segments", [])
    ground_truth = video.get("ground_truth", [])
    endpoints = [float(segment["end"]) for segment in segments]
    endpoints.extend(float(end) for _, end in ground_truth)
    return max(endpoints, default=None)


def _evaluate_video(
    segments: list[dict],
    scores: list[float],
    ground_truth: list[tuple[float, float]],
    duration: float | None = None,
) -> tuple[dict, list[dict]]:
    selected = semantic_sliding_window(
        segments,
        scores,
        original_duration=duration,
    )
    recall = compute_recall(selected, ground_truth)
    precision = compute_precision(selected, ground_truth)
    return {
        "recall": recall,
        "precision": precision,
        "f1_score": compute_f1_score(recall, precision),
    }, selected


def _mean_metrics(metrics: list[dict]) -> dict[str, float]:
    if not metrics:
        return {"recall": 0.0, "precision": 0.0, "f1_score": 0.0}
    return {
        name: float(np.mean([item[name] for item in metrics]))
        for name in ("recall", "precision", "f1_score")
    }


def _search_weights(
    videos: list[dict],
    step: float,
    fusion_mode: str = "adaptive",
) -> tuple[float, float, dict[str, float]]:
    if not np.isfinite(step) or not 0.0 < step <= 1.0:
        raise ValueError("Grid Search step 必須介於 0 與 1 之間")
    if fusion_mode not in {"adaptive", "static"}:
        raise ValueError("fusion_mode 必須是 adaptive 或 static")
    if not videos or not any(video.get("ground_truth") for video in videos):
        return ALPHA, BETA, {"recall": 0.0, "precision": 0.0, "f1_score": 0.0}

    # Include the 1.0 endpoint even when the requested step does not divide it.
    alphas = sorted(set(float(value) for value in np.arange(0.0, 1.0, step)) | {1.0})
    best_alpha, best_beta = ALPHA, BETA
    best_metrics = {"recall": 0.0, "precision": 0.0, "f1_score": -1.0}

    for alpha in alphas:
        beta = 1.0 - alpha
        fold_metrics = []
        for video in videos:
            segments = video.get("segments", [])
            ground_truth = video.get("ground_truth", [])
            if not segments or not ground_truth:
                continue
            required = {"s_text", "s_visual"}
            if fusion_mode == "adaptive":
                required.add("visual_reliability")
            for index, segment in enumerate(segments):
                missing = required.difference(segment)
                if missing:
                    raise ValueError(
                        f"影片 {video.get('video_id', '<unknown>')} 第 {index + 1} "
                        f"段缺少 Grid Search 特徵：{', '.join(sorted(missing))}"
                    )
            scores = [
                fuse_scores(
                    segment.get("s_text", 0.0),
                    segment.get("s_visual", 0.0),
                    alpha,
                    beta,
                    visual_reliability=(
                        1.0
                        if fusion_mode == "static"
                        else segment.get("visual_reliability", 0.0)
                    ),
                )
                for segment in segments
            ]
            metrics, _ = _evaluate_video(
                segments,
                scores,
                ground_truth,
                duration=_video_duration(video),
            )
            fold_metrics.append(metrics)

        metrics = _mean_metrics(fold_metrics)
        candidate_key = (
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
        )
        best_key = (
            best_metrics["f1_score"],
            best_metrics["precision"],
            best_metrics["recall"],
        )
        if candidate_key > best_key:
            best_alpha, best_beta, best_metrics = alpha, beta, metrics

    return best_alpha, best_beta, best_metrics


def grid_search_weights(
    segments: list[dict],
    ground_truth: list[tuple[float, float]],
    step: float = 0.1,
    original_duration: float | None = None,
    fusion_mode: str = "adaptive",
) -> tuple[float, float]:
    """Search alpha/beta on one validation video using macro F1 as the objective.

    The historical two-value return shape is retained for existing callers.
    """
    alpha, beta, metrics = _search_weights(
        [
            {
                "segments": segments,
                "ground_truth": ground_truth,
                "duration": original_duration,
            }
        ],
        step,
        fusion_mode,
    )
    print(
        f"[GridSearch:{fusion_mode}] Best alpha={alpha:.2f}, beta={beta:.2f}, "
        f"Recall={metrics['recall']:.3f}, Precision={metrics['precision']:.3f}, "
        f"F1={metrics['f1_score']:.3f}"
    )
    return alpha, beta


def grid_search_weights_by_video(
    videos: list[dict],
    step: float = 0.1,
    fusion_mode: str = "adaptive",
) -> tuple[float, float, dict[str, float]]:
    """Search weights while evaluating each video's timeline independently."""
    alpha, beta, metrics = _search_weights(videos, step, fusion_mode)
    print(
        f"[GridSearch:{fusion_mode}] Best alpha={alpha:.2f}, beta={beta:.2f}, "
        f"Recall={metrics['recall']:.3f}, Precision={metrics['precision']:.3f}, "
        f"F1={metrics['f1_score']:.3f}"
    )
    return alpha, beta, metrics


def _compute_recall(
    selected: list[dict], ground_truth: list[tuple[float, float]]
) -> float:
    """Compute GT coverage recall (overlap / GT duration >= 50%)."""
    return compute_recall(selected, ground_truth)


def leave_one_out_cv(
    all_videos: list[dict],
    fusion_mode: str = "adaptive",
) -> list[dict]:
    """Evaluate fusion weights with each video held out in turn."""
    if len(all_videos) < 2:
        raise ValueError("LOOCV 至少需要兩部影片")

    results = []
    for fold, test_video in enumerate(all_videos):
        train_videos = [
            video for index, video in enumerate(all_videos) if index != fold
        ]
        alpha, beta, _ = _search_weights(
            train_videos,
            step=0.1,
            fusion_mode=fusion_mode,
        )
        segments = test_video["segments"]
        ground_truth = test_video["ground_truth"]
        scores = [
            fuse_scores(
                segment.get("s_text", 0.0),
                segment.get("s_visual", 0.0),
                alpha,
                beta,
                visual_reliability=(
                    1.0
                    if fusion_mode == "static"
                    else segment.get("visual_reliability", 0.0)
                ),
            )
            for segment in segments
        ]
        metrics, selected = _evaluate_video(
            segments,
            scores,
            ground_truth,
            duration=_video_duration(test_video),
        )
        results.append(
            {
                "fold": fold,
                "alpha": alpha,
                "beta": beta,
                **metrics,
                "selected_count": len(selected),
            }
        )
        print(
            f"[LOOCV] Fold {fold}: alpha={alpha:.2f}, beta={beta:.2f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"Precision={metrics['precision']:.3f}, F1={metrics['f1_score']:.3f}"
        )

    return results


def _validate_segments_and_scores(
    segments: list[dict], scores: list[float]
) -> list[tuple[dict, float]]:
    if len(segments) != len(scores):
        raise ValueError("segments 與 scores 數量必須一致")
    paired = []
    for index, (segment, score) in enumerate(zip(segments, scores)):
        if not isinstance(segment, dict):
            raise ValueError(f"第 {index + 1} 個片段必須是字典")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
            score = float(score)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"第 {index + 1} 個片段時間或分數格式無效") from exc
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"第 {index + 1} 個片段時間範圍無效")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"第 {index + 1} 個片段分數必須介於 0 與 1")
        paired.append(({**segment, "start": start, "end": end}, score))
    return sorted(paired, key=lambda item: (item[0]["start"], item[0]["end"]))


def _make_output_segment(
    source: dict,
    score: float,
    start: float,
    end: float,
) -> dict:
    return {
        "start": start,
        "end": end,
        "s_text": source.get("s_text", 0.0),
        "s_visual": source.get("s_visual", 0.0),
        "text": source.get("text", ""),
        "score": score,
        "teaching_stage": source.get("teaching_stage", "unclassified"),
        "stage_label": source.get("stage_label", ""),
        "segment_summary": source.get("segment_summary", ""),
    }


def _merge_overlapping_selected(selected: list[dict]) -> list[dict]:
    merged = []
    for segment in selected:
        if merged and segment["start"] < merged[-1]["end"]:
            previous = merged[-1]
            previous["end"] = max(previous["end"], segment["end"])
            if segment["score"] > previous["score"]:
                for key in (
                    "s_text",
                    "s_visual",
                    "text",
                    "score",
                    "teaching_stage",
                    "stage_label",
                    "segment_summary",
                ):
                    previous[key] = segment[key]
        else:
            merged.append(dict(segment))
    return merged


def _split_long_selected(selected: list[dict], max_segment_sec: float) -> list[dict]:
    split = []
    for segment in selected:
        start = segment["start"]
        while start < segment["end"]:
            end = min(segment["end"], start + max_segment_sec)
            split.append({**segment, "start": start, "end": end})
            start = end
    return split


def semantic_sliding_window(
    segments: list[dict],
    scores: list[float],
    min_sec: float = SLIDING_WINDOW_MIN_SEC,
    max_summary_ratio: float = MAX_SUMMARY_RATIO,
    min_summary_sec: float = MIN_SUMMARY_SEC,
    original_duration: float | None = None,
    budget_strategy: str = "score",
) -> list[dict]:
    """Select chronological semantic segments without exceeding the duration budget.

    ``budget_strategy="score"`` retains the historical priority; ``"score_per_second"``
    is available for paired evaluation of shorter-segment prioritization.
    """
    if not segments and not scores:
        return []
    if budget_strategy not in {"score", "score_per_second"}:
        raise ValueError("budget_strategy 必須是 score 或 score_per_second")
    if not np.isfinite(min_sec) or min_sec < 0:
        raise ValueError("min_sec 必須是有限的非負數")
    if not np.isfinite(min_summary_sec) or min_summary_sec < 0:
        raise ValueError("min_summary_sec 必須是有限的非負數")
    if not np.isfinite(max_summary_ratio) or not 0.0 <= max_summary_ratio <= 1.0:
        raise ValueError("max_summary_ratio 必須介於 0 與 1")

    paired = _validate_segments_and_scores(segments, scores)
    if not paired or max_summary_ratio == 0.0:
        return []

    max_segment_end = max(segment["end"] for segment, _ in paired)
    if original_duration is None:
        duration = max_segment_end
    else:
        duration = float(original_duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("original_duration 必須是有限的正數")
        if max_segment_end > duration + 1e-6:
            raise ValueError("片段結束時間超出 original_duration")

    budget = duration * max_summary_ratio
    if budget <= 0:
        return []

    ordered_segments = [item[0] for item in paired]
    ordered_scores = [item[1] for item in paired]
    effective_min_sec = min(min_sec, budget)
    max_segment_sec = min(
        max(effective_min_sec * 2.0, min_summary_sec),
        budget,
    )
    if max_segment_sec <= 0:
        max_segment_sec = budget

    select_threshold = FUSION_SCORE_THRESHOLD
    # 足夠多片段時使用固定底線與分數中位數兩者較高者，抑制低分段過度入選。
    if len(ordered_scores) >= 5:
        select_threshold = max(
            select_threshold,
            float(np.median(ordered_scores)),
        )

    candidates = []
    index = 0
    while index < len(ordered_segments):
        if ordered_scores[index] < select_threshold:
            index += 1
            continue

        source = ordered_segments[index]
        start = source["start"]
        end = min(source["end"], start + max_segment_sec)
        candidate_score = ordered_scores[index]
        next_index = index + 1

        while next_index < len(ordered_segments):
            next_segment = ordered_segments[next_index]
            candidate_end = max(end, next_segment["end"])
            if ordered_scores[next_index] < select_threshold:
                break
            if candidate_end - start > max_segment_sec:
                break
            end = candidate_end
            if ordered_scores[next_index] > candidate_score:
                candidate_score = ordered_scores[next_index]
                source = next_segment
            next_index += 1

        candidates.append(
            _make_output_segment(source, candidate_score, start, end)
        )
        index = max(index + 1, next_index)

    candidates = _split_long_selected(
        _merge_overlapping_selected(candidates),
        max_segment_sec,
    )
    if not candidates:
        best_index = int(np.argmax(ordered_scores))
        source = ordered_segments[best_index]
        start = source["start"]
        end = min(source["end"], start + min(max_segment_sec, budget))
        if end > start:
            candidates = [
                _make_output_segment(
                    source,
                    ordered_scores[best_index],
                    start,
                    end,
                )
            ]

    def _priority(segment: dict) -> tuple[float, float]:
        length = segment["end"] - segment["start"]
        primary = (
            segment["score"] / length
            if budget_strategy == "score_per_second"
            else segment["score"]
        )
        return primary, segment["score"]

    ranked = sorted(candidates, key=_priority, reverse=True)
    kept = []
    used = 0.0
    for segment in ranked:
        length = segment["end"] - segment["start"]
        if used + length <= budget + 1e-9:
            kept.append(segment)
            used += length

    if not kept and ranked:
        top = ranked[0]
        clipped_end = min(top["end"], top["start"] + budget)
        if clipped_end > top["start"]:
            kept = [{**top, "end": clipped_end}]

    return sorted(kept, key=lambda segment: (segment["start"], segment["end"]))
