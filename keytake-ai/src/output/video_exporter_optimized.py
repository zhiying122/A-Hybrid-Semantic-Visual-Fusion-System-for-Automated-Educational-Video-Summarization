"""Export selected video highlights and their timeline index."""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import ffmpeg


def _as_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"片段欄位 {field_name!r} 必須是數字") from exc
    if not math.isfinite(number):
        raise ValueError(f"片段欄位 {field_name!r} 不可為 NaN 或無限大")
    return number


def _normalize_segments(
    segments: list[dict], duration: float, padding_sec: float = 0.0
) -> list[dict]:
    if not segments:
        raise ValueError("沒有選取任何片段")
    if padding_sec < 0:
        raise ValueError("padding_sec 不可小於 0")

    normalized = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or "start" not in segment or "end" not in segment:
            raise ValueError(f"第 {index + 1} 個片段缺少 start 或 end")

        start = _as_float(segment["start"], "start")
        end = _as_float(segment["end"], "end")
        if start < 0 or end <= start:
            raise ValueError(f"第 {index + 1} 個片段的時間範圍無效：{start} ~ {end}")
        if start >= duration:
            raise ValueError(f"第 {index + 1} 個片段起點超出影片長度")

        padded_start = max(0.0, start - padding_sec)
        padded_end = min(duration, end + padding_sec)
        if padded_end <= padded_start:
            continue

        normalized.append({
            "start": start,
            "end": min(end, duration),
            "clip_start": padded_start,
            "clip_end": padded_end,
            "source": segment,
        })

    normalized = [item for item in normalized if item["end"] > item["start"]]
    if not normalized:
        raise ValueError("沒有可輸出的有效片段")

    normalized.sort(key=lambda item: (item["clip_start"], item["clip_end"]))

    # Merge overlapping padded ranges to avoid repeating the same footage.
    merged = []
    for item in normalized:
        if merged and item["clip_start"] <= merged[-1]["clip_end"]:
            merged[-1]["clip_end"] = max(merged[-1]["clip_end"], item["clip_end"])
        else:
            merged.append({
                "clip_start": item["clip_start"],
                "clip_end": item["clip_end"],
            })
    return merged


def _probe_duration(source_video: str) -> float:
    if not os.path.isfile(source_video):
        raise FileNotFoundError(f"找不到來源影片：{source_video}")

    try:
        probe = ffmpeg.probe(source_video)
        duration = float(probe["format"]["duration"])
    except (ffmpeg.Error, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"無法讀取影片資訊：{source_video}") from exc

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("來源影片的長度無效")
    return duration


def export_summary_video(
    source_video: str,
    segments: list[dict],
    output_path: str,
    padding_sec: float = 0.3,
) -> str:
    """Cut selected ranges, concatenate them, and return the output path."""
    if not segments:
        raise ValueError("沒有選取任何片段")
    if not output_path:
        raise ValueError("必須指定輸出影片路徑")

    source = Path(source_video).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("輸出路徑不可與來源影片相同")

    duration = _probe_duration(str(source))
    clips = _normalize_segments(segments, duration, padding_sec)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="summary_export_") as temp_dir:
        temp_path = Path(temp_dir)
        clip_paths = []

        for index, clip in enumerate(clips):
            clip_path = temp_path / f"clip_{index:04d}.mp4"
            clip_duration = clip["clip_end"] - clip["clip_start"]

            try:
                (
                    ffmpeg
                    .input(str(source), ss=clip["clip_start"])
                    .output(
                        str(clip_path),
                        t=clip_duration,
                        vcodec="libx264",
                        preset="veryfast",
                        crf=23,
                        acodec="aac",
                        audio_bitrate="128k",
                        pix_fmt="yuv420p",
                        movflags="+faststart",
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as exc:
                detail = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
                raise RuntimeError(f"剪輯第 {index + 1} 段失敗：{detail[-1500:]}") from exc

            if not clip_path.exists() or clip_path.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg 未產生有效片段：{clip_path.name}")
            clip_paths.append(clip_path)

        concat_file = temp_path / "concat.txt"
        with concat_file.open("w", encoding="utf-8", newline="\n") as file:
            for clip_path in clip_paths:
                # Temporary paths are generated by this function and contain no quotes.
                file.write(f"file '{clip_path.as_posix()}'\n")

        try:
            (
                ffmpeg
                .input(str(concat_file), format="concat", safe=0)
                .output(str(destination), c="copy", movflags="+faststart")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            detail = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            raise RuntimeError(f"串接摘要影片失敗：{detail[-1500:]}") from exc

    summary_duration = sum(clip["clip_end"] - clip["clip_start"] for clip in clips)
    print(f"[Exporter] 摘要影片已輸出：{destination}")
    print(f"[Exporter] 片段數：{len(clips)}，預估摘要時長：{summary_duration:.1f}s")
    return str(destination)


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(score, 3) if math.isfinite(score) else 0.0


def _format_timestamp(seconds: Any) -> str:
    try:
        total_seconds = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total_seconds = 0
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def export_index(
    segments: list[dict],
    output_path: str,
    course_summary: dict | None = None,
) -> str:
    """Export timeline metadata as UTF-8 JSON."""
    from src.semantic.scorer import STAGE_TO_SEG_TYPE, STAGE_LABELS

    if not output_path:
        raise ValueError("必須指定索引 JSON 輸出路徑")

    index = []
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict) or "start" not in segment or "end" not in segment:
            raise ValueError(f"第 {position + 1} 個片段缺少 start 或 end")

        start = _as_float(segment["start"], "start")
        end = _as_float(segment["end"], "end")
        if start < 0 or end <= start:
            raise ValueError(f"第 {position + 1} 個片段的時間範圍無效：{start} ~ {end}")

        timestamp = _format_timestamp(start)
        text = str(segment.get("text") or "").strip()
        label = str(segment.get("segment_summary") or "").strip()
        if not label:
            label = text or f"片段 {timestamp}"
        label = label[:40]

        stage = str(segment.get("teaching_stage") or "")
        if stage in STAGE_TO_SEG_TYPE:
            segment_type = STAGE_TO_SEG_TYPE[stage]
            stage_label = STAGE_LABELS.get(stage, stage)
        else:
            text_score = _safe_score(segment.get("s_text", 0.0))
            visual_score = _safe_score(segment.get("s_visual", 0.0))
            if text_score >= 0.5 and visual_score >= 0.5:
                segment_type = "combined"
            elif visual_score >= text_score:
                segment_type = "visual"
            else:
                segment_type = "semantic"
            stage_label = {
                "combined": "雙模態對齊",
                "visual": "板書書寫",
                "semantic": "語意重點",
            }[segment_type]

        index.append({
            "timestamp": timestamp,
            "start_sec": start,
            "end_sec": end,
            "label": label,
            "type": segment_type,
            "stage": stage or segment_type,
            "stage_label": stage_label,
            "s_text": _safe_score(segment.get("s_text", 0.0)),
            "s_visual": _safe_score(segment.get("s_visual", 0.0)),
        })

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "segments": index,
        "course_summary": course_summary if isinstance(course_summary, dict) else {},
    }
    destination.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Exporter] 片段索引已輸出：{destination}")
    return str(destination)
