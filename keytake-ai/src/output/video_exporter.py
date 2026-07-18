"""
影片輸出模組
將 pipeline 選出的時間段列表，用 FFmpeg 剪輯並合併成摘要影片
對應計畫書 4.3：輸出語意連貫且保留完整板書推導的課程精華
"""

import os
import json
import ffmpeg
import tempfile


def export_summary_video(
    source_video: str,
    segments: list[dict],
    output_path: str,
    padding_sec: float = 0.3
) -> str:
    """
    將選取片段剪輯並串接為摘要影片

    Args:
        source_video: 原始影片路徑
        segments: [{"start": float, "end": float}, ...]
        output_path: 輸出影片路徑
        padding_sec: 每段前後各加的緩衝秒數，避免畫面太突兀

    Returns:
        輸出影片路徑
    """
    if not segments:
        raise ValueError("沒有選取任何片段")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 取得原始影片資訊
    probe = ffmpeg.probe(source_video)
    duration = float(probe["format"]["duration"])

    # 每段個別裁切為暫存檔
    tmp_dir = tempfile.mkdtemp()
    clip_paths = []

    for i, seg in enumerate(segments):
        start = max(0.0, seg["start"] - padding_sec)
        end = min(duration, seg["end"] + padding_sec)
        clip_path = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")

        (
            ffmpeg
            .input(source_video, ss=start, to=end)
            .output(clip_path, c="copy", avoid_negative_ts="make_zero")
            .overwrite_output()
            .run(quiet=True)
        )
        clip_paths.append(clip_path)

    # 建立 concat list 檔
    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    # 串接所有片段
    (
        ffmpeg
        .input(concat_list, format="concat", safe=0)
        .output(output_path, c="copy")
        .overwrite_output()
        .run(quiet=True)
    )

    # 清理暫存檔
    for p in clip_paths:
        os.remove(p)
    os.remove(concat_list)
    os.rmdir(tmp_dir)

    summary_duration = sum(s["end"] - s["start"] for s in segments)
    print(f"[Exporter] 摘要影片已輸出：{output_path}")
    print(f"           片段數：{len(segments)}，摘要時長：{summary_duration:.1f}s")
    return output_path


def export_index(segments: list[dict], output_path: str,
                  course_summary: dict = None) -> str:
    """
    匯出精華片段索引 JSON（供前端顯示時間軸用）

    v4 升級：
      - label 優先使用 LLM 生成的 segment_summary，降級才用截斷的逐字稿
      - type 優先使用 LLM 分析的 teaching_stage，降級才用分數判斷
      - 加入 stage_label（中文教學階段標籤）
      - 加入 course_summary（全課摘要）供前端顯示課程卡片
    """
    from src.semantic.scorer import STAGE_TO_SEG_TYPE, STAGE_LABELS

    index = []
    for seg in segments:
        m, s = divmod(int(seg["start"]), 60)

        # label：優先用 LLM 摘要句，降級用截斷逐字稿
        label = seg.get("segment_summary", "").strip()
        if not label:
            label = seg.get("text", f"片段 {m:02d}:{s:02d}")[:30]
        else:
            label = label[:40]

        # type：優先用 LLM 教學階段，降級用分數
        teaching_stage = seg.get("teaching_stage", "")
        if teaching_stage and teaching_stage in STAGE_TO_SEG_TYPE:
            seg_type = STAGE_TO_SEG_TYPE[teaching_stage]
            stage_label = STAGE_LABELS.get(teaching_stage, teaching_stage)
        else:
            s_text   = seg.get("s_text", 0.0)
            s_visual = seg.get("s_visual", 0.0)
            if s_text >= 0.5 and s_visual >= 0.5:
                seg_type = "combined"
            elif s_visual >= s_text:
                seg_type = "visual"
            else:
                seg_type = "semantic"
            stage_label = {"combined": "雙模態對齊", "visual": "板書書寫", "semantic": "語意重點"}.get(seg_type, "")

        index.append({
            "timestamp": f"{m:02d}:{s:02d}",
            "start_sec":   seg["start"],
            "end_sec":     seg["end"],
            "label":       label,
            "type":        seg_type,
            "stage":       teaching_stage or seg_type,
            "stage_label": stage_label,
            "s_text":      round(seg.get("s_text", 0.0), 3),
            "s_visual":    round(seg.get("s_visual", 0.0), 3),
        })

    output_data = {
        "segments": index,
        "course_summary": course_summary or {},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[Exporter] 片段索引已輸出：{output_path}")
    return output_path
