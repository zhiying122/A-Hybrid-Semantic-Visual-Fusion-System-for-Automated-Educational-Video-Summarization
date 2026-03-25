"""
KeyTake AI - 主流程
整合四個步驟的完整 Pipeline：
步驟一 → 步驟二 → 步驟三 → 步驟四 → 輸出摘要片段
"""

import cv2
import os
from src.preprocessing.preprocessor import preprocess
from src.semantic.transcriber import transcribe
from src.semantic.scorer import SemanticScorer
from src.visual.hand_tracker import HandTracker
from src.visual.visual_scorer import compute_visual_score
from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
from data.prompt_corpus.corpus import PROMPT_CORPUS
from config import ALPHA, BETA, FUSION_SCORE_THRESHOLD


def run_pipeline(video_path: str, output_dir: str = "output") -> dict:
    """
    完整摘要 Pipeline
    回傳：{"segments": [...], "summary_duration": float, "original_duration": float}
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 步驟一：影音前處理 ──────────────────────────────
    print("\n[Step 1] 影音前處理...")
    paths = preprocess(video_path, output_dir)

    # ── 步驟二：語音轉錄 + 語意評分 ─────────────────────
    print("\n[Step 2] 語音轉錄與語意評分...")
    segments = transcribe(paths["audio"])
    scorer = SemanticScorer(prompt_corpus=PROMPT_CORPUS)
    segments = scorer.score(segments)

    # ── 步驟三：視覺特徵提取（逐幀分析）────────────────
    print("\n[Step 3] 視覺特徵提取...")
    tracker = HandTracker()
    cap = cv2.VideoCapture(paths["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    # 為每個語意片段計算視覺分數（取片段中間幀）
    for seg in segments:
        mid_time = (seg["start"] + seg["end"]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if not ret:
            seg["s_visual"] = 0.0
            continue

        event = tracker.process_frame(frame)
        if event["triggered"] and event["roi_center"]:
            roi = tracker.extract_roi(frame, event["roi_center"])
            # 取 TF-IDF 關鍵詞作為 OCR 比對依據
            asr_keywords = [seg["text"]]
            seg["s_visual"] = compute_visual_score(roi, event["roi_center"], asr_keywords, frame)
        else:
            seg["s_visual"] = event["s_visual_raw"]

    cap.release()

    # ── 步驟四：多模態融合 + 動態剪輯 ───────────────────
    print("\n[Step 4] 多模態融合與動態剪輯...")
    fusion_scores = [fuse_scores(seg["s_text"], seg["s_visual"], ALPHA, BETA) for seg in segments]
    selected = semantic_sliding_window(segments, fusion_scores)

    summary_duration = sum(s["end"] - s["start"] for s in selected)
    print(f"\n[完成] 原始時長: {total_duration:.1f}s → 摘要時長: {summary_duration:.1f}s")
    print(f"       時間節省率: {1 - summary_duration/total_duration:.1%}")
    print(f"       選取片段數: {len(selected)}")

    return {
        "segments": selected,
        "summary_duration": summary_duration,
        "original_duration": total_duration,
        "time_saving_rate": 1 - summary_duration / total_duration
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python main.py <影片路徑>")
        sys.exit(1)
    result = run_pipeline(sys.argv[1])
    print("\n摘要片段：")
    for seg in result["segments"]:
        print(f"  {seg['start']:.1f}s ~ {seg['end']:.1f}s")
