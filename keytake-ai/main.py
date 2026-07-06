"""
KeyTake AI - 主流程 (v2)
─────────────────────────────────────────────────────────────────────
整合四個步驟的完整 Pipeline：
步驟一 → 步驟二 → 步驟三 (v2架構) → 步驟四（自適應置信度加權融合）→ 輸出

v2 改進（回應評審意見）：
  - 整合 VisualReliabilityEstimator，動態評估視覺觀測的可靠度
  - 手勢不在板書前時自動降低視覺權重，提升摘要品質
  - 各幀 R_visual 記錄於 segment metadata，可供後續分析
"""

import cv2
import os
import jieba  # 用於萃取關鍵字
from src.preprocessing.preprocessor import preprocess
from src.semantic.transcriber import transcribe
from src.semantic.scorer import SemanticScorer
from src.visual.hand_tracker import HandTracker

# 引入 v2 視覺架構所需模組
from src.visual.visual_scorer import compute_visual_score_v2, SRGANEnhancer
from src.visual.text_detector import TextDetector
from src.visual.sbert_calculator import SBERTCalculator
from src.visual.visual_reliability import (
    VisualReliabilityEstimator,
    build_signals_from_tracker_result,
)

from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
from src.fusion.evaluator import compute_false_alarm_rate, compute_recall, compute_time_saving_rate
from data.prompt_corpus.corpus import PROMPT_CORPUS
from config import ALPHA, BETA, FUSION_SCORE_THRESHOLD

# 嘗試載入目標誤報率，若未設定則預設 0.25 (25%)
try:
    from config import TARGET_FALSE_ALARM_RATE
except ImportError:
    TARGET_FALSE_ALARM_RATE = 0.25


def extract_keywords(text: str) -> list[str]:
    """
    簡單的關鍵字萃取，過濾掉常見停用詞，
    供 OCR 字元重疊降級備援使用。
    """
    words = jieba.lcut(text)
    stop_words = {"的", "是", "在", "了", "這", "那", "就", "跟", "和", "這個", "我們", "可以", "來", "看"}
    keywords = [w for w in words if w not in stop_words and len(w.strip()) > 1]
    return keywords if keywords else [text]  # 若全被過濾，至少保留原句


def run_pipeline(
    video_path: str, 
    output_dir: str = "output", 
    progress_callback=None,
    ground_truth: list[tuple[float, float]] = None
) -> dict:
    """
    完整摘要 Pipeline
    回傳：{"segments": [...], "summary_duration": float, "original_duration": float, ...}
    """
    def _progress(step: int, message: str = ""):
        if progress_callback:
            progress_callback(step, message)

    os.makedirs(output_dir, exist_ok=True)

    # ── 步驟一：影音前處理 ──────────────────────────────
    print("\n[Step 1] 影音前處理...")
    _progress(0, "影音前處理中...")
    paths = preprocess(video_path, output_dir)

    # ── 步驟二：語音轉錄 + 語意評分 ─────────────────────
    print("\n[Step 2] 語音轉錄與語意評分...")
    _progress(1, "Whisper 語音轉錄中...")
    segments = transcribe(paths["audio"])
    
    _progress(1, "SBERT 語意評分中...")
    scorer = SemanticScorer(prompt_corpus=PROMPT_CORPUS)
    segments = scorer.score(segments)

    # ── 步驟三：視覺特徵提取（逐幀分析 - v2架構）────────────────
    print("\n[Step 3] 視覺特徵提取 (v2 架構)...")
    _progress(2, "初始化視覺與語意模型...")
    
    # 【關鍵修正】在迴圈外初始化所有模型，避免 Memory Out
    tracker = HandTracker()
    text_detector = TextDetector()
    sbert_calculator = SBERTCalculator()
    srgan_enhancer = SRGANEnhancer()
    reliability_estimator = VisualReliabilityEstimator()  # v2：視覺可靠度估測器
    
    cap = cv2.VideoCapture(paths["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    _progress(2, "執行多模態特徵提取...")
    for seg in segments:
        mid_time = (seg["start"] + seg["end"]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if not ret:
            seg["s_visual"] = 0.0
            seg["visual_reliability"] = 0.0
            continue

        h, w = frame.shape[:2]
        event = tracker.process_frame(frame)

        # v2：估算視覺可靠度
        text_detection = text_detector.detect(frame)
        reliability_signals = build_signals_from_tracker_result(
            event,
            text_boxes=text_detection.boxes,
            frame_w=w,
            frame_h=h,
        )
        reliability_result = reliability_estimator.estimate(
            reliability_signals, base_beta=BETA
        )
        seg["visual_reliability"] = reliability_result.reliability

        if event["triggered"] and event["roi_center"]:
            roi = tracker.extract_roi(frame, event["roi_center"])
            
            # 【關鍵修正】萃取關鍵字，而非丟入一整句話
            asr_keywords = extract_keywords(seg["text"])
            
            # 【關鍵修正】使用 v2 API 並傳入已實例化的模型
            v_result = compute_visual_score_v2(
                roi=roi,
                hand_center=event["roi_center"],
                asr_keywords=asr_keywords,
                frame=frame,
                text_detector=text_detector,
                sbert_calculator=sbert_calculator,
                srgan_enhancer=srgan_enhancer
            )
            seg["s_visual"] = v_result.score
        else:
            seg["s_visual"] = event["s_visual_raw"]

    cap.release()

    # ── 步驟四：多模態融合 + 動態剪輯 ───────────────────
    print("\n[Step 4] 多模態融合與動態剪輯（自適應置信度加權）...")
    _progress(3, "融合語意與視覺分數（含可靠度加權）...")
    fusion_scores = [
        fuse_scores(
            seg["s_text"],
            seg["s_visual"],
            ALPHA,
            BETA,
            visual_reliability=seg.get("visual_reliability", 1.0),
        )
        for seg in segments
    ]
    selected = semantic_sliding_window(segments, fusion_scores)

    summary_duration = sum(s["end"] - s["start"] for s in selected)
    time_saving_rate = 1 - (summary_duration / total_duration) if total_duration > 0 else 0

    print(f"\n[完成] 原始時長: {total_duration:.1f}s → 摘要時長: {summary_duration:.1f}s")
    print(f"       時間節省率: {time_saving_rate:.1%}")
    print(f"       選取片段數: {len(selected)}")

    # ── 效能評估指標 (Requirement 4) ───────────────────
    result_dict = {
        "segments": selected,
        "summary_duration": summary_duration,
        "original_duration": total_duration,
        "time_saving_rate": time_saving_rate
    }

    if ground_truth:
        recall = compute_recall(selected, ground_truth)
        far = compute_false_alarm_rate(selected, ground_truth, total_duration)
        
        print(f"\n[效能評估指標]")
        print(f"  - 重點召回率 (Recall): {recall:.1%}")
        
        far_status = "達成目標" if far < TARGET_FALSE_ALARM_RATE else "未達目標"
        print(f"  - 誤報率 (FAR): {far:.1%} (< {TARGET_FALSE_ALARM_RATE:.1%} {far_status})")
        
        result_dict["recall"] = recall
        result_dict["false_alarm_rate"] = far

    return result_dict


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="KeyTake AI 主程式")
    parser.add_argument("video_path", help="要處理的影片路徑")
    parser.add_argument("--gt", type=str, help="Ground Truth JSON 檔案路徑 (用於計算 Recall 與 FAR)，格式例如 [[10.5, 20.0], [45.0, 60.0]]", default=None)
    args = parser.parse_args()

    # 讀取 Ground Truth（支援兩種格式）
    gt_data = None
    if args.gt and os.path.exists(args.gt):
        with open(args.gt, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 格式一：annotator.py 產出的 {"ground_truth": [{"start": ..., "end": ...}]}
        if isinstance(raw, dict) and "ground_truth" in raw:
            gt_list = raw["ground_truth"]
            gt_data = []
            for g in gt_list:
                if isinstance(g, dict):
                    gt_data.append((float(g.get("start", g.get("start_sec", 0))),
                                    float(g.get("end", g.get("end_sec", 0)))))
                else:
                    gt_data.append((float(g[0]), float(g[1])))
        # 格式二：簡單陣列 [[10.5, 20.0], [45.0, 60.0]]
        elif isinstance(raw, list):
            gt_data = [(float(g[0]), float(g[1])) for g in raw]

    result = run_pipeline(args.video_path, ground_truth=gt_data)
    
    print("\n摘要片段：")
    for seg in result["segments"]:
        print(f"  {seg['start']:.1f}s ~ {seg['end']:.1f}s")