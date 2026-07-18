"""
KeyTake AI - 主流程 (v5)
─────────────────────────────────────────────────────────────────────
v5 升級：四模態融合 + 學習性融合 + 多元輸出

新增功能：
  - 韻律特徵分析（Prosodic Features）：語速、音量、音高變化、停頓比例作為第三模態
  - CLIP 視覺-文字對齊：跳過 OCR 直接做圖文語意匹配，改善困難場域
  - MLP 可學習融合：以監督式學習取代固定 α/β 公式（可選）
  - LLM 評分快取：避免重複 API 呼叫，結果穩定且省費用
  - 多元學習素材輸出：章節標題、Markdown 筆記、心智圖、Q&A 閃卡

v4 功能（保留）：
  - LLM 對每段輸出：重要性分數 + 教學階段 + 一句摘要
  - generate_course_summary()：全課脈絡分析
  - export_index 使用 LLM 摘要句作為片段標籤

v3 功能（保留）：
  1. GRU 手勢意圖分類器：5 種意圖，每種有不同重要性分數
  2. LLM 語意評分：支援 OpenAI / Groq / Ollama
  3. 多幀視覺取樣：每段取 3 幀（頭25%/中50%/尾75%）
"""

import cv2
import os
import jieba
import numpy as np
from src.preprocessing.preprocessor import preprocess
from src.semantic.transcriber import transcribe
from src.semantic.scorer import SemanticScorer
from src.semantic.prosodic_analyzer import ProsodicAnalyzer
from src.visual.hand_tracker import HandTracker
from src.visual.visual_scorer import compute_visual_score_v2, SRGANEnhancer
from src.visual.text_detector import TextDetector
from src.visual.sbert_calculator import SBERTCalculator
from src.visual.clip_scorer import CLIPScorer
from src.visual.visual_reliability import (
    VisualReliabilityEstimator,
    build_signals_from_tracker_result,
)
from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
from src.fusion.mlp_fusion import fuse_scores_mlp
from src.fusion.evaluator import compute_false_alarm_rate, compute_recall, compute_time_saving_rate
from src.output.study_materials import StudyMaterialsGenerator
from data.prompt_corpus.corpus import PROMPT_CORPUS
from config import ALPHA, BETA, FUSION_SCORE_THRESHOLD

try:
    from config import TARGET_FALSE_ALARM_RATE
except ImportError:
    TARGET_FALSE_ALARM_RATE = 0.25

try:
    from config import PROSODIC_WEIGHT, USE_MLP_FUSION, CLIP_WEIGHT
except ImportError:
    PROSODIC_WEIGHT = 0.15
    USE_MLP_FUSION = False
    CLIP_WEIGHT = 0.3

# 每個片段取幾幀做視覺分析（v3 新增）
VISUAL_FRAMES_PER_SEGMENT = 3


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
    
    _progress(1, "LLM 語意評分與教學結構分析中...")
    scorer = SemanticScorer(prompt_corpus=PROMPT_CORPUS)
    segments = scorer.score(segments)

    # v4 新增：全課結構分析（LLM 模式才有意義，SBERT 模式也會回傳基本結果）
    _progress(1, "全課脈絡分析中...")
    course_summary = scorer.generate_course_summary(segments)
    if course_summary.get("title"):
        print(f"\n[課程分析] 標題：{course_summary['title']}")
        print(f"           摘要：{course_summary.get('summary', '')[:80]}")

    # ── 步驟 2.5：韻律特徵提取（v5 新增）─────────────────────
    print("\n[Step 2.5] 韻律特徵提取...")
    _progress(1, "分析語速、音量、音高變化...")
    prosodic_analyzer = ProsodicAnalyzer()
    segments = prosodic_analyzer.analyze_segments(paths["audio"], segments)
    avg_prosodic = np.mean([seg.get("s_prosodic", 0.5) for seg in segments])
    print(f"           平均韻律分數：{avg_prosodic:.3f}")

    # ── 步驟三：視覺特徵提取（逐幀分析 - v2架構）────────────────
    print("\n[Step 3] 視覺特徵提取 (v2 架構 + CLIP 跨模態對齊)...")
    _progress(2, "初始化視覺與語意模型...")
    
    # 【關鍵修正】在迴圈外初始化所有模型，避免 Memory Out
    tracker = HandTracker()
    text_detector = TextDetector()
    sbert_calculator = SBERTCalculator()
    srgan_enhancer = SRGANEnhancer()
    clip_scorer = CLIPScorer.get_instance()  # v5：CLIP 跨模態對齊
    reliability_estimator = VisualReliabilityEstimator()  # v2：視覺可靠度估測器
    
    cap = cv2.VideoCapture(paths["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    _progress(2, "執行多模態特徵提取（多幀取樣）...")
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # v3 多幀取樣：取片段的 25% / 50% / 75% 三個時間點
        sample_times = [
            seg_start + (seg_end - seg_start) * ratio
            for ratio in [0.25, 0.50, 0.75]
        ]

        frame_scores = []
        best_event = None

        for sample_time in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            event = tracker.process_frame(frame)

            # 用手勢意圖分數計算視覺可靠度
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

            if event["triggered"] and event["roi_center"]:
                roi = tracker.extract_roi(frame, event["roi_center"])
                asr_keywords = extract_keywords(seg["text"])
                v_result = compute_visual_score_v2(
                    roi=roi,
                    hand_center=event["roi_center"],
                    asr_keywords=asr_keywords,
                    frame=frame,
                    text_detector=text_detector,
                    sbert_calculator=sbert_calculator,
                    srgan_enhancer=srgan_enhancer,
                )
                # v5：CLIP 跨模態對齊分數（與 OCR→SBERT 路徑互補）
                clip_score = clip_scorer.compute_clip_score_with_keywords(
                    roi, asr_keywords
                ) if clip_scorer.is_available else 0.0
                # 混合 CLIP 與 OCR→SBERT 分數
                combined_visual = (
                    (1 - CLIP_WEIGHT) * v_result.score + CLIP_WEIGHT * clip_score
                ) if clip_score > 0 else v_result.score

                # v3：用手勢意圖分數調整視覺分數
                intent_score = event.get("intent_score", 1.0)
                adjusted_score = combined_visual * intent_score
            else:
                # 未觸發時也用意圖分數（比純變異數更準確）
                adjusted_score = event.get("intent_score", event["s_visual_raw"])
                reliability_result.reliability = reliability_result.reliability * 0.5

            frame_scores.append({
                "visual_score": adjusted_score,
                "reliability": reliability_result.reliability,
                "gesture_label": event.get("gesture_result", {}).label
                    if hasattr(event.get("gesture_result", None), "label") else "unknown",
            })

        if frame_scores:
            # 取三幀中視覺分數最高的那幀作為代表
            best = max(frame_scores, key=lambda x: x["visual_score"])
            seg["s_visual"] = best["visual_score"]
            seg["visual_reliability"] = best["reliability"]
            seg["gesture_label"] = best.get("gesture_label", "unknown")
        else:
            seg["s_visual"] = 0.0
            seg["visual_reliability"] = 0.0
            seg["gesture_label"] = "無手部"

    cap.release()

    # ── 步驟四：多模態融合 + 動態剪輯 ───────────────────
    print("\n[Step 4] 多模態融合（語意 × 視覺 × 手勢意圖 × 韻律）...")
    _progress(3, "四模態融合分數計算中...")

    if USE_MLP_FUSION:
        # v5：使用 MLP 可學習融合模型（需先訓練）
        print("  [融合策略] MLP 可學習融合")
        fusion_scores = fuse_scores_mlp(segments)
    else:
        # 原始解析式融合 + 韻律加權
        fusion_scores = []
        for seg in segments:
            # 基礎融合分數（語意 × 視覺，含可靠度加權）
            base_score = fuse_scores(
                seg["s_text"],
                seg["s_visual"],
                ALPHA,
                BETA,
                visual_reliability=seg.get("visual_reliability", 1.0),
            )
            # v5：韻律加權（韻律分數高的片段獲得提升）
            prosodic = seg.get("s_prosodic", 0.5)
            # 韻律提升：base * (1 + PROSODIC_WEIGHT * (prosodic - 0.5) * 2)
            # 韻律分數 0.5 時無影響，高於 0.5 提升，低於 0.5 壓低
            prosodic_boost = 1.0 + PROSODIC_WEIGHT * (prosodic - 0.5) * 2
            final_score = float(np.clip(base_score * prosodic_boost, 0.0, 1.0))
            fusion_scores.append(final_score)

    selected = semantic_sliding_window(segments, fusion_scores)

    summary_duration = sum(s["end"] - s["start"] for s in selected)
    time_saving_rate = 1 - (summary_duration / total_duration) if total_duration > 0 else 0

    # 手勢分布統計（供分析用）
    gesture_counts: dict = {}
    for seg in segments:
        label = seg.get("gesture_label", "unknown")
        gesture_counts[label] = gesture_counts.get(label, 0) + 1

    print(f"\n[完成] 原始時長: {total_duration:.1f}s → 摘要時長: {summary_duration:.1f}s")
    print(f"       時間節省率: {time_saving_rate:.1%}")
    print(f"       選取片段數: {len(selected)}")
    print(f"       手勢分布: {gesture_counts}")

    # ── 效能評估指標 (Requirement 4) ───────────────────
    result_dict = {
        "segments": selected,
        "summary_duration": summary_duration,
        "original_duration": total_duration,
        "time_saving_rate": time_saving_rate,
        "course_summary": course_summary,          # v4：全課摘要供前端顯示
        "gesture_distribution": gesture_counts,    # v4：手勢分布統計
    }

    # v5：自動產生學習素材（章節標題、筆記、心智圖、閃卡）
    try:
        _progress(3, "產生學習素材中...")
        materials_gen = StudyMaterialsGenerator()
        materials_paths = materials_gen.export_all(
            selected, course_summary, output_dir
        )
        result_dict["study_materials"] = materials_paths
        print(f"\n[學習素材] 已匯出至 {output_dir}")
    except Exception as e:
        print(f"\n[學習素材] 產生失敗（不影響主流程）：{e}")
        result_dict["study_materials"] = None

    if ground_truth:
        recall = compute_recall(selected, ground_truth)
        far = compute_false_alarm_rate(selected, ground_truth, total_duration)
        tsr = compute_time_saving_rate(selected, total_duration)
        
        print(f"\n[效能評估指標]")
        print(f"  - 重點召回率 (Recall):  {recall:.1%}")
        
        far_status = "達成目標 ✓" if far < TARGET_FALSE_ALARM_RATE else "未達目標 ✗"
        print(f"  - 誤報率 (FAR):         {far:.1%} (目標 < {TARGET_FALSE_ALARM_RATE:.1%}) {far_status}")
        print(f"  - 時間節省率 (TSR):     {tsr:.1%}")
        
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