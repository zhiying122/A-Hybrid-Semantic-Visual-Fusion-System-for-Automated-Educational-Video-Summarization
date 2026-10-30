"""KeyTake AI — 主流程

完整摘要 pipeline 入口。依序執行：
  1. 影音前處理（FFmpeg 格式轉換、音訊提取）
  2. 語音轉錄（Whisper）+ 語意評分（SBERT / LLM）
  3. 韻律特徵提取（librosa）
  4. 視覺特徵提取（MediaPipe + GRU + OCR + CLIP）
  5. 多模態融合 + 語意感知滑動視窗選段
  6. 輸出摘要影片 + 學習素材
"""

import cv2
import os
import jieba
import numpy as np
from src.preprocessing.preprocessor_optimized import preprocess
from src.semantic.transcriber_optimized import transcribe_with_metadata
from src.semantic.semantic_scorer_optimized import SemanticScorer
from src.semantic.prosodic_analyzer_optimized import ProsodicAnalyzer
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
from src.output.study_materials_generator_optimized import StudyMaterialsGenerator
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

_STOP_WORDS = {"的", "是", "在", "了", "這", "那", "就", "跟", "和", "這個", "我們", "可以", "來", "看"}


def extract_keywords(text: str) -> list[str]:
    """jieba 斷詞後過濾停用詞，供 OCR 字元重疊備援使用。"""
    words = jieba.lcut(text)
    keywords = [w for w in words if w not in _STOP_WORDS and len(w.strip()) > 1]
    return keywords if keywords else [text]


def _apply_prosodic_adjustment(base_score: float, segment: dict) -> float:
    """只在韻律可用時以中心化倍率微調基礎融合分數；缺值不改變原分數。"""
    available = segment.get("prosodic_available", "s_prosodic" in segment)
    if not isinstance(available, bool):
        raise ValueError("prosodic_available 必須是布林值")
    if not available:
        return float(base_score)

    prosodic = float(segment["s_prosodic"])
    if not np.isfinite(prosodic) or not 0.0 <= prosodic <= 1.0:
        raise ValueError("s_prosodic 必須是介於 0 與 1 的有限數值")
    # 以 0.5 為中性點：高於中性提高分數，低於中性降低分數，再裁回 [0, 1]。
    prosodic_boost = 1.0 + PROSODIC_WEIGHT * (prosodic - 0.5) * 2
    return float(np.clip(base_score * prosodic_boost, 0.0, 1.0))


def run_pipeline(
    video_path: str,
    output_dir: str = "output",
    progress_callback=None,
    ground_truth: list[tuple[float, float]] = None
) -> dict:
    """執行完整摘要 Pipeline。

    Returns:
        dict，含 segments、summary_duration、original_duration 等欄位。
    """
    def _progress(step: int, message: str = ""):
        if progress_callback:
            progress_callback(step, message)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[Step 1] 影音前處理...")
    _progress(0, "影音前處理中...")
    paths = preprocess(video_path, output_dir)

    print("\n[Step 2] 語音轉錄與語意評分...")
    _progress(1, "Whisper 語音轉錄中...")
    transcription = transcribe_with_metadata(paths["audio"])
    segments = transcription["segments"]

    _progress(1, "LLM 語意評分與教學結構分析中...")
    scorer = SemanticScorer(prompt_corpus=PROMPT_CORPUS)
    segments = scorer.score(segments)

    _progress(1, "全課脈絡分析中...")
    course_summary = scorer.generate_course_summary(segments)
    if course_summary.get("title"):
        print(f"\n[課程分析] 標題：{course_summary['title']}")
        print(f"           摘要：{course_summary.get('summary', '')[:80]}")

    print("\n[Step 2.5] 韻律特徵提取...")
    _progress(1, "分析語速、音量、音高變化...")
    prosodic_analyzer = ProsodicAnalyzer()
    segments = prosodic_analyzer.analyze_segments(paths["audio"], segments)
    available_prosodic = [
        float(seg["s_prosodic"])
        for seg in segments
        if seg.get("prosodic_available", "s_prosodic" in seg)
    ]
    if available_prosodic:
        avg_prosodic = float(np.mean(available_prosodic))
        print(f"           平均韻律分數：{avg_prosodic:.3f}")
    else:
        print("           韻律分析不可用；融合時略過韻律加權")

    print("\n[Step 3] 視覺特徵提取（MediaPipe + GRU + OCR + CLIP）...")
    _progress(2, "初始化視覺與語意模型...")

    tracker = HandTracker()
    text_detector = TextDetector()
    sbert_calculator = SBERTCalculator()
    srgan_enhancer = SRGANEnhancer()
    clip_scorer = CLIPScorer.get_instance()
    reliability_estimator = VisualReliabilityEstimator()

    cap = cv2.VideoCapture(paths["video"])
    if not cap.isOpened():
        cap.release()
        tracker.close()
        raise RuntimeError(f"無法開啟前處理後的影片：{paths['video']}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
        cap.release()
        tracker.close()
        raise RuntimeError(f"影片影格率或影格數無效：{paths['video']}")
    total_duration = total_frames / fps

    _progress(2, "執行多模態特徵提取（多幀取樣）...")
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # 每片段取頭/中/尾三個時間點採樣，降低單幀噪聲
        sample_times = [
            seg_start + (seg_end - seg_start) * ratio
            for ratio in [0.25, 0.50, 0.75]
        ]

        frame_scores = []

        for sample_time in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            event = tracker.process_frame(frame)

            text_detection = text_detector.detect(frame)
            if event["triggered"] and event["roi_center"]:
                roi, (roi_x, roi_y) = tracker.extract_roi_with_origin(
                    frame, event["roi_center"]
                )
                roi_hand_center = (
                    event["roi_center"][0] - roi_x,
                    event["roi_center"][1] - roi_y,
                )
            else:
                roi = frame
                roi_hand_center = None

            asr_keywords = extract_keywords(seg["text"])
            v_result = compute_visual_score_v2(
                roi=roi,
                hand_center=roi_hand_center,
                asr_keywords=asr_keywords,
                frame=frame,
                text_detector=text_detector,
                sbert_calculator=sbert_calculator,
                srgan_enhancer=srgan_enhancer,
            )
            clip_score = (
                clip_scorer.compute_clip_score_with_keywords(roi, asr_keywords)
                if clip_scorer.is_available
                else 0.0
            )
            # CLIP 可用且為正時才與 OCR 視覺分數加權；否則保留 OCR 路徑結果，避免缺測被當成負證據。
            combined_visual = (
                (1 - CLIP_WEIGHT) * v_result.score + CLIP_WEIGHT * clip_score
            ) if clip_score > 0 else v_result.score
            has_hand_trigger = bool(event["triggered"] and event["roi_center"])
            adjusted_score = (
                combined_visual * event.get("intent_score", 1.0)
                if has_hand_trigger
                else combined_visual
            )
            event["visual_available"] = bool(
                v_result.text_detected or clip_score > 0 or v_result.score > 0
            )

            reliability_signals = build_signals_from_tracker_result(
                event,
                text_boxes=text_detection.boxes,
                frame_w=w,
                frame_h=h,
            )
            reliability_result = reliability_estimator.estimate(
                reliability_signals, base_beta=BETA
            )

            frame_scores.append({
                "visual_score": adjusted_score,
                "reliability": reliability_result.reliability,
                "gesture_intent_score": event.get("intent_score", 0.0)
                    if event.get("hand_detected") else 0.0,
                "gesture_label": event.get("gesture_result", {}).label
                    if hasattr(event.get("gesture_result", None), "label") else "unknown",
            })

        if frame_scores:
            best = max(frame_scores, key=lambda x: x["visual_score"])
            seg["s_visual"] = best["visual_score"]
            seg["visual_reliability"] = best["reliability"]
            seg["gesture_intent_score"] = best["gesture_intent_score"]
            seg["gesture_label"] = best.get("gesture_label", "unknown")
        else:
            seg["s_visual"] = 0.0
            seg["visual_reliability"] = 0.0
            seg["gesture_intent_score"] = 0.0
            seg["gesture_label"] = "無手部"

    cap.release()

    print("\n[Step 4] 多模態融合（語意 × 視覺 × 手勢意圖 × 韻律）...")
    _progress(3, "四模態融合分數計算中...")

    # MLP 與解析式融合是互斥路徑；目前旗標預設關閉，避免小樣本模型改變正式輸出。
    if USE_MLP_FUSION:
        print("  [融合策略] MLP 可學習融合")
        fusion_scores = fuse_scores_mlp(segments)
    else:
        fusion_scores = []
        for seg in segments:
            base_score = fuse_scores(
                seg["s_text"],
                seg["s_visual"],
                ALPHA,
                BETA,
                visual_reliability=seg.get("visual_reliability", 0.0),
            )
            fusion_scores.append(_apply_prosodic_adjustment(base_score, seg))

    selected = semantic_sliding_window(
        segments,
        fusion_scores,
        original_duration=total_duration,
    )

    summary_duration = sum(s["end"] - s["start"] for s in selected)
    time_saving_rate = 1 - (summary_duration / total_duration) if total_duration > 0 else 0

    gesture_counts: dict = {}
    for seg in segments:
        label = seg.get("gesture_label", "unknown")
        gesture_counts[label] = gesture_counts.get(label, 0) + 1

    print(f"\n[完成] 原始時長: {total_duration:.1f}s → 摘要時長: {summary_duration:.1f}s")
    print(f"       時間節省率: {time_saving_rate:.1%}")
    print(f"       選取片段數: {len(selected)}")
    print(f"       手勢分布: {gesture_counts}")

    result_dict = {
        "segments": selected,
        "summary_duration": summary_duration,
        "original_duration": total_duration,
        "time_saving_rate": time_saving_rate,
        "course_summary": course_summary,
        "gesture_distribution": gesture_counts,
        "transcription": transcription["metadata"],
    }

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
    parser.add_argument(
        "--gt",
        type=str,
        help="Ground Truth JSON 檔案路徑，格式 [[10.5, 20.0], [45.0, 60.0]]",
        default=None,
    )
    args = parser.parse_args()

    gt_data = None
    if args.gt and os.path.exists(args.gt):
        with open(args.gt, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "ground_truth" in raw:
            gt_list = raw["ground_truth"]
            gt_data = []
            for g in gt_list:
                if isinstance(g, dict):
                    gt_data.append((float(g.get("start", g.get("start_sec", 0))),
                                    float(g.get("end", g.get("end_sec", 0)))))
                else:
                    gt_data.append((float(g[0]), float(g[1])))
        elif isinstance(raw, list):
            gt_data = [(float(g[0]), float(g[1])) for g in raw]

    result = run_pipeline(args.video_path, ground_truth=gt_data)

    print("\n摘要片段：")
    for seg in result["segments"]:
        print(f"  {seg['start']:.1f}s ~ {seg['end']:.1f}s")
