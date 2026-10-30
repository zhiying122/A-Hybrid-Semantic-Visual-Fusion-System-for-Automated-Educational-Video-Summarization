"""片段特徵抽取模組

獨立執行語意與視覺特徵抽取並輸出結果，
供消融實驗與重複評估使用，避免頻繁觸發耗時的模型推論。
"""

import os
import cv2

from src.preprocessing.preprocessor_optimized import preprocess
from src.semantic.transcriber_optimized import transcribe
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
from data.prompt_corpus.corpus import PROMPT_CORPUS
from config import BETA, CLIP_WEIGHT
from src.fusion.mlp_fusion import MLP_FEATURE_VERSION

# 將停用詞提升為全域常數，避免每次呼叫函式時重複建構 Set 消耗效能
STOP_WORDS = {
    "的", "是", "在", "了", "這", "那", "就", "跟", "和", "這個",
    "我們", "可以", "來", "看"
}


def _extract_keywords(text: str) -> list[str]:
    # 保留 jieba 的延遲載入 (Lazy Import)，避免不需要時拖慢模組初始化
    import jieba
    words = jieba.lcut(text)
    kws = [w for w in words if w not in STOP_WORDS and len(w.strip()) > 1]
    return kws if kws else [text]


def extract_features(video_path: str, output_dir: str = "output/_features") -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)

    paths = preprocess(video_path, output_dir)
    segments = transcribe(paths["audio"])
    
    scorer = SemanticScorer(prompt_corpus=PROMPT_CORPUS)
    segments = scorer.score(segments)
    segments = ProsodicAnalyzer().analyze_segments(paths["audio"], segments)

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

    try:
        for seg in segments:
            seg_start, seg_end = seg["start"], seg["end"]
            sample_times = [seg_start + (seg_end - seg_start) * r for r in (0.25, 0.50, 0.75)]
            
            frame_scores = []
            for t in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                h, w = frame.shape[:2]
                event = tracker.process_frame(frame)
                text_detection = text_detector.detect(frame)
                if event.get("triggered") and event.get("roi_center"):
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

                kws = _extract_keywords(seg["text"])
                v_result = compute_visual_score_v2(
                    roi=roi, hand_center=roi_hand_center, asr_keywords=kws,
                    frame=frame, text_detector=text_detector,
                    sbert_calculator=sbert_calculator, srgan_enhancer=srgan_enhancer,
                )
                clip_score = (
                    clip_scorer.compute_clip_score_with_keywords(roi, kws)
                    if clip_scorer.is_available else 0.0
                )
                combined_visual = (
                    (1 - CLIP_WEIGHT) * v_result.score + CLIP_WEIGHT * clip_score
                ) if clip_score > 0 else v_result.score
                has_hand_trigger = bool(
                    event.get("triggered") and event.get("roi_center")
                )
                visual_score = (
                    combined_visual * event.get("intent_score", 1.0)
                    if has_hand_trigger else combined_visual
                )
                event["visual_available"] = bool(
                    v_result.text_detected or clip_score > 0 or v_result.score > 0
                )
                signals = build_signals_from_tracker_result(
                    event, text_boxes=text_detection.boxes, frame_w=w, frame_h=h
                )
                reliability = reliability_estimator.estimate(signals, base_beta=BETA)

                frame_scores.append(
                    (
                        visual_score,
                        reliability.reliability,
                        event.get("intent_score", 0.0)
                        if event.get("hand_detected")
                        else 0.0,
                    )
                )

            if frame_scores:
                # 取該段落中視覺分數最高的幀作為代表
                best_score, best_reliability, best_intent = max(
                    frame_scores, key=lambda x: x[0]
                )
                seg["s_visual"] = float(best_score)
                seg["visual_reliability"] = float(best_reliability)
                seg["gesture_intent_score"] = float(best_intent)
            else:
                seg["s_visual"] = 0.0
                seg["visual_reliability"] = 0.0
                seg["gesture_intent_score"] = 0.0
    finally:
        cap.release()
        tracker.close()

    return [
        {
            "start": round(s["start"], 3),
            "end": round(s["end"], 3),
            "text": s.get("text", ""),
            "s_text": round(float(s.get("s_text", 0.0)), 4),
            "s_visual": round(float(s.get("s_visual", 0.0)), 4),
            "visual_reliability": round(float(s.get("visual_reliability", 0.0)), 4),
            "gesture_intent_score": round(float(s.get("gesture_intent_score", 0.0)), 4),
            "s_prosodic": round(float(s.get("s_prosodic", 0.5)), 4),
            "prosodic_available": bool(s.get("prosodic_available", False)),
            "teaching_stage": s.get("teaching_stage", "unclassified"),
            "feature_version": MLP_FEATURE_VERSION,
        }
        for s in segments
    ]