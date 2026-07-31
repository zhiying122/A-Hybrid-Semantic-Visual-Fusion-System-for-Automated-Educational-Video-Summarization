"""
KeyTake AI 一鍵體驗腳本
─────────────────────────────────────────────────────────────────────
無需影片、無需 GPU、無需 API Key，即可完整體驗系統的所有功能。

本腳本使用預置的範例標註資料，模擬完整的分析 → 融合 → 輸出流程，
產出學習筆記、心智圖、閃卡等所有素材，並執行效能評估。

用法：
  cd keytake-ai
  python tools/demo.py

輸出：
  demo_output/
  ├── study_notes.md      ← Markdown 學習筆記
  ├── mind_map.json       ← 心智圖結構
  ├── flashcards.json     ← Q&A 閃卡
  ├── chapters.json       ← 章節標題
  ├── index.json          ← 精華片段索引
  └── eval_report.json    ← 效能評估報告
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
from src.fusion.evaluator import compute_recall, compute_false_alarm_rate, compute_time_saving_rate
from src.output.video_exporter import export_index
from src.output.study_materials import StudyMaterialsGenerator
from config import ALPHA, BETA, FUSION_SCORE_THRESHOLD


def load_example_data():
    """載入範例標註資料"""
    ann_dir = os.path.join(os.path.dirname(__file__), "..", "data", "annotations", "example")
    all_segments = []
    all_gt = []

    for i in range(1, 4):
        seg_path = os.path.join(ann_dir, f"segments_example_{i:02d}.json")
        gt_path = os.path.join(ann_dir, f"gt_example_{i:02d}.json")

        if not os.path.exists(seg_path) or not os.path.exists(gt_path):
            continue

        with open(seg_path, encoding="utf-8") as f:
            segments = json.load(f)
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)

        # 為範例片段補充 v4/v5 欄位（模擬完整 pipeline 輸出）
        for j, seg in enumerate(segments):
            seg.setdefault("s_prosodic", 0.5 + (seg.get("s_text", 0.5) - 0.5) * 0.3)
            seg.setdefault("visual_reliability", 0.8)
            seg.setdefault("gesture_label", "指引" if seg.get("s_visual", 0) > 0.5 else "無手部")
            # 模擬教學階段
            if seg.get("s_text", 0) > 0.85:
                seg["teaching_stage"] = "definition"
                seg["stage_label"] = "概念定義"
            elif seg.get("s_text", 0) > 0.7:
                seg["teaching_stage"] = "derivation"
                seg["stage_label"] = "公式推導"
            elif seg.get("s_text", 0) > 0.5:
                seg["teaching_stage"] = "example"
                seg["stage_label"] = "例題說明"
            else:
                seg["teaching_stage"] = "transition"
                seg["stage_label"] = "過渡換場"
            seg["segment_summary"] = seg.get("text", "")[:20]

        all_segments.extend(segments)
        ground_truth = [(g["start_sec"], g["end_sec"]) for g in gt_data["ground_truth"]]
        all_gt.extend(ground_truth)

    return all_segments, all_gt


def main():
    print("=" * 60)
    print("  KeyTake AI — 一鍵體驗（使用範例標註資料）")
    print("=" * 60)

    # 載入範例資料
    print("\n[1/5] 載入範例標註資料...")
    segments, ground_truth = load_example_data()
    total_duration = segments[-1]["end"] if segments else 0
    print(f"      {len(segments)} 個片段，{len(ground_truth)} 個 Ground Truth")
    print(f"      總時長：{total_duration:.1f} 秒")

    # 多模態融合
    print(f"\n[2/5] 多模態融合（α={ALPHA}, β={BETA}）...")
    fusion_scores = []
    for seg in segments:
        base = fuse_scores(
            seg.get("s_text", 0.0),
            seg.get("s_visual", 0.0),
            ALPHA, BETA,
            visual_reliability=seg.get("visual_reliability", 1.0),
        )
        # 韻律加權
        prosodic = seg.get("s_prosodic", 0.5)
        boosted = base * (1.0 + 0.15 * (prosodic - 0.5) * 2)
        fusion_scores.append(min(1.0, max(0.0, boosted)))

    selected = semantic_sliding_window(segments, fusion_scores)
    print(f"      選取 {len(selected)} 個精華片段")

    # 效能評估
    print("\n[3/5] 效能評估...")
    recall = compute_recall(selected, ground_truth)
    far = compute_false_alarm_rate(selected, ground_truth, total_duration)
    tsr = compute_time_saving_rate(selected, total_duration)
    summary_duration = sum(s["end"] - s["start"] for s in selected)

    print(f"      重點召回率 (Recall)  : {recall:.1%}")
    print(f"      誤報率 (FAR)         : {far:.1%}")
    print(f"      時間節省率 (TSR)     : {tsr:.1%}")
    print(f"      摘要時長             : {summary_duration:.1f}s / {total_duration:.1f}s")

    # 產出學習素材
    print("\n[4/5] 產生學習素材...")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "demo_output")
    os.makedirs(output_dir, exist_ok=True)

    course_summary = {
        "title": "線性代數 — 矩陣基礎與行列式",
        "summary": "本節課介紹矩陣的定義、基本運算，以及行列式為零的幾何意義。",
        "key_concepts": ["矩陣定義", "行列式", "線性變換", "零空間"],
    }

    # 匯出片段索引
    index_path = os.path.join(output_dir, "index.json")
    export_index(selected, index_path, course_summary=course_summary)

    # 產生學習素材
    materials = StudyMaterialsGenerator()
    paths = materials.export_all(selected, course_summary, output_dir)

    # 儲存評估報告
    eval_report = {
        "recall": round(recall, 4),
        "false_alarm_rate": round(far, 4),
        "time_saving_rate": round(tsr, 4),
        "original_duration_sec": round(total_duration, 1),
        "summary_duration_sec": round(summary_duration, 1),
        "segments_total": len(segments),
        "segments_selected": len(selected),
        "ground_truth_count": len(ground_truth),
        "fusion_params": {"alpha": ALPHA, "beta": BETA},
    }
    eval_path = os.path.join(output_dir, "eval_report.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    # 結果摘要
    print(f"\n[5/5] 完成！所有輸出存放於 {output_dir}/")
    print(f"      ├── study_notes.md    （學習筆記）")
    print(f"      ├── mind_map.json     （心智圖）")
    print(f"      ├── flashcards.json   （Q&A 閃卡）")
    print(f"      ├── chapters.json     （章節標題）")
    print(f"      ├── index.json        （精華片段索引）")
    print(f"      └── eval_report.json  （效能評估）")

    print(f"\n{'='*60}")
    print(f"  效能摘要")
    print(f"  Recall: {recall:.1%}  |  FAR: {far:.1%}  |  TSR: {tsr:.1%}")
    print(f"{'='*60}")

    # 顯示學習筆記預覽
    notes_path = os.path.join(output_dir, "study_notes.md")
    if os.path.exists(notes_path):
        with open(notes_path, encoding="utf-8") as f:
            preview = f.read()[:500]
        print(f"\n─── 學習筆記預覽 ───")
        print(preview)
        if len(preview) >= 500:
            print("... (更多內容請查看 demo_output/study_notes.md)")


if __name__ == "__main__":
    main()
