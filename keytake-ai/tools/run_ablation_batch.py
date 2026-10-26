"""
批次消融實驗（Batch Ablation Study）— 多部真實影片，論文彙整表用
─────────────────────────────────────────────────────────────────────
對整個影片資料夾逐部跑消融實驗（純語意 / 靜態融合 / 自適應融合），
自動抽特徵（或讀快取）、逐部評估、跨影片取平均±標準差，
輸出可直接貼進論文的跨影片彙整表（JSON + Markdown）。

資料夾約定（與現有工具一致）：
  data/videos/<vid>.mp4
  data/annotations/gt_<vid>.json      ← annotator.py merge 產生的 GT

用法：
  cd keytake-ai
  python tools/run_ablation_batch.py \
      --videos data/videos \
      --annotations data/annotations \
      --output results/ablation_batch.json

  # 已抽過特徵時，改讀 results/features_<vid>.json 直接重算（快）
  python tools/run_ablation_batch.py --use-cached-features
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
from src.fusion.evaluator import (
    compute_recall, compute_precision, compute_f1_score,
    compute_false_alarm_rate,
)
from config import ALPHA, BETA

MODES = ("semantic", "static", "adaptive")
MODE_LABELS = {"semantic": "純語意", "static": "靜態融合", "adaptive": "自適應融合"}
METRICS = ("recall", "precision", "f1_score", "false_alarm_rate", "time_saving_rate")


def _load_gt(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        raw = json.load(f)
    gt_list = raw.get("ground_truth", raw if isinstance(raw, list) else [])
    gt = []
    for g in gt_list:
        if isinstance(g, dict):
            gt.append((float(g.get("start_sec", g.get("start"))),
                       float(g.get("end_sec", g.get("end")))))
        else:
            gt.append((float(g[0]), float(g[1])))
    return gt, (raw if isinstance(raw, dict) else {})


def _clip_to_range(segments, range_end):
    if range_end is None:
        return segments
    out = []
    for s in segments:
        if s["start"] >= range_end:
            continue
        out.append({**s, "end": min(s["end"], range_end)})
    return out


def _score(features, mode):
    if mode == "semantic":
        return [float(s.get("s_text", 0.0)) for s in features]
    rel = (lambda s: 1.0) if mode == "static" else (lambda s: s.get("visual_reliability", 1.0))
    return [fuse_scores(s.get("s_text", 0.0), s.get("s_visual", 0.0),
                        ALPHA, BETA, visual_reliability=rel(s)) for s in features]


def _evaluate(features, gt, mode, range_end, eval_duration):
    scores = _score(features, mode)
    selected = _clip_to_range(semantic_sliding_window(features, scores), range_end)
    recall = compute_recall(selected, gt)
    precision = compute_precision(selected, gt)
    f1 = compute_f1_score(recall, precision)
    far = compute_false_alarm_rate(selected, gt, eval_duration)
    summary_dur = sum(s["end"] - s["start"] for s in selected)
    tsr = 1.0 - summary_dur / eval_duration if eval_duration > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1_score": f1,
            "false_alarm_rate": far, "time_saving_rate": tsr}


def _get_features(vid_id, video_path, use_cached, feature_dir):
    """取得特徵：優先讀快取，否則抽取並存檔。"""
    cache_path = os.path.join(feature_dir, f"features_{vid_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            print(f"  讀取特徵快取：{cache_path}")
            return json.load(f)
    if use_cached:
        print(f"  [跳過] 找不到特徵快取且指定 --use-cached-features：{cache_path}")
        return None
    from src.fusion.feature_extractor import extract_features
    print(f"  抽取特徵中（首次較慢）：{video_path}")
    features = extract_features(video_path)
    os.makedirs(feature_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print(f"  特徵已存至：{cache_path}")
    return features


def _markdown(report):
    lines = [
        "# 批次消融實驗彙整（多部真實影片）",
        "",
        f"> 生成時間：{report['generated_at'][:19]}　"
        f"影片數：{report['n_videos']}　融合參數：α={ALPHA}, β={BETA}",
        "",
        "## 一、跨影片平均（mean ± std）",
        "",
        "| 融合策略 | Recall | Precision | F1 | FAR | TSR |",
        "|---------|--------|-----------|----|----|-----|",
    ]
    agg = report["aggregate"]
    for mode in MODES:
        a = agg[mode]
        lines.append(
            f"| {MODE_LABELS[mode]} "
            f"| {a['recall']['mean']:.3f}±{a['recall']['std']:.3f} "
            f"| {a['precision']['mean']:.3f}±{a['precision']['std']:.3f} "
            f"| {a['f1_score']['mean']:.3f}±{a['f1_score']['std']:.3f} "
            f"| {a['false_alarm_rate']['mean']:.3f}±{a['false_alarm_rate']['std']:.3f} "
            f"| {a['time_saving_rate']['mean']:.3f}±{a['time_saving_rate']['std']:.3f} |"
        )
    lines += ["", "## 二、逐部影片明細（Recall / F1 / FAR）", "",
              "| 影片 | 純語意 | 靜態融合 | 自適應融合 |",
              "|------|--------|----------|-----------|"]
    for v in report["per_video"]:
        r = v["results"]
        cell = lambda m: f"R={r[m]['recall']:.2f} F1={r[m]['f1_score']:.2f} FAR={r[m]['false_alarm_rate']:.2f}"
        lines.append(f"| {v['video']} | {cell('semantic')} | {cell('static')} | {cell('adaptive')} |")
    lines += ["", "> 所有數據均為真實影片 + 人工標註，非模擬。三策略在同一份特徵上評估，僅融合方式不同。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="批次消融實驗（多部真實影片）")
    parser.add_argument("--videos", default="data/videos")
    parser.add_argument("--annotations", default="data/annotations")
    parser.add_argument("--output", default="results/ablation_batch.json")
    parser.add_argument("--feature-dir", default="results",
                        help="特徵快取存放/讀取目錄")
    parser.add_argument("--use-cached-features", action="store_true",
                        help="只使用既有特徵快取，不重抽（找不到就跳過該片）")
    args = parser.parse_args()

    if not os.path.isdir(args.videos):
        print(f"找不到影片資料夾：{args.videos}")
        sys.exit(1)

    video_files = sorted(
        f for f in os.listdir(args.videos)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not video_files:
        print(f"資料夾內沒有影片：{args.videos}")
        sys.exit(1)

    per_video = []
    for i, vf in enumerate(video_files):
        vid_id = os.path.splitext(vf)[0]
        video_path = os.path.join(args.videos, vf)
        gt_path = os.path.join(args.annotations, f"gt_{vid_id}.json")

        print(f"\n[{i+1}/{len(video_files)}] {vf}")
        if not os.path.exists(gt_path):
            print(f"  [跳過] 找不到 Ground Truth：{gt_path}")
            continue

        gt, meta = _load_gt(gt_path)
        features = _get_features(vid_id, video_path, args.use_cached_features, args.feature_dir)
        if features is None:
            continue

        range_end = float(meta["eval_range_sec"][1]) if "eval_range_sec" in meta else None
        eval_duration = range_end if range_end is not None else max(s["end"] for s in features)

        results = {m: {k: round(v, 4) for k, v in
                       _evaluate(features, gt, m, range_end, eval_duration).items()}
                   for m in MODES}
        per_video.append({
            "video": vid_id,
            "annotator_count": meta.get("annotator_count", "unknown"),
            "cohens_kappa": meta.get("cohens_kappa"),
            "ground_truth_count": len(gt),
            "eval_range_sec": [0.0, round(eval_duration, 2)],
            "results": results,
        })
        print(f"  純語意 R={results['semantic']['recall']:.3f} | "
              f"靜態 R={results['static']['recall']:.3f} | "
              f"自適應 R={results['adaptive']['recall']:.3f}")

    if not per_video:
        print("\n沒有任何影片成功評估（缺 GT 或缺特徵快取）。")
        sys.exit(1)

    # 跨影片彙整
    aggregate = {}
    for mode in MODES:
        aggregate[mode] = {}
        for metric in METRICS:
            vals = [v["results"][mode][metric] for v in per_video]
            aggregate[mode][metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }

    report = {
        "generated_at": datetime.now().isoformat(),
        "data_type": "REAL (真實影片 + 人工標註)",
        "n_videos": len(per_video),
        "fusion_params": {"alpha": ALPHA, "beta": BETA},
        "aggregate": aggregate,
        "per_video": per_video,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown(report))

    # 終端輸出
    print("\n" + "=" * 66)
    print(f"  批次消融彙整（{len(per_video)} 部真實影片，mean±std）")
    print("=" * 66)
    print(f"  {'策略':<20}{'Recall':>10}{'Prec':>10}{'F1':>10}{'FAR':>9}{'TSR':>9}")
    print("  " + "-" * 64)
    for mode in MODES:
        a = aggregate[mode]
        print(f"  {MODE_LABELS[mode]:<18}"
              f"{a['recall']['mean']:>8.3f}±{a['recall']['std']:.2f}"
              f"{a['precision']['mean']:>7.3f}±{a['precision']['std']:.2f}"
              f"{a['f1_score']['mean']:>7.3f}±{a['f1_score']['std']:.2f}"
              f"{a['false_alarm_rate']['mean']:>7.3f}"
              f"{a['time_saving_rate']['mean']:>8.3f}")
    print("  " + "-" * 64)
    print(f"  報告：{args.output}")
    print(f"  表格：{md_path}")


if __name__ == "__main__":
    main()
