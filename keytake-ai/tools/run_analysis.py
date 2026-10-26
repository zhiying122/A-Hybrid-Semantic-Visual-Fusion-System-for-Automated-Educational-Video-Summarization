"""
單片深度分析（真實數據，論文 4.2/4.3 小節用）
─────────────────────────────────────────────────────────────────────
在『已抽好的真實特徵』上做兩種不需額外標註成本的真實分析：

  A. 參數敏感度分析（α/β sweep）：
     固定用 adaptive 融合，掃 β ∈ {0.0, 0.1, ..., 1.0}（α = 1-β），
     看 Recall / Precision / F1 / FAR 如何隨視覺權重變化。
     → 說明「為何選定的 β 是合理的」，並顯示融合對視覺權重的敏感度。

  B. 視覺可靠度 / 分數分布：
     統計 s_text、s_visual、visual_reliability 的分布，
     說明本片視覺訊號強弱、自適應機制的作用區間。

全部使用真實特徵，非模擬。

用法：
  cd keytake-ai
  python tools/run_analysis.py \
      --features results/features_test_10min.json \
      --gt data/annotations/gt_test_10min.json \
      --output results/analysis_test_10min.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
from src.fusion.evaluator import (
    compute_recall, compute_precision, compute_f1_score, compute_false_alarm_rate,
)


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


def _clip(segments, range_end):
    if range_end is None:
        return segments
    return [{**s, "end": min(s["end"], range_end)}
            for s in segments if s["start"] < range_end]


def _eval_with_beta(features, gt, beta, range_end, eval_duration):
    """以指定 β（adaptive 模式，α=1-β）評估。"""
    alpha = 1.0 - beta
    scores = [
        fuse_scores(s.get("s_text", 0.0), s.get("s_visual", 0.0),
                    alpha, beta, visual_reliability=s.get("visual_reliability", 1.0))
        for s in features
    ]
    selected = _clip(semantic_sliding_window(features, scores), range_end)
    recall = compute_recall(selected, gt)
    precision = compute_precision(selected, gt)
    f1 = compute_f1_score(recall, precision)
    far = compute_false_alarm_rate(selected, gt, eval_duration)
    summary_dur = sum(s["end"] - s["start"] for s in selected)
    tsr = 1.0 - summary_dur / eval_duration if eval_duration > 0 else 0.0
    return {"beta": round(beta, 2), "alpha": round(alpha, 2),
            "recall": round(recall, 4), "precision": round(precision, 4),
            "f1_score": round(f1, 4), "false_alarm_rate": round(far, 4),
            "time_saving_rate": round(tsr, 4)}


def _distribution(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(arr.max()), 4),
    }


def _histogram(values, bins=10):
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return [{"range": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "count": int(counts[i])}
            for i in range(len(counts))]


def _markdown(report):
    lines = [
        f"# 單片深度分析：{report['video']}（真實數據）",
        "",
        f"> 片段數：{report['n_segments']}　GT 區間數：{report['ground_truth_count']}"
        f"　評估範圍：0~{report['eval_range_sec'][1]}s",
        "",
        "## A. 參數敏感度分析（adaptive 融合，掃視覺權重 β）",
        "",
        "| β (視覺權重) | α (語意權重) | Recall | Precision | F1 | FAR | TSR |",
        "|-------------|-------------|--------|-----------|----|----|-----|",
    ]
    for row in report["beta_sweep"]:
        lines.append(
            f"| {row['beta']:.1f} | {row['alpha']:.1f} | {row['recall']:.3f} "
            f"| {row['precision']:.3f} | {row['f1_score']:.3f} "
            f"| {row['false_alarm_rate']:.3f} | {row['time_saving_rate']:.3f} |"
        )
    best = report["best_beta_by_f1"]
    lines += [
        "",
        f"本片 F1 最佳的視覺權重：β = {best['beta']:.1f}（F1 = {best['f1_score']:.3f}）。",
        "",
        "## B. 分數與可靠度分布",
        "",
        "| 指標 | mean | std | min | p25 | median | p75 | max |",
        "|------|------|-----|-----|-----|--------|-----|-----|",
    ]
    for name, key in [("語意分數 s_text", "s_text"),
                      ("視覺分數 s_visual", "s_visual"),
                      ("視覺可靠度 R_visual", "visual_reliability")]:
        d = report["distributions"][key]
        lines.append(
            f"| {name} | {d['mean']:.3f} | {d['std']:.3f} | {d['min']:.3f} "
            f"| {d['p25']:.3f} | {d['median']:.3f} | {d['p75']:.3f} | {d['max']:.3f} |"
        )
    lines += [
        "",
        "> 說明：視覺分數與可靠度偏低時，代表本片手部多未指向板書，"
        "自適應機制會自動壓低視覺權重、以語意為主，避免視覺雜訊污染摘要。",
        "",
        "> 所有數據均來自真實影片特徵，非模擬。",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="單片深度分析（真實數據）")
    parser.add_argument("--features", required=True, help="features_*.json 路徑")
    parser.add_argument("--gt", required=True, help="Ground Truth JSON 路徑")
    parser.add_argument("--output", default="results/analysis.json")
    parser.add_argument("--range-end", type=float, default=None)
    args = parser.parse_args()

    with open(args.features, encoding="utf-8") as f:
        features = json.load(f)
    gt, meta = _load_gt(args.gt)

    range_end = args.range_end
    if range_end is None and "eval_range_sec" in meta:
        range_end = float(meta["eval_range_sec"][1])
    eval_duration = range_end if range_end is not None else max(s["end"] for s in features)

    # A. β sweep
    beta_sweep = [_eval_with_beta(features, gt, b, range_end, eval_duration)
                  for b in np.arange(0.0, 1.01, 0.1)]
    best = max(beta_sweep, key=lambda r: r["f1_score"])

    # B. 分布
    distributions = {
        key: _distribution([s.get(key, 0.0) for s in features])
        for key in ("s_text", "s_visual", "visual_reliability")
    }
    histograms = {
        key: _histogram([s.get(key, 0.0) for s in features])
        for key in ("s_text", "s_visual", "visual_reliability")
    }

    vid = os.path.splitext(os.path.basename(args.features))[0].replace("features_", "")
    report = {
        "video": vid,
        "data_type": "REAL (真實影片特徵)",
        "n_segments": len(features),
        "ground_truth_count": len(gt),
        "eval_range_sec": [0.0, round(eval_duration, 2)],
        "beta_sweep": beta_sweep,
        "best_beta_by_f1": best,
        "distributions": distributions,
        "histograms": histograms,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown(report))

    print("=" * 60)
    print(f"  單片深度分析：{vid}（真實數據）")
    print("=" * 60)
    print("  [A] 參數敏感度（β sweep）：")
    print(f"      {'β':>5}{'Recall':>9}{'Prec':>9}{'F1':>9}{'FAR':>8}")
    for r in beta_sweep:
        print(f"      {r['beta']:>5.1f}{r['recall']:>9.3f}{r['precision']:>9.3f}"
              f"{r['f1_score']:>9.3f}{r['false_alarm_rate']:>8.3f}")
    print(f"      → F1 最佳 β = {best['beta']:.1f}（F1={best['f1_score']:.3f}）")
    print("  [B] 分布（mean）：")
    for k in ("s_text", "s_visual", "visual_reliability"):
        print(f"      {k:<20} mean={distributions[k]['mean']:.3f} "
              f"median={distributions[k]['median']:.3f}")
    print(f"  報告：{args.output}")
    print(f"  表格：{md_path}")


if __name__ == "__main__":
    main()
