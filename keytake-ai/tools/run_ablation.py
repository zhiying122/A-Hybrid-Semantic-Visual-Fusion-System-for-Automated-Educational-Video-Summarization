"""
消融實驗腳本（Ablation Study）— 真實影片，論文用
─────────────────────────────────────────────────────────────────────
在「同一份真實影片特徵」上，比較三種融合策略：

  1. 純語意 (semantic-only)      : Score = s_text
  2. 靜態融合 (static fusion)    : Score = α·s_text + β·s_visual（固定 β）
  3. 自適應融合 (adaptive fusion): β_eff = β·R_visual，Score = (1-β_eff)·s_text + β_eff·s_visual

這是論文「證明自適應置信度加權有效」最直接的證據，且全程使用真實資料，
非模擬。特徵抽取一次後存成 features_*.json，之後可反覆重算不必重跑 pipeline。

用法：
  cd keytake-ai
  # 對單一影片跑消融（會先抽特徵，較慢）
  python tools/run_ablation.py --video data/videos/test_10min.mp4 \
      --gt data/annotations/gt_test_10min.json \
      --output results/ablation_test_10min.json

  # 已抽過特徵，直接用快取重算（快）
  python tools/run_ablation.py --features results/features_test_10min.json \
      --gt data/annotations/gt_test_10min.json \
      --output results/ablation_test_10min.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
from src.fusion.evaluation_data import normalize_evaluation_data
from src.fusion.evaluator import (
    compute_recall, compute_precision, compute_f1_score,
    compute_false_alarm_rate,
)
from config import ALPHA, BETA


def _load_gt(gt_path: str):
    with open(gt_path, encoding="utf-8") as f:
        raw = json.load(f)
    gt_list = raw.get("ground_truth", []) if isinstance(raw, dict) else raw
    if not isinstance(gt_list, list):
        raise ValueError(f"Ground Truth 必須是區間清單：{gt_path}")
    gt = []
    for g in gt_list:
        if isinstance(g, dict):
            s = g.get("start_sec", g.get("start"))
            e = g.get("end_sec", g.get("end"))
            gt.append((float(s), float(e)))
        else:
            gt.append((float(g[0]), float(g[1])))
    meta = raw if isinstance(raw, dict) else {}
    return gt, meta


def _clip_to_range(segments, range_end):
    if range_end is None:
        return segments
    out = []
    for s in segments:
        if s["start"] >= range_end:
            continue
        out.append({**s, "end": min(s["end"], range_end)})
    return out


def _score_segments(features: list[dict], mode: str) -> list[float]:
    """依融合模式對每段計算分數。"""
    if mode == "semantic":
        return [float(s.get("s_text", 0.0)) for s in features]
    if mode == "static":
        # 靜態：可靠度恆為 1，退化為固定 α/β
        return [
            fuse_scores(s.get("s_text", 0.0), s.get("s_visual", 0.0),
                        ALPHA, BETA, visual_reliability=1.0)
            for s in features
        ]
    if mode == "adaptive":
        return [
            fuse_scores(s.get("s_text", 0.0), s.get("s_visual", 0.0),
                        ALPHA, BETA, visual_reliability=s.get("visual_reliability", 0.0))
            for s in features
        ]
    raise ValueError(f"未知模式：{mode}")


def _evaluate(
    features,
    gt,
    mode,
    range_end,
    eval_duration,
    budget_strategy="score",
):
    scores = _score_segments(features, mode)
    selected = semantic_sliding_window(
        features,
        scores,
        original_duration=eval_duration,
        budget_strategy=budget_strategy,
    )
    selected = _clip_to_range(selected, range_end)

    recall = compute_recall(selected, gt)
    precision = compute_precision(selected, gt)
    f1 = compute_f1_score(recall, precision)
    far = compute_false_alarm_rate(selected, gt, eval_duration)
    summary_dur = sum(s["end"] - s["start"] for s in selected)
    tsr = 1.0 - summary_dur / eval_duration if eval_duration > 0 else 0.0

    return {
        "mode": mode,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "false_alarm_rate": round(far, 4),
        "time_saving_rate": round(tsr, 4),
        "selected_count": len(selected),
    }


def _markdown(report: dict) -> str:
    labels = {"semantic": "純語意", "static": "靜態融合 (固定 β)",
              "adaptive": "自適應融合 (β·R_visual)"}
    lines = [
        f"# 消融實驗結果：{report['video']}",
        "",
        f"> 資料類型：{report['data_type']}　標註者數：{report['annotator_count']}"
        f"　Cohen's Kappa：{report.get('cohens_kappa')}",
        f"> 評估範圍：0 ~ {report['eval_range_sec'][1]}s　GT 區間數：{report['ground_truth_count']}",
        "",
        "| 融合策略 | Recall | Precision | F1 | FAR | TSR |",
        "|---------|--------|-----------|----|----|-----|",
    ]
    for mode in ("semantic", "static", "adaptive"):
        m = report["results"][mode]
        lines.append(
            f"| {labels[mode]} | {m['recall']:.3f} | {m['precision']:.3f} "
            f"| {m['f1_score']:.3f} | {m['false_alarm_rate']:.3f} | {m['time_saving_rate']:.3f} |"
        )
    lines += [
        "",
        f"融合參數：α = {ALPHA}, β = {BETA}",
        "",
        "> 三種策略在同一份真實影片特徵上評估，僅融合方式不同，其餘皆相同。",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="消融實驗（真實影片）")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="影片路徑（會先抽特徵）")
    src.add_argument("--features", help="已抽好的 features_*.json（直接重算）")
    parser.add_argument("--gt", required=True, help="Ground Truth JSON 路徑")
    parser.add_argument("--output", default="results/ablation.json")
    parser.add_argument("--range-end", type=float, default=None,
                        help="評估範圍結束秒數（預設讀 GT 的 eval_range_sec）")
    parser.add_argument(
        "--budget-strategy",
        choices=("score", "score_per_second"),
        default="score",
    )
    parser.add_argument("--save-features", default=None,
                        help="抽特徵後另存路徑（預設 results/features_<vid>.json）")
    args = parser.parse_args()

    gt, meta = _load_gt(args.gt)

    # 取得特徵
    if args.features:
        with open(args.features, encoding="utf-8") as f:
            features = json.load(f)
        vid_name = os.path.splitext(os.path.basename(args.features))[0].replace("features_", "")
        print(f"讀取特徵快取：{args.features}（{len(features)} 段）")
    else:
        from src.fusion.feature_extractor import extract_features
        print(f"抽取特徵中（首次較慢）：{args.video}")
        features = extract_features(args.video)
        vid_name = os.path.splitext(os.path.basename(args.video))[0]
        feat_path = args.save_features or f"results/features_{vid_name}.json"
        os.makedirs(os.path.dirname(feat_path) or ".", exist_ok=True)
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        print(f"特徵已存至：{feat_path}（下次可用 --features 直接重算）")

    # 評估範圍
    eval_range = (
        (0.0, args.range_end)
        if args.range_end is not None
        else meta.get("eval_range_sec")
    )
    features, gt, eval_duration = normalize_evaluation_data(
        features,
        gt,
        eval_range,
    )
    range_end = eval_duration

    # 三模式評估
    results = {
        mode: _evaluate(
            features,
            gt,
            mode,
            range_end,
            eval_duration,
            budget_strategy=args.budget_strategy,
        )
               for mode in ("semantic", "static", "adaptive")}

    report = {
        "generated_at": datetime.now().isoformat(),
        "video": vid_name,
        "data_type": "REAL (真實影片 + 人工標註)",
        "annotator_count": meta.get("annotator_count", "unknown"),
        "cohens_kappa": meta.get("cohens_kappa"),
        "eval_range_sec": [0.0, round(eval_duration, 2)],
        "ground_truth_count": len(gt),
        "fusion_params": {"alpha": ALPHA, "beta": BETA},
        "budget_strategy": args.budget_strategy,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown(report))

    # 終端輸出
    print("\n" + "=" * 62)
    print(f"  消融實驗：{vid_name}（真實影片，評估範圍 0~{eval_duration:.0f}s）")
    print("=" * 62)
    print(f"  {'策略':<22}{'Recall':>8}{'Prec':>8}{'F1':>8}{'FAR':>8}{'TSR':>8}")
    print("  " + "-" * 60)
    labels = {"semantic": "純語意", "static": "靜態融合", "adaptive": "自適應融合"}
    for mode in ("semantic", "static", "adaptive"):
        m = results[mode]
        print(f"  {labels[mode]:<20}{m['recall']:>8.3f}{m['precision']:>8.3f}"
              f"{m['f1_score']:>8.3f}{m['false_alarm_rate']:>8.3f}{m['time_saving_rate']:>8.3f}")
    print("  " + "-" * 60)
    print(f"  報告：{args.output}")
    print(f"  表格：{md_path}")


if __name__ == "__main__":
    main()
