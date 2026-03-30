"""
跨場域泛化測試腳本（US-5）
對應計畫書 4.4：跨場域壓力測試，支援三種場域分類評估

場域分類：
  - blackboard  : 傳統黑板
  - slides      : 投影片
  - challenging : 高反光/低對比

資料夾結構（建議）：
  data/videos/blackboard/   ← 黑板影片
  data/videos/slides/       ← 投影片影片
  data/videos/challenging/  ← 高反光影片
  data/annotations/blackboard/gt_*.json
  data/annotations/slides/gt_*.json
  data/annotations/challenging/gt_*.json

用法：
  cd keytake-ai
  python tools/run_domain_eval.py --videos data/videos/ --annotations data/annotations/ --output results/domain_report.json

  # 也可以指定單一場域
  python tools/run_domain_eval.py --domain blackboard
"""

import json
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run_pipeline
from src.fusion.evaluator import (
    compute_recall,
    compute_false_alarm_rate,
    compute_time_saving_rate,
    compute_bert_score,
)
from config import (
    TARGET_RECALL,
    TARGET_FALSE_ALARM_RATE,
    TARGET_BERT_SCORE,
    TARGET_TIME_SAVING_RATE,
    DOMAIN_BLACKBOARD,
    DOMAIN_SLIDES,
    DOMAIN_CHALLENGING,
)

ALL_DOMAINS = [DOMAIN_BLACKBOARD, DOMAIN_SLIDES, DOMAIN_CHALLENGING]

DOMAIN_LABELS = {
    DOMAIN_BLACKBOARD:  "傳統黑板",
    DOMAIN_SLIDES:      "投影片",
    DOMAIN_CHALLENGING: "高反光/低對比",
}


def eval_domain(domain: str, video_dir: str, ann_dir: str, skip_bertscore: bool) -> dict:
    """對單一場域跑完整評估，回傳該場域的彙總結果"""
    domain_video_dir = os.path.join(video_dir, domain)
    domain_ann_dir   = os.path.join(ann_dir, domain)

    if not os.path.exists(domain_video_dir):
        print(f"  [跳過] 找不到影片資料夾：{domain_video_dir}")
        return {"domain": domain, "skipped": True, "reason": "影片資料夾不存在"}

    video_files = sorted(
        f for f in os.listdir(domain_video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )

    if not video_files:
        print(f"  [跳過] 資料夾內無影片：{domain_video_dir}")
        return {"domain": domain, "skipped": True, "reason": "無影片檔案"}

    print(f"\n{'='*50}")
    print(f"場域：{DOMAIN_LABELS[domain]}（{domain}）  共 {len(video_files)} 部影片")
    print(f"{'='*50}")

    video_results = []

    for i, vf in enumerate(video_files):
        vid_id    = os.path.splitext(vf)[0]
        vid_path  = os.path.join(domain_video_dir, vf)
        gt_path   = os.path.join(domain_ann_dir, f"gt_{vid_id}.json")

        print(f"  [{i+1}/{len(video_files)}] {vf}")

        if not os.path.exists(gt_path):
            print(f"    跳過（找不到 Ground Truth：{gt_path}）")
            continue

        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)
        ground_truth = [(g["start_sec"], g["end_sec"]) for g in gt_data["ground_truth"]]

        t0 = time.time()
        try:
            result = run_pipeline(vid_path, output_dir=f"output/{domain}/{vid_id}")
        except Exception as e:
            print(f"    ✗ Pipeline 失敗：{e}")
            video_results.append({"video": vf, "error": str(e)})
            continue

        elapsed  = time.time() - t0
        selected = result["segments"]
        total    = result["original_duration"]

        recall = compute_recall(selected, ground_truth)
        far    = compute_false_alarm_rate(selected, ground_truth, total)
        tsr    = compute_time_saving_rate(selected, total)

        bs = None
        if not skip_bertscore:
            summary_texts = [s.get("text", "") for s in selected if s.get("text")]
            ref_texts     = [s.get("text", "") for s in selected[:len(summary_texts)]]
            if summary_texts and ref_texts:
                try:
                    bs = compute_bert_score(summary_texts, ref_texts)
                except Exception:
                    bs = None

        status = "✓" if recall >= TARGET_RECALL else "✗"
        print(f"    {status} Recall={recall:.3f}  FAR={far:.3f}  TSR={tsr:.3f}  ({elapsed:.1f}s)")

        video_results.append({
            "video":              vf,
            "recall":             round(recall, 4),
            "false_alarm_rate":   round(far, 4),
            "bert_score":         round(bs, 4) if bs is not None else "skipped",
            "time_saving_rate":   round(tsr, 4),
            "processing_time_sec": round(elapsed, 1),
            "segments_selected":  len(selected),
        })

    # 彙總該場域指標
    valid = [r for r in video_results if "error" not in r and "skipped" not in r]
    if not valid:
        return {"domain": domain, "label": DOMAIN_LABELS[domain],
                "evaluated": 0, "videos": video_results}

    def avg(key):
        vals = [r[key] for r in valid if isinstance(r.get(key), float)]
        return round(sum(vals) / len(vals), 4) if vals else None

    domain_summary = {
        "domain":               domain,
        "label":                DOMAIN_LABELS[domain],
        "evaluated":            len(valid),
        "avg_recall":           avg("recall"),
        "avg_false_alarm_rate": avg("false_alarm_rate"),
        "avg_bert_score":       avg("bert_score") if not skip_bertscore else "skipped",
        "avg_time_saving_rate": avg("time_saving_rate"),
        "recall_pass":          (avg("recall") or 0) >= TARGET_RECALL,
        "far_pass":             (avg("false_alarm_rate") or 1) < TARGET_FALSE_ALARM_RATE,
        "tsr_pass":             (avg("time_saving_rate") or 0) >= TARGET_TIME_SAVING_RATE,
        "videos":               video_results,
    }
    return domain_summary


def print_comparison_table(domain_results: list[dict]):
    """印出跨場域效能對比表"""
    valid = [d for d in domain_results if not d.get("skipped") and d.get("evaluated", 0) > 0]
    if not valid:
        return

    print("\n" + "=" * 65)
    print("跨場域效能對比報告")
    print("=" * 65)
    header = f"{'場域':<14} {'Recall':>8} {'FAR':>8} {'BERTScore':>10} {'TSR':>8}"
    print(header)
    print("-" * 65)

    for d in valid:
        bs_str = f"{d['avg_bert_score']:.4f}" if isinstance(d.get("avg_bert_score"), float) else "skipped"
        recall_mark = "✓" if d.get("recall_pass") else "✗"
        far_mark    = "✓" if d.get("far_pass")    else "✗"
        tsr_mark    = "✓" if d.get("tsr_pass")    else "✗"
        print(
            f"{d['label']:<14} "
            f"{d['avg_recall']:>6.4f}{recall_mark} "
            f"{d['avg_false_alarm_rate']:>6.4f}{far_mark} "
            f"{bs_str:>10} "
            f"{d['avg_time_saving_rate']:>6.4f}{tsr_mark}"
        )

    print("-" * 65)
    print(f"目標值          >{TARGET_RECALL:.2f}    <{TARGET_FALSE_ALARM_RATE:.2f}    >{TARGET_BERT_SCORE:.2f}      >{TARGET_TIME_SAVING_RATE:.2f}")
    print("=" * 65)


def export_markdown_report(domain_results: list[dict], output_path: str):
    """產出 Markdown 格式跨場域對比報告（供論文直接引用）"""
    lines = [
        "# 跨場域效能對比報告\n",
        f"| 場域 | Recall | FAR | BERTScore | TSR | Recall達標 | FAR達標 | TSR達標 |",
        f"|------|--------|-----|-----------|-----|-----------|---------|---------|",
    ]
    for d in domain_results:
        if d.get("skipped") or d.get("evaluated", 0) == 0:
            lines.append(f"| {d.get('label', d['domain'])} | — | — | — | — | — | — | — |")
            continue
        bs = f"{d['avg_bert_score']:.4f}" if isinstance(d.get("avg_bert_score"), float) else "skipped"
        lines.append(
            f"| {d['label']} "
            f"| {d['avg_recall']:.4f} "
            f"| {d['avg_false_alarm_rate']:.4f} "
            f"| {bs} "
            f"| {d['avg_time_saving_rate']:.4f} "
            f"| {'✓' if d.get('recall_pass') else '✗'} "
            f"| {'✓' if d.get('far_pass') else '✗'} "
            f"| {'✓' if d.get('tsr_pass') else '✗'} |"
        )
    lines += [
        f"\n**目標值**：Recall > {TARGET_RECALL}，FAR < {TARGET_FALSE_ALARM_RATE}，"
        f"BERTScore > {TARGET_BERT_SCORE}，TSR > {TARGET_TIME_SAVING_RATE}",
    ]
    md_path = output_path.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown 報告儲存至：{md_path}")


def main():
    parser = argparse.ArgumentParser(description="跨場域泛化測試（US-5）")
    parser.add_argument("--videos",      default="data/videos",      help="影片根目錄（含 blackboard/slides/challenging 子資料夾）")
    parser.add_argument("--annotations", default="data/annotations", help="標註根目錄（同上結構）")
    parser.add_argument("--output",      default="results/domain_report.json")
    parser.add_argument("--domain",      choices=ALL_DOMAINS + ["all"], default="all",
                        help="指定單一場域或 all（預設）")
    parser.add_argument("--skip-bertscore", action="store_true", help="跳過 BERTScore（較耗時）")
    args = parser.parse_args()

    domains_to_run = ALL_DOMAINS if args.domain == "all" else [args.domain]

    all_domain_results = []
    for domain in domains_to_run:
        result = eval_domain(domain, args.videos, args.annotations, args.skip_bertscore)
        all_domain_results.append(result)

    print_comparison_table(all_domain_results)

    # 儲存完整報告
    report = {
        "targets": {
            "recall":           TARGET_RECALL,
            "false_alarm_rate": TARGET_FALSE_ALARM_RATE,
            "bert_score":       TARGET_BERT_SCORE,
            "time_saving_rate": TARGET_TIME_SAVING_RATE,
        },
        "domains": all_domain_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    export_markdown_report(all_domain_results, args.output)
    print(f"\n完整報告儲存至：{args.output}")


if __name__ == "__main__":
    main()
