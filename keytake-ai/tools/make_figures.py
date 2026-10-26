"""
論文圖表生成（真實數據，一鍵輸出 PNG）
─────────────────────────────────────────────────────────────────────
讀取既有的真實結果 JSON，產生論文用圖：

  圖 2：三策略效能長條圖（Recall / F1 / FAR，含誤差棒）
        ← results/ablation_batch.json
  圖 3：F1 隨視覺權重 β 變化曲線
        ← results/analysis_<vid>.json
  圖 4：摘要選段 vs Ground Truth 時間軸對照
        ← results/ablation_<vid>.json + data/annotations/gt_<vid>.json

輸出至 results/figures/。全部為真實數據，非模擬。

用法：
  cd keytake-ai
  python tools/make_figures.py
  # 或指定 4.3 / 4.4 用的單片
  python tools/make_figures.py --analysis results/analysis_test_10min.json \
      --timeline-features results/features_lecture_01_math_homework.json \
      --timeline-gt data/annotations/gt_lecture_01_math_homework.json
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 無視窗環境
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIG_DIR = "results/figures"
MODE_LABELS = {"semantic": "Semantic-only", "static": "Static Fusion",
               "adaptive": "Adaptive Fusion"}
MODE_COLORS = {"semantic": "#8FAADC", "static": "#C55A11", "adaptive": "#70AD47"}


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 圖 2：三策略長條圖
# ─────────────────────────────────────────────────────────────
def fig2_bar(batch_path: str):
    with open(batch_path, encoding="utf-8") as f:
        rep = json.load(f)
    agg = rep["aggregate"]

    metrics = [("recall", "Recall"), ("f1_score", "F1"), ("false_alarm_rate", "FAR")]
    modes = ["semantic", "static", "adaptive"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, mode in enumerate(modes):
        means = [agg[mode][m]["mean"] for m, _ in metrics]
        stds = [agg[mode][m]["std"] for m, _ in metrics]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4,
               label=MODE_LABELS[mode], color=MODE_COLORS[mode],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.9)
    ax.set_title(f"Fusion Strategy Comparison (n={rep['n_videos']} real videos)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_strategy_comparison.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  圖 2 已存：{out}")


# ─────────────────────────────────────────────────────────────
# 圖 3：β 敏感度曲線
# ─────────────────────────────────────────────────────────────
def fig3_beta_sweep(analysis_path: str):
    with open(analysis_path, encoding="utf-8") as f:
        rep = json.load(f)
    sweep = rep["beta_sweep"]
    betas = [r["beta"] for r in sweep]
    f1 = [r["f1_score"] for r in sweep]
    recall = [r["recall"] for r in sweep]
    far = [r["false_alarm_rate"] for r in sweep]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(betas, f1, "-o", color="#C55A11", label="F1", linewidth=2)
    ax.plot(betas, recall, "-s", color="#4472C4", label="Recall", linewidth=1.5)
    ax.plot(betas, far, "-^", color="#7F7F7F", label="FAR", linewidth=1.5)

    # 標出 F1 最佳點
    best_i = int(np.argmax(f1))
    ax.annotate(f"best F1={f1[best_i]:.3f}\n(β={betas[best_i]:.1f})",
                xy=(betas[best_i], f1[best_i]),
                xytext=(betas[best_i] + 0.15, f1[best_i] + 0.05),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    # 標出採用值 β=0.4
    ax.axvline(0.4, color="green", linestyle="--", alpha=0.6)
    ax.text(0.41, 0.05, "adopted β=0.4", color="green", fontsize=8)

    ax.set_xlabel("Visual weight β  (α = 1 − β)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.7)
    ax.set_title(f"Sensitivity to Visual Weight ({rep['video']})")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_beta_sensitivity.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  圖 3 已存：{out}")


# ─────────────────────────────────────────────────────────────
# 圖 4：時間軸對照（selected vs GT）
# ─────────────────────────────────────────────────────────────
def _load_gt(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        raw = json.load(f)
    gt_list = raw.get("ground_truth", [])
    gt = []
    for g in gt_list:
        gt.append((float(g.get("start_sec", g.get("start"))),
                   float(g.get("end_sec", g.get("end")))))
    range_end = raw.get("eval_range_sec", [0, None])[1]
    return gt, range_end


def fig4_timeline(features_path: str, gt_path: str):
    """用 adaptive 融合選段，畫 selected vs GT 時間軸。"""
    from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window
    from config import ALPHA, BETA

    with open(features_path, encoding="utf-8") as f:
        features = json.load(f)
    gt, range_end = _load_gt(gt_path)
    if range_end is None:
        range_end = max(s["end"] for s in features)

    scores = [fuse_scores(s.get("s_text", 0), s.get("s_visual", 0), ALPHA, BETA,
                          visual_reliability=s.get("visual_reliability", 1.0))
              for s in features]
    selected = semantic_sliding_window(features, scores)
    selected = [s for s in selected if s["start"] < range_end]

    vid = os.path.splitext(os.path.basename(features_path))[0].replace("features_", "")

    fig, ax = plt.subplots(figsize=(9, 2.4))
    for s, e in gt:
        ax.barh(1, min(e, range_end) - s, left=s, height=0.6,
                color="#4472C4", edgecolor="black", linewidth=0.3)
    for seg in selected:
        ax.barh(0, seg["end"] - seg["start"], left=seg["start"], height=0.6,
                color="#C55A11", edgecolor="black", linewidth=0.3)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["System\nSummary", "Ground\nTruth"])
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, range_end)
    ax.set_title(f"Summary vs Ground Truth Timeline ({vid})")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_timeline.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  圖 4 已存：{out}")


def main():
    parser = argparse.ArgumentParser(description="生成論文圖表（真實數據）")
    parser.add_argument("--batch", default="results/ablation_batch.json",
                        help="圖 2 來源：批次消融彙整 JSON")
    parser.add_argument("--analysis", default="results/analysis_test_10min.json",
                        help="圖 3 來源：單片參數敏感度 JSON")
    parser.add_argument("--timeline-features",
                        default="results/features_lecture_01_math_homework.json",
                        help="圖 4 來源：單片特徵 JSON")
    parser.add_argument("--timeline-gt",
                        default="data/annotations/gt_lecture_01_math_homework.json",
                        help="圖 4 來源：對應 GT")
    args = parser.parse_args()

    _ensure_dir()
    print("生成論文圖表（真實數據）...")

    if os.path.exists(args.batch):
        fig2_bar(args.batch)
    else:
        print(f"  [跳過圖 2] 找不到 {args.batch}")

    if os.path.exists(args.analysis):
        fig3_beta_sweep(args.analysis)
    else:
        print(f"  [跳過圖 3] 找不到 {args.analysis}")

    if os.path.exists(args.timeline_features) and os.path.exists(args.timeline_gt):
        fig4_timeline(args.timeline_features, args.timeline_gt)
    else:
        print(f"  [跳過圖 4] 找不到特徵或 GT")

    print(f"\n完成。圖檔位於 {FIG_DIR}/")


if __name__ == "__main__":
    main()
