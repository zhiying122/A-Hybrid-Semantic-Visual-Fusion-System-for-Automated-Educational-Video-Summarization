"""
Grid Search 執行腳本
對應計畫書 4.2 步驟四：在驗證集上以 0.1 步長找最佳 α/β 組合

用法：
  cd keytake-ai
  python tools/run_grid_search.py --annotations data/annotations/ --output results/grid_search.json
"""

import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.adaptive_fusion import grid_search_weights, fuse_scores, semantic_sliding_window
from src.fusion.evaluator import compute_recall
from config import FUSION_SCORE_THRESHOLD


def load_annotated_video(seg_path: str, gt_path: str) -> dict:
    """載入一部影片的語意分數片段與 Ground Truth"""
    with open(seg_path, encoding="utf-8") as f:
        segments = json.load(f)
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    ground_truth = [(g["start_sec"], g["end_sec"]) for g in gt_data["ground_truth"]]
    return {"segments": segments, "ground_truth": ground_truth}


def main():
    parser = argparse.ArgumentParser(description="Grid Search 最佳 α/β 權重")
    parser.add_argument("--annotations", default="data/annotations",
                        help="標註資料夾（含 segments_*.json 與 gt_*.json）")
    parser.add_argument("--output", default="results/grid_search.json")
    parser.add_argument("--step", type=float, default=0.1, help="搜尋步長")
    args = parser.parse_args()

    # 掃描標註資料夾
    ann_dir = args.annotations
    if not os.path.exists(ann_dir):
        print(f"找不到標註資料夾：{ann_dir}")
        print("請先執行 tools/export_transcript.py 與 tools/annotator.py 產生標註資料")
        sys.exit(1)

    seg_files = sorted(f for f in os.listdir(ann_dir) if f.startswith("segments_"))
    if not seg_files:
        print("找不到 segments_*.json 檔案，請確認標註資料夾內容")
        sys.exit(1)

    all_segments, all_gt = [], []
    for sf in seg_files:
        vid_id = sf.replace("segments_", "").replace(".json", "")
        gt_file = f"gt_{vid_id}.json"
        gt_path = os.path.join(ann_dir, gt_file)
        if not os.path.exists(gt_path):
            print(f"跳過 {sf}（找不到對應的 {gt_file}）")
            continue
        data = load_annotated_video(os.path.join(ann_dir, sf), gt_path)
        all_segments.extend(data["segments"])
        all_gt.extend(data["ground_truth"])

    if not all_segments:
        print("沒有可用的標註資料")
        sys.exit(1)

    print(f"載入 {len(seg_files)} 部影片，共 {len(all_segments)} 個片段，{len(all_gt)} 個 GT")

    # 執行 Grid Search
    best_alpha, best_beta = grid_search_weights(all_segments, all_gt, step=args.step)

    # 用最佳參數計算最終 Recall
    scores = [fuse_scores(s["s_text"], s.get("s_visual", 0.0), best_alpha, best_beta)
              for s in all_segments]
    selected = [s for s, sc in zip(all_segments, scores) if sc >= FUSION_SCORE_THRESHOLD]
    final_recall = compute_recall(selected, all_gt)

    result = {
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "recall_on_validation": round(final_recall, 4),
        "total_segments": len(all_segments),
        "total_gt": len(all_gt)
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n最佳參數：α={best_alpha}, β={best_beta}")
    print(f"驗證集 Recall：{final_recall:.4f}")
    print(f"結果儲存至：{args.output}")


if __name__ == "__main__":
    main()
