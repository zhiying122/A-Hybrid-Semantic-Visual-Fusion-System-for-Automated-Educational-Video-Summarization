"""
MLP 融合模型訓練腳本
對應 v5 升級：以監督式學習取代靜態 α/β 融合公式

前置需求：
  1. 已完成標註流程（tools/annotator.py）
  2. 已完成 export_transcript.py 產生 segments_*.json
  3. 已有 gt_*.json Ground Truth 檔案

訓練完成後，在 config.py 設定 USE_MLP_FUSION = True 即可啟用。

用法：
  cd keytake-ai
  python tools/train_mlp_fusion.py --annotations data/annotations/ --output weights/mlp_fusion.pth
"""

import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.mlp_fusion import MLPFusionTrainer


def main():
    parser = argparse.ArgumentParser(description="訓練 MLP 融合模型")
    parser.add_argument("--annotations", default="data/annotations",
                        help="標註資料夾（含 segments_*.json 與 gt_*.json）")
    parser.add_argument("--output", default="weights/mlp_fusion.pth",
                        help="模型輸出路徑")
    parser.add_argument("--epochs", type=int, default=100, help="訓練輪數")
    parser.add_argument("--lr", type=float, default=0.001, help="學習率")
    args = parser.parse_args()

    ann_dir = args.annotations
    if not os.path.exists(ann_dir):
        # 嘗試 example/ 子資料夾
        example_dir = os.path.join(ann_dir, "example")
        if os.path.exists(example_dir):
            ann_dir = example_dir
        else:
            print(f"找不到標註資料夾：{ann_dir}")
            sys.exit(1)

    # 載入所有標註資料
    seg_files = sorted(f for f in os.listdir(ann_dir) if f.startswith("segments_"))
    if not seg_files:
        print("找不到 segments_*.json 檔案")
        sys.exit(1)

    all_segments = []
    all_gt = []

    for sf in seg_files:
        vid_id = sf.replace("segments_", "").replace(".json", "")
        gt_path = os.path.join(ann_dir, f"gt_{vid_id}.json")
        if not os.path.exists(gt_path):
            print(f"跳過 {sf}（找不到 {gt_path}）")
            continue

        with open(os.path.join(ann_dir, sf), encoding="utf-8") as f:
            segments = json.load(f)
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)
        ground_truth = [(g["start_sec"], g["end_sec"]) for g in gt_data["ground_truth"]]

        all_segments.extend(segments)
        all_gt.extend(ground_truth)

    if not all_segments:
        print("沒有可用的標註資料")
        sys.exit(1)

    print(f"載入 {len(seg_files)} 部影片，共 {len(all_segments)} 個片段，{len(all_gt)} 個 GT")
    print(f"開始訓練 MLP 融合模型（{args.epochs} epochs, lr={args.lr}）...\n")

    # 訓練
    trainer = MLPFusionTrainer(model_path=args.output)
    stats = trainer.train(all_segments, all_gt, epochs=args.epochs, lr=args.lr)

    print(f"\n{'='*50}")
    print(f"訓練完成！")
    print(f"  樣本數：{stats['total_samples']}（正樣本比例：{stats['positive_ratio']:.2%}）")
    print(f"  最佳 Epoch：{stats['best_epoch']}")
    print(f"  驗證 Loss：{stats['best_val_loss']:.4f}")
    print(f"  模型儲存至：{args.output}")
    print(f"\n下一步：在 config.py 設定 USE_MLP_FUSION = True 即可啟用")


if __name__ == "__main__":
    main()
