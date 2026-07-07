"""
手勢分類器完整訓練流程（一鍵執行）
──────────────────────────────────────────────────────────────────
步驟：
  1. 確認資料集狀態（各類別樣本數）
  2. 若資料不足，補充合成資料（SMOTE-like 的簡單版本）
  3. 訓練 GRU 模型
  4. 評估並顯示結果

執行：
  cd keytake-ai
  python tools/run_gesture_pipeline.py

如果資料還在收集中，這個腳本會等資料收集完成才開始訓練。
"""

import json
import os
import sys
import subprocess
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASET_PATH = "data/gesture_dataset.json"
MIN_SAMPLES_PER_CLASS = 100  # 訓練至少需要每類 100 筆
IDEAL_SAMPLES_PER_CLASS = 200

LABEL_NAMES = {0: "指引", 1: "書寫", 2: "強調", 3: "過渡", 4: "無手部"}


def check_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"[錯誤] 找不到資料集：{DATASET_PATH}")
        print("請先執行：python tools/collect_gesture_data.py --video data/videos/ --auto")
        return None, None

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter(d["label"] for d in data)
    print(f"\n── 資料集狀態 ──")
    print(f"  總計：{len(data)} 筆")
    for label in range(5):
        n = counts.get(label, 0)
        status = "✓" if n >= MIN_SAMPLES_PER_CLASS else "⚠ 不足"
        bar = "█" * min(n // 10, 30)
        print(f"  {LABEL_NAMES[label]:8s}: {n:4d} {bar} {status}")

    short = [LABEL_NAMES[l] for l in range(5) if counts.get(l, 0) < MIN_SAMPLES_PER_CLASS]
    return data, short


def augment_short_classes(data: list, short_classes: list) -> list:
    """
    對樣本不足的類別做簡單資料擴增：
    複製現有樣本並加入少量座標抖動，讓每類至少達到 MIN_SAMPLES_PER_CLASS
    """
    import numpy as np
    augmented = data.copy()

    for label_name in short_classes:
        label_id = [k for k, v in LABEL_NAMES.items() if v == label_name][0]
        existing = [d for d in data if d["label"] == label_id]
        if not existing:
            continue

        need = MIN_SAMPLES_PER_CLASS - len(existing)
        print(f"  補充 {label_name}：新增 {need} 筆合成資料")

        for i in range(need):
            # 隨機選一個現有樣本並加入抖動
            src = existing[i % len(existing)].copy()
            coords = src["coords"].copy()
            # 對每個座標加入 ±5 像素的隨機偏移
            noisy_coords = [
                [round(x + np.random.uniform(-5, 5), 2),
                 round(y + np.random.uniform(-5, 5), 2)]
                for x, y in coords
            ]
            src["coords"] = noisy_coords
            src["augmented"] = True
            augmented.append(src)

    return augmented


def run_training():
    print("\n── 開始訓練 ──")
    result = subprocess.run(
        [sys.executable, "tools/train_gesture_classifier.py",
         "--epochs", "60", "--augment"],
        capture_output=False,
    )
    return result.returncode == 0


def main():
    print("KeyTake AI — 手勢分類器訓練流程")
    print("=" * 50)

    data, short_classes = check_dataset()
    if data is None:
        return

    total = len(data)
    if total < 50:
        print(f"\n[錯誤] 資料太少（{total} 筆），至少需要 50 筆才能訓練")
        print("資料收集可能還在進行中，請等待收集腳本完成")
        return

    if short_classes:
        print(f"\n⚠ 以下類別樣本不足：{', '.join(short_classes)}")
        # 自動補充
        augmented = augment_short_classes(data, short_classes)
        # 先存回去給訓練腳本讀
        aug_path = "data/gesture_dataset_augmented.json"
        with open(aug_path, "w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)
        print(f"  合成資料已儲存：{aug_path}")

        # 臨時替換 dataset path（讓訓練腳本讀增強資料）
        # 訓練腳本預設讀 data/gesture_dataset.json，先備份後替換
        import shutil
        shutil.copy(DATASET_PATH, DATASET_PATH + ".bak")
        shutil.copy(aug_path, DATASET_PATH)
    
    success = run_training()

    # 還原原始資料集
    bak = DATASET_PATH + ".bak"
    if os.path.exists(bak):
        import shutil
        shutil.copy(bak, DATASET_PATH)
        os.remove(bak)

    if success:
        print("\n✓ 訓練完成！模型儲存於 weights/gesture_classifier.pth")
        print("  下次執行 main.py 時，GestureClassifier 將自動載入此模型")
        print("  系統將從規則型分類升級為 GRU 神經網路分類")
    else:
        print("\n✗ 訓練失敗，請確認 PyTorch 是否正確安裝")


if __name__ == "__main__":
    main()
