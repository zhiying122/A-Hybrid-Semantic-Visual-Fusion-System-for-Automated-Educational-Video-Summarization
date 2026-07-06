"""
手勢分類器訓練腳本
──────────────────────────────────────────────────────────────────
讀取 data/gesture_dataset.json，訓練 GRU 模型，儲存到 weights/gesture_classifier.pth

執行：
  cd keytake-ai
  python tools/train_gesture_classifier.py
  python tools/train_gesture_classifier.py --epochs 50 --lr 0.001

訓練完成後，GestureClassifier 會自動載入模型，
pipeline 中的手勢分類將從規則型升級為神經網路模型。

訓練建議：
  - 每類至少 200 筆，總計約 1000 筆時效果開始穩定
  - 如果某類資料少，可以用 --augment 開啟資料增強
  - 在沒有 GPU 的機器上訓練 50 epoch 大約需要 2~5 分鐘
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visual.gesture_classifier import (
    GestureGRU, extract_trajectory_features, pad_or_truncate,
    GestureIntent, INTENT_LABELS,
)


DATASET_PATH = "data/gesture_dataset.json"
MODEL_SAVE_PATH = "weights/gesture_classifier.pth"
SEQ_LEN = 20
INPUT_SIZE = 4


# ──────────────────────────────────────────────────────────────────
# 資料集
# ──────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    def __init__(self, samples: list[dict], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        coords = [(c[0], c[1]) for c in s["coords"]]
        frame_w = s.get("frame_w", 640)
        frame_h = s.get("frame_h", 480)

        features = extract_trajectory_features(coords, frame_w, frame_h)
        features = pad_or_truncate(features, SEQ_LEN)

        if self.augment:
            features = self._augment(features)

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(s["label"], dtype=torch.long)
        return x, y

    def _augment(self, features: np.ndarray) -> np.ndarray:
        """簡單資料增強：位置偏移 + 速度縮放 + 加入少量雜訊"""
        f = features.copy()
        # 位置微小偏移
        f[:, :2] += np.random.uniform(-0.05, 0.05, size=(SEQ_LEN, 2))
        f[:, :2] = np.clip(f[:, :2], 0, 1)
        # 速度縮放（模擬快/慢動作）
        scale = np.random.uniform(0.8, 1.2)
        f[:, 2:] *= scale
        # 高斯雜訊
        f += np.random.normal(0, 0.01, size=f.shape)
        return f.astype(np.float32)


def load_and_split(
    dataset_path: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 把不認識的 label 過濾掉
    valid_labels = {0, 1, 2, 3, 4}
    data = [d for d in data if d.get("label") in valid_labels]

    print(f"[Data] 有效樣本：{len(data)} 筆")
    counts = Counter(d["label"] for d in data)
    for label, cnt in sorted(counts.items()):
        print(f"  {INTENT_LABELS[GestureIntent(label)]}: {cnt}")

    np.random.seed(42)
    np.random.shuffle(data)
    n = len(data)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test  = data[:n_test]
    val   = data[n_test:n_test + n_val]
    train = data[n_test + n_val:]

    print(f"[Data] 訓練 {len(train)} / 驗證 {len(val)} / 測試 {len(test)}")
    return train, val, test


def make_sampler(train_data: list[dict]) -> WeightedRandomSampler:
    """對少數類別進行過採樣，緩解類別不平衡問題"""
    counts = Counter(d["label"] for d in train_data)
    total = len(train_data)
    weights = [total / (counts[d["label"]] * len(counts)) for d in train_data]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ──────────────────────────────────────────────────────────────────
# 訓練
# ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 使用裝置：{device}")

    if not os.path.exists(DATASET_PATH):
        print(f"[錯誤] 找不到資料集：{DATASET_PATH}")
        print("請先執行：python tools/collect_gesture_data.py --video <影片路徑>")
        return

    train_data, val_data, test_data = load_and_split(DATASET_PATH)
    if len(train_data) < 50:
        print("[警告] 訓練資料太少（<50筆），建議先收集更多資料")

    train_ds = GestureDataset(train_data, augment=args.augment)
    val_ds   = GestureDataset(val_data)
    test_ds  = GestureDataset(test_data)

    sampler = make_sampler(train_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GestureGRU(
        input_size=INPUT_SIZE,
        hidden_size=64,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print(f"\n[Train] 開始訓練 {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # ── 訓練 ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)
        scheduler.step()

        # ── 驗證 ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(y)

        train_acc = train_correct / max(train_total, 1)
        val_acc   = val_correct / max(val_total, 1)
        avg_loss  = train_loss / max(train_total, 1)

        # 每 10 epoch 或最後 1 epoch 印一次
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}"
                  f"  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n[Train] 最佳 Val Acc：{best_val_acc:.3f}  模型儲存至：{MODEL_SAVE_PATH}")

    # ── 測試集評估 ──
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    from sklearn.metrics import classification_report
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    target_names = [INTENT_LABELS[GestureIntent(i)] for i in range(5)]
    print("\n── 測試集分類報告 ──")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手勢分類器訓練腳本")
    parser.add_argument("--epochs",     type=int,   default=60,    help="訓練輪數")
    parser.add_argument("--lr",         type=float, default=5e-4,  help="學習率")
    parser.add_argument("--batch-size", type=int,   default=32,    help="批次大小")
    parser.add_argument("--augment",    action="store_true",        help="開啟資料增強")
    args = parser.parse_args()
    train(args)
