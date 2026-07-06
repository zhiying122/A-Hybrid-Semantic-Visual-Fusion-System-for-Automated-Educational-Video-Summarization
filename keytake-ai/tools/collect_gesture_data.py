"""
手勢訓練資料收集工具
──────────────────────────────────────────────────────────────────
用於為 GestureClassifier 建立訓練資料集。

操作方式：
  1. 把幾部教學影片放進 data/videos/
  2. 執行本工具，它會自動用 MediaPipe 追蹤每部影片的手部軌跡
  3. 對每個偵測到的軌跡片段，你用鍵盤輸入標籤（0~4）
  4. 資料自動存入 data/gesture_dataset.json

標籤對應：
  0 = POINTING  指引
  1 = WRITING   書寫
  2 = EMPHASIS  強調
  3 = TRANSITION 過渡
  4 = IDLE      無手部（通常自動生成，不需手動標註）

執行：
  cd keytake-ai
  python tools/collect_gesture_data.py --video data/videos/lecture_01.mp4
  python tools/collect_gesture_data.py --video data/videos/ --auto  # 批次處理整個資料夾

提示：
  - 每部 60 分鐘影片大約能收集到 500~2000 個有效軌跡片段
  - 目標：每類至少 200 個樣本
  - 使用 --auto 模式時，工具會用規則型分類器預標籤，你只需要確認或更正
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visual.hand_tracker import HandTracker
from src.visual.gesture_classifier import (
    GestureIntent, INTENT_LABELS, rule_based_classify, extract_trajectory_features,
)
from config import VISUAL_SAMPLE_FPS


DATASET_PATH = "data/gesture_dataset.json"
WINDOW_SECONDS = 1.0    # 每個樣本收集 1 秒的軌跡
MIN_COORDS = 5          # 至少要有 5 個座標點才算有效樣本


def load_dataset() -> list[dict]:
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_dataset(data: list[dict]):
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Dataset] 已儲存 {len(data)} 筆資料到 {DATASET_PATH}")


def collect_from_video(
    video_path: str,
    auto_label: bool = False,
    sample_interval_sec: float = 0.5,
) -> list[dict]:
    """
    從單部影片收集手勢軌跡資料。

    每隔 sample_interval_sec 秒提取一段軌跡，
    用規則型分類器預標籤，再讓使用者確認（非 auto 模式）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片：{video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    video_name = os.path.basename(video_path)

    print(f"\n[收集] {video_name}  時長 {duration:.0f}s  FPS={fps:.1f}")
    print("標籤：0=指引  1=書寫  2=強調  3=過渡  4=忽略  Enter=接受預測  q=結束\n")

    tracker = HandTracker(window_size=int(fps * WINDOW_SECONDS))
    samples = []
    frame_idx = 0
    last_sample_time = -sample_interval_sec

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        frame_idx += 1

        # 只在取樣間隔收集
        if current_time - last_sample_time < sample_interval_sec:
            tracker.process_frame(frame)
            continue

        event = tracker.process_frame(frame)
        last_sample_time = current_time

        # 只在有偵測到手的時候收集
        if not event.get("triggered") and event.get("s_visual_raw", 0) < 0.1:
            continue

        coords = tracker.coord_history.copy()
        if len(coords) < MIN_COORDS:
            continue

        # 規則型預測
        rule_result = rule_based_classify(coords)
        predicted_label = int(rule_result.intent)

        if auto_label:
            # 自動標籤模式：直接接受規則預測
            label = predicted_label
        else:
            # 互動模式：顯示軌跡並請使用者確認
            _show_trajectory(frame, coords, rule_result.label, current_time)
            label = _get_label_input(predicted_label)
            if label == -1:  # 使用者選擇結束
                break
            if label == 4:   # 忽略這個樣本
                continue

        # 萃取特徵（記錄原始座標，訓練時再轉換）
        h, w = frame.shape[:2]
        sample = {
            "video": video_name,
            "time": round(current_time, 2),
            "label": label,
            "intent_name": INTENT_LABELS[GestureIntent(label)],
            "coords": [[round(x, 2), round(y, 2)] for x, y in coords[-20:]],
            "frame_w": w,
            "frame_h": h,
        }
        samples.append(sample)

        label_name = INTENT_LABELS[GestureIntent(label)]
        print(f"  t={current_time:.1f}s  標籤: {label_name}  累計: {len(samples)}")

    cap.release()
    cv2.destroyAllWindows()
    return samples


def _show_trajectory(
    frame: np.ndarray,
    coords: list[tuple[float, float]],
    predicted_label: str,
    current_time: float,
):
    """在畫面上畫出軌跡並顯示預測標籤"""
    vis = frame.copy()
    if len(coords) > 1:
        pts = np.array(coords, dtype=np.int32)
        # 畫軌跡線，顏色從藍（舊）到紅（新）
        for i in range(1, len(pts)):
            ratio = i / len(pts)
            color = (int(255 * (1 - ratio)), 50, int(255 * ratio))
            cv2.line(vis, tuple(pts[i-1]), tuple(pts[i]), color, 2)
        # 標記最新位置
        cv2.circle(vis, tuple(pts[-1]), 8, (0, 0, 255), -1)

    cv2.putText(vis, f"t={current_time:.1f}s  預測: {predicted_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, "0=指引 1=書寫 2=強調 3=過渡 Enter=接受 q=結束",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    cv2.imshow("Gesture Label Tool", vis)
    cv2.waitKey(1)


def _get_label_input(predicted: int) -> int:
    """等待使用者輸入標籤，回傳 -1 表示結束"""
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            return -1
        elif key == 13:  # Enter
            return predicted
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            return int(chr(key))


def print_stats(dataset: list[dict]):
    """顯示資料集統計"""
    from collections import Counter
    counts = Counter(d["intent_name"] for d in dataset)
    print("\n── 資料集統計 ──")
    for name, count in sorted(counts.items()):
        print(f"  {name:12s}: {count:4d} 筆")
    print(f"  {'合計':12s}: {len(dataset):4d} 筆")
    short = [k for k, v in counts.items() if v < 200]
    if short:
        print(f"\n⚠ 以下類別樣本不足 200 筆，建議補充：{', '.join(short)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手勢訓練資料收集工具")
    parser.add_argument("--video", type=str, required=True,
                        help="影片路徑或包含影片的資料夾")
    parser.add_argument("--auto", action="store_true",
                        help="自動標籤模式（使用規則型分類器預標籤，不需人工確認）")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="取樣間隔秒數（預設 0.5）")
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"[Dataset] 現有資料：{len(dataset)} 筆")

    # 收集影片路徑
    video_paths = []
    if os.path.isdir(args.video):
        for fname in sorted(os.listdir(args.video)):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_paths.append(os.path.join(args.video, fname))
    else:
        video_paths = [args.video]

    print(f"[Dataset] 準備處理 {len(video_paths)} 部影片")

    for vp in video_paths:
        new_samples = collect_from_video(vp, auto_label=args.auto, sample_interval_sec=args.interval)
        dataset.extend(new_samples)
        save_dataset(dataset)
        print(f"  ➜ 新增 {len(new_samples)} 筆")

    print_stats(dataset)
