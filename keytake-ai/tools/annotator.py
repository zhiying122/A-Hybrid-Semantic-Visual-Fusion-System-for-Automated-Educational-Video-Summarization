"""
Ground Truth 標註工具（CLI 介面）
對應計畫書 4.4 黃金標準建立：雙盲專家評級
- 逐段顯示 Whisper 轉錄文字與時間戳記
- 標註者以 Likert 1~5 分評分
- 兩位標註者都給 4 分以上才列入 Ground Truth
- 自動計算 Cohen's Kappa 確認標註一致性（標準 > 0.6）
- 結果存為 JSON 供後續 Grid Search 使用

用法：
  python tools/annotator.py <transcript.json> --annotator A
  python tools/annotator.py <transcript.json> --annotator B
  python tools/annotator.py --merge annotator_A.json annotator_B.json
"""

import json
import argparse
import os
from datetime import datetime


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def annotate(transcript_path: str, annotator_id: str, output_path: str):
    """逐段讓標註者評分"""
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"\n{'='*55}")
    print(f"  KeyTake AI 標註工具  |  標註者：{annotator_id}")
    print(f"  共 {len(segments)} 個片段，請以 1~5 分評估重要性")
    print(f"  1=不重要  3=普通  5=非常重要  s=跳過  q=儲存並離開")
    print(f"{'='*55}\n")

    results = []
    for i, seg in enumerate(segments):
        print(f"[{i+1}/{len(segments)}]  {fmt_time(seg['start'])} ~ {fmt_time(seg['end'])}")
        print(f"  {seg['text']}\n")

        while True:
            raw = input("  評分 (1-5 / s / q): ").strip().lower()
            if raw == "q":
                _save(results, output_path, annotator_id)
                print(f"\n已儲存至 {output_path}，共標註 {len(results)} 段")
                return
            if raw == "s":
                break
            if raw in {"1", "2", "3", "4", "5"}:
                results.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "score": int(raw)
                })
                break
            print("  請輸入 1~5、s 或 q")
        print()

    _save(results, output_path, annotator_id)
    print(f"\n標註完成！結果儲存至 {output_path}")


def _save(results: list, path: str, annotator_id: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "annotator": annotator_id,
            "timestamp": datetime.now().isoformat(),
            "segments": results
        }, f, ensure_ascii=False, indent=2)


def merge(path_a: str, path_b: str, output_path: str = "ground_truth.json"):
    """
    合併兩位標註者的結果：
    - 兩人都給 >= 4 分 → 列入 Ground Truth
    - 計算 Cohen's Kappa 確認一致性
    """
    with open(path_a, encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        data_b = json.load(f)

    segs_a = {(s["start"], s["end"]): s["score"] for s in data_a["segments"]}
    segs_b = {(s["start"], s["end"]): s["score"] for s in data_b["segments"]}

    common_keys = set(segs_a) & set(segs_b)
    if not common_keys:
        print("警告：兩份標註沒有重疊片段，請確認使用相同的逐字稿")
        return

    # Cohen's Kappa（二元化：>= 4 為正例）
    kappa = _cohens_kappa(
        [1 if segs_a[k] >= 4 else 0 for k in common_keys],
        [1 if segs_b[k] >= 4 else 0 for k in common_keys]
    )
    print(f"\nCohen's Kappa = {kappa:.3f}  (標準 > 0.6)")
    if kappa < 0.6:
        print("⚠️  一致性不足，建議由指導教授協助判定歧義片段")

    # 兩人都 >= 4 才列入 GT
    ground_truth = []
    for k in common_keys:
        if segs_a[k] >= 4 and segs_b[k] >= 4:
            ground_truth.append({"start": k[0], "end": k[1]})

    ground_truth.sort(key=lambda x: x["start"])
    result = {
        "cohens_kappa": round(kappa, 4),
        "total_annotated": len(common_keys),
        "ground_truth_count": len(ground_truth),
        "ground_truth": ground_truth
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Ground Truth 片段數：{len(ground_truth)} / {len(common_keys)}")
    print(f"結果儲存至 {output_path}")


def _cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """計算二元標籤的 Cohen's Kappa"""
    n = len(rater_a)
    if n == 0:
        return 0.0
    agree = sum(a == b for a, b in zip(rater_a, rater_b))
    p_o = agree / n
    p_a = (sum(rater_a) / n) * (sum(rater_b) / n)
    p_b = ((n - sum(rater_a)) / n) * ((n - sum(rater_b)) / n)
    p_e = p_a + p_b
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KeyTake AI 標註工具")
    sub = parser.add_subparsers(dest="cmd")

    # 標註模式
    ann = sub.add_parser("annotate", help="對逐字稿進行評分")
    ann.add_argument("transcript", help="Whisper 輸出的 JSON 逐字稿路徑")
    ann.add_argument("--annotator", required=True, help="標註者 ID（如 A 或 B）")
    ann.add_argument("--output", default=None, help="輸出路徑（預設自動命名）")

    # 合併模式
    mrg = sub.add_parser("merge", help="合併兩位標註者結果並計算 Kappa")
    mrg.add_argument("file_a", help="標註者 A 的 JSON 檔")
    mrg.add_argument("file_b", help="標註者 B 的 JSON 檔")
    mrg.add_argument("--output", default="ground_truth.json")

    args = parser.parse_args()

    if args.cmd == "annotate":
        out = args.output or f"annotations/annotator_{args.annotator}.json"
        annotate(args.transcript, args.annotator, out)
    elif args.cmd == "merge":
        merge(args.file_a, args.file_b, args.output)
    else:
        parser.print_help()
