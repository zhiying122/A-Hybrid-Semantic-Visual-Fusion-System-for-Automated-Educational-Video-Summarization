"""
逐字稿匯出工具
將影片透過 Whisper 轉錄後，輸出帶時間戳記的 JSON 逐字稿
供 annotator.py 標註使用，以及 run_grid_search.py 載入語意分數

用法：
  cd keytake-ai
  # 單一影片
  python tools/export_transcript.py --video data/videos/lecture_01.mp4 --output data/annotations/

  # 批次處理整個資料夾
  python tools/export_transcript.py --videos data/videos/ --output data/annotations/
"""

import json
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.preprocessor import preprocess
from src.semantic.transcriber import transcribe
from src.semantic.scorer import SemanticScorer
from data.prompt_corpus.corpus import PROMPT_CORPUS


def export_one(video_path: str, output_dir: str) -> dict:
    """
    處理單一影片：前處理 → 轉錄 → 語意評分 → 輸出 JSON

    輸出兩個檔案：
    - transcript_{vid_id}.json  : 原始逐字稿（供 annotator.py 使用）
    - segments_{vid_id}.json    : 帶語意分數的片段（供 run_grid_search.py 使用）
    """
    os.makedirs(output_dir, exist_ok=True)
    vid_id = os.path.splitext(os.path.basename(video_path))[0]

    print(f"\n處理：{video_path}")

    # 步驟一：影音前處理
    tmp_dir = os.path.join(output_dir, f"_tmp_{vid_id}")
    try:
        paths = preprocess(video_path, tmp_dir)
    except Exception as e:
        print(f"  ✗ 前處理失敗：{e}")
        return {}

    # 步驟二：Whisper 轉錄
    try:
        segments = transcribe(paths["audio"])
    except Exception as e:
        print(f"  ✗ 轉錄失敗：{e}")
        return {}

    # 輸出原始逐字稿（供標註者使用）
    transcript_path = os.path.join(output_dir, f"transcript_{vid_id}.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 逐字稿：{transcript_path}（{len(segments)} 段）")

    # 步驟三：語意評分
    try:
        scorer = SemanticScorer(PROMPT_CORPUS)
        scored_segments = scorer.score(segments)
    except Exception as e:
        print(f"  ✗ 語意評分失敗：{e}")
        scored_segments = segments

    # 輸出帶分數的片段（供 Grid Search 使用）
    segments_path = os.path.join(output_dir, f"segments_{vid_id}.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(scored_segments, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 語意分數：{segments_path}")

    # 清理暫存
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"vid_id": vid_id, "segments": len(segments)}


def main():
    parser = argparse.ArgumentParser(description="影片逐字稿匯出工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="單一影片路徑")
    group.add_argument("--videos", help="影片資料夾路徑（批次處理）")
    parser.add_argument("--output", default="data/annotations", help="輸出資料夾")
    args = parser.parse_args()

    if args.video:
        result = export_one(args.video, args.output)
        if result:
            print(f"\n完成：{result['vid_id']}，共 {result['segments']} 個片段")
            print(f"下一步：python tools/annotator.py annotate "
                  f"{args.output}/transcript_{result['vid_id']}.json --annotator A")
    else:
        video_dir = args.videos
        if not os.path.exists(video_dir):
            print(f"找不到資料夾：{video_dir}")
            sys.exit(1)

        video_files = sorted(
            f for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        )
        if not video_files:
            print(f"資料夾內沒有影片：{video_dir}")
            sys.exit(1)

        print(f"找到 {len(video_files)} 部影片，開始批次處理...")
        results = []
        for vf in video_files:
            r = export_one(os.path.join(video_dir, vf), args.output)
            if r:
                results.append(r)

        print(f"\n批次完成：{len(results)}/{len(video_files)} 部成功")
        print(f"輸出目錄：{args.output}")
        print("\n下一步：")
        print("  1. python tools/annotator.py annotate <transcript_*.json> --annotator A")
        print("  2. python tools/annotator.py annotate <transcript_*.json> --annotator B")
        print("  3. python tools/annotator.py merge annotations/annotator_A.json annotations/annotator_B.json")
        print("  4. python tools/run_grid_search.py --annotations data/annotations/")


if __name__ == "__main__":
    main()
