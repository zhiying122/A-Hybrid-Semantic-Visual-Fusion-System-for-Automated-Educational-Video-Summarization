"""列印逐字稿供人工快速檢視（唯讀，不修改任何檔案）。

用法：
  python tools/print_transcript.py data/annotations/transcript_<vid>.json
"""
import json
import sys


def fmt(sec):
    return f"{int(sec)//60:02d}:{int(sec)%60:02d}"


def main():
    if len(sys.argv) < 2:
        print("用法：python tools/print_transcript.py <transcript.json>")
        return
    with open(sys.argv[1], encoding="utf-8") as f:
        segs = json.load(f)
    print(f"共 {len(segs)} 段\n")
    for i, s in enumerate(segs):
        text = s.get("text", "").strip()
        print(f"{i+1:>3} | {fmt(s['start'])}-{fmt(s['end'])} | {text}")


if __name__ == "__main__":
    main()
