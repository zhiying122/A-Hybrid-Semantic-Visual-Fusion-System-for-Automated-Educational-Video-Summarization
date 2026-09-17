"""
fetch_archive_videos.py — 從 Internet Archive 下載公開教學影片
────────────────────────────────────────────────────────────────
Internet Archive（archive.org）收錄大量公眾領域/CC 授權的講課影片，
且提供直接 HTTP 下載，不需要 JS runtime，適合離線驗證本系統。

用法：
  # 自動搜尋 lecture 類影片並下載前 N 部（每部裁切前 clip-seconds 秒）
  python tools/fetch_archive_videos.py --count 3 --clip-seconds 90

  # 指定 archive.org item identifier
  python tools/fetch_archive_videos.py --identifiers <id1> <id2>

下載結果存放於 data/videos/downloaded/
"""

import argparse
import json
import os
import subprocess
import urllib.request
import urllib.parse

UA = {"User-Agent": "Mozilla/5.0 (KeyTake-AI test fetcher)"}
VIDEO_EXTS = (".mp4", ".m4v", ".mpeg", ".mpg", ".ogv")


def _get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout)


def search_lectures(count: int) -> list[str]:
    """搜尋 archive.org 上的講課類影片，回傳 identifier 清單。"""
    q = urllib.parse.quote("subject:lecture AND mediatype:movies AND format:(MPEG4)")
    url = (
        f"https://archive.org/advancedsearch.php?q={q}"
        f"&fl[]=identifier&rows={count * 3}&output=json"
    )
    try:
        data = json.load(_get(url))
        ids = [d["identifier"] for d in data["response"]["docs"]]
        return ids[: count * 3]
    except Exception as e:
        print(f"[search] 搜尋失敗：{e}")
        return []


def find_video_file(identifier: str) -> tuple[str, str] | None:
    """查詢 item 的檔案清單，找出最小的可下載影片檔。回傳 (檔名, 直連URL)。"""
    meta_url = f"https://archive.org/metadata/{identifier}"
    try:
        meta = json.load(_get(meta_url))
    except Exception as e:
        print(f"  [meta] {identifier} 讀取失敗：{e}")
        return None

    files = meta.get("files", [])
    candidates = []
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith(VIDEO_EXTS):
            size = int(f.get("size", 0) or 0)
            candidates.append((size, name))
    if not candidates:
        return None
    # 選最小的檔案以節省下載時間
    candidates.sort()
    _, name = candidates[0]
    dl_url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}"
    return name, dl_url


def download_file(url: str, out_path: str) -> bool:
    print(f"  下載中：{url}")
    try:
        with _get(url, timeout=300) as resp, open(out_path, "wb") as fout:
            chunk = resp.read(1024 * 256)
            total = 0
            while chunk:
                fout.write(chunk)
                total += len(chunk)
                chunk = resp.read(1024 * 256)
        return os.path.getsize(out_path) > 0
    except Exception as e:
        print(f"  ✗ 下載失敗：{e}")
        return False


def clip_local(video_path: str, clip_seconds: int):
    base, ext = os.path.splitext(video_path)
    clipped = f"{base}_clip.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-t", str(clip_seconds),
           "-c", "copy", clipped]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(clipped) and os.path.getsize(clipped) > 0:
            os.remove(video_path)
            print(f"  ✂ 已裁切為前 {clip_seconds} 秒 → {clipped}")
            return clipped
    except Exception as e:
        # copy 失敗時嘗試重新編碼
        try:
            cmd2 = ["ffmpeg", "-y", "-i", video_path, "-t", str(clip_seconds),
                    "-c:v", "libx264", "-c:a", "aac", clipped]
            subprocess.run(cmd2, check=True, capture_output=True)
            if os.path.exists(clipped) and os.path.getsize(clipped) > 0:
                os.remove(video_path)
                print(f"  ✂ 已重新編碼裁切為前 {clip_seconds} 秒 → {clipped}")
                return clipped
        except Exception as e2:
            print(f"  ⚠ 裁切失敗（保留完整檔）：{e2}")
    return video_path


def main():
    parser = argparse.ArgumentParser(description="從 Internet Archive 下載教學影片")
    parser.add_argument("--identifiers", nargs="*", help="指定 archive.org identifier")
    parser.add_argument("--count", type=int, default=3, help="自動搜尋下載幾部")
    parser.add_argument("--clip-seconds", type=int, default=90,
                        help="每部裁切前 N 秒（0 = 不裁切）")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "videos", "downloaded"),
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ids = args.identifiers or search_lectures(args.count)
    if not ids:
        print("[fetch_archive_videos] 找不到可下載的影片")
        return

    print(f"[fetch_archive_videos] 候選 identifier：{ids}")
    downloaded = []
    for identifier in ids:
        if len(downloaded) >= args.count:
            break
        print(f"\n[item] {identifier}")
        found = find_video_file(identifier)
        if not found:
            print("  （無可用影片檔，略過）")
            continue
        name, url = found
        safe_name = identifier[:40].replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_name}.mp4")
        if download_file(url, out_path):
            final = out_path
            if args.clip_seconds:
                final = clip_local(out_path, args.clip_seconds)
            size_mb = os.path.getsize(final) / 1024 / 1024
            print(f"  ✓ 完成：{final}  ({size_mb:.1f} MB)")
            downloaded.append(final)

    print(f"\n{'='*50}")
    print(f"完成：成功下載 {len(downloaded)} 部影片")
    for p in downloaded:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
