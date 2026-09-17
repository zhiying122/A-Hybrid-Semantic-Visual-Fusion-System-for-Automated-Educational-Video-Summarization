"""
fetch_test_videos.py — 自動下載公開教學影片供 pipeline 驗證使用
────────────────────────────────────────────────────────────────
使用 yt-dlp（Python 模組）抓取數部公開的教學/講課影片，
下載後可直接餵給 main.py 做端到端驗證。

用法：
  # 下載預設清單（每部只抓前 N 秒以節省時間與硬碟）
  python tools/fetch_test_videos.py --clip-seconds 120

  # 自訂影片清單
  python tools/fetch_test_videos.py --urls https://... https://... --clip-seconds 90

  # 下載完整影片（不裁切）
  python tools/fetch_test_videos.py --full

下載結果存放於 data/videos/downloaded/
"""

import argparse
import os
import sys

# 預設公開教學影片清單（可自行增修）
# 選擇「單人講者 + 板書/投影片」型態的教學影片，最貼近本系統設計場域。
DEFAULT_VIDEOS = [
    # (輸出檔名, YouTube 網址)
    ("khan_algebra", "https://www.youtube.com/watch?v=NybHckSEQBI"),   # Khan Academy 代數
    ("mit_calculus", "https://www.youtube.com/watch?v=7K1sB05pE0A"),  # MIT 微積分講座
    ("physics_lecture", "https://www.youtube.com/watch?v=ZM8ECpBuQYE"),  # 物理講課
]


def download_one(name: str, url: str, out_dir: str, clip_seconds: int | None):
    """下載單一影片，可選擇只抓前 clip_seconds 秒。"""
    import yt_dlp

    out_path = os.path.join(out_dir, f"{name}.mp4")
    # 優先使用「漸進式」單一檔案格式（video+audio 已合併），
    # 這類格式可直接 HTTP 下載，避開需要 ffmpeg 串流合併時遇到的 403。
    ydl_opts = {
        "format": "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 3,
        "fragment_retries": 3,
    }

    print(f"\n[下載] {name}  ←  {url}")
    downloaded_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        base = os.path.splitext(out_path)[0]
        for ext in (".mp4", ".mkv", ".webm"):
            cand = base + ext
            if os.path.exists(cand):
                downloaded_path = cand
                break
    except Exception as e:
        print(f"  ✗ 下載失敗：{e}")
        return None

    if not downloaded_path:
        print(f"  ✗ 下載完成但找不到輸出檔：{out_path}")
        return None

    # 下載完成後，若指定了 clip_seconds，用本地 ffmpeg 裁切前 N 秒
    if clip_seconds:
        clipped = _clip_local(downloaded_path, clip_seconds)
        if clipped:
            downloaded_path = clipped

    size_mb = os.path.getsize(downloaded_path) / 1024 / 1024
    print(f"  ✓ 完成：{downloaded_path}  ({size_mb:.1f} MB)")
    return downloaded_path


def _clip_local(video_path: str, clip_seconds: int):
    """用本地 ffmpeg 將影片裁切為前 clip_seconds 秒。"""
    import subprocess

    base, ext = os.path.splitext(video_path)
    clipped = f"{base}_clip{ext}"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(clip_seconds),
        "-c", "copy", clipped,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(clipped) and os.path.getsize(clipped) > 0:
            os.remove(video_path)
            os.rename(clipped, video_path)
            print(f"  ✂ 已裁切為前 {clip_seconds} 秒")
            return video_path
    except Exception as e:
        print(f"  ⚠ 裁切失敗（保留完整檔）：{e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="下載公開教學影片供驗證")
    parser.add_argument("--urls", nargs="*", help="自訂影片網址清單")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "videos", "downloaded"),
        help="輸出目錄",
    )
    parser.add_argument("--clip-seconds", type=int, default=120,
                        help="每部影片只抓前 N 秒（預設 120；設 0 或加 --full 表示完整下載）")
    parser.add_argument("--full", action="store_true", help="下載完整影片（忽略 --clip-seconds）")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clip_seconds = None if args.full or args.clip_seconds == 0 else args.clip_seconds

    if args.urls:
        jobs = [(f"custom_{i+1:02d}", u) for i, u in enumerate(args.urls)]
    else:
        jobs = DEFAULT_VIDEOS

    print(f"[fetch_test_videos] 輸出目錄：{out_dir}")
    print(f"[fetch_test_videos] 裁切長度：{'完整' if clip_seconds is None else str(clip_seconds) + ' 秒'}")
    print(f"[fetch_test_videos] 影片數量：{len(jobs)}")

    downloaded = []
    for name, url in jobs:
        path = download_one(name, url, out_dir, clip_seconds)
        if path:
            downloaded.append(path)

    print(f"\n{'='*50}")
    print(f"完成：成功下載 {len(downloaded)}/{len(jobs)} 部影片")
    for p in downloaded:
        print(f"  - {p}")

    if not downloaded:
        print("\n[提示] 全部下載失敗，可能是網路限制或影片不可用。")
        print("       可用 --urls 指定其他公開影片，或改用 data/videos/test_10min.mp4 進行驗證。")
        sys.exit(1)


if __name__ == "__main__":
    main()
