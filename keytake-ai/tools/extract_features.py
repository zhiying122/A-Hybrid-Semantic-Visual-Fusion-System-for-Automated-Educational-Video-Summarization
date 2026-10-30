"""Extract complete per-segment multimodal features for MLP training."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.feature_extractor import extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description="擷取 MLP 訓練用多模態特徵")
    parser.add_argument("--video", required=True, help="輸入影片路徑")
    parser.add_argument(
        "--output",
        help="輸出 JSON 路徑；預設 results/features_<影片名稱>.json",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        parser.error(f"找不到輸入影片：{video_path}")

    output_path = Path(args.output) if args.output else (
        Path("results") / f"features_{video_path.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="keytake-feature-extract-") as work_dir:
        features = extract_features(str(video_path), output_dir=work_dir)
    output_path.write_text(
        json.dumps(features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已輸出 {len(features)} 個片段特徵至：{output_path}")


if __name__ == "__main__":
    main()
