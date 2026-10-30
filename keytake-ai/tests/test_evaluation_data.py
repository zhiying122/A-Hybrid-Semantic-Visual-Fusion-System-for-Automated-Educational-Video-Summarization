import json

from src.fusion.evaluation_data import load_annotated_video


def test_load_annotated_video_clips_and_offsets_eval_range(tmp_path):
    features_path = tmp_path / "features_demo.json"
    gt_path = tmp_path / "gt_demo.json"
    features_path.write_text(
        json.dumps(
            [
                {"start": 8, "end": 12, "s_text": 0.4},
                {"start": 14, "end": 22, "s_text": 0.8},
                {"start": 22, "end": 24, "s_text": 0.2},
            ]
        ),
        encoding="utf-8",
    )
    gt_path.write_text(
        json.dumps(
            {
                "eval_range_sec": [10, 20],
                "ground_truth": [
                    {"start_sec": 8, "end_sec": 12},
                    {"start_sec": 15, "end_sec": 25},
                ],
            }
        ),
        encoding="utf-8",
    )

    video = load_annotated_video(str(features_path), str(gt_path))

    assert video["video_id"] == "demo"
    assert video["duration"] == 10
    assert video["segments"] == [
        {"start": 0, "end": 2, "s_text": 0.4},
        {"start": 4, "end": 10, "s_text": 0.8},
    ]
    assert video["ground_truth"] == [(0, 2), (5, 10)]
