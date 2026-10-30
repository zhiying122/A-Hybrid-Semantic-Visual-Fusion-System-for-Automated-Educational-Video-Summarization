import json
import sys

import pytest

from tools import train_mlp_fusion
from src.fusion.mlp_fusion import MLP_FEATURE_VERSION


def _feature_segment():
    return {
        "start": 0.0,
        "end": 1.0,
        "s_text": 0.7,
        "s_visual": 0.6,
        "visual_reliability": 0.8,
        "gesture_intent_score": 0.3,
        "s_prosodic": 0.5,
        "prosodic_available": True,
        "teaching_stage": "unclassified",
        "feature_version": MLP_FEATURE_VERSION,
    }


def test_load_feature_file_rejects_old_schema_version(tmp_path):
    feature = _feature_segment()
    feature["feature_version"] = "old-11d"
    path = tmp_path / "features_old.json"
    path.write_text(json.dumps([feature]), encoding="utf-8")

    with pytest.raises(ValueError, match="特徵版本不相容"):
        train_mlp_fusion._load_feature_file(path)


def test_independent_test_video_is_excluded_from_training(
    tmp_path,
    monkeypatch,
):
    feature_dir = tmp_path / "features"
    annotation_dir = tmp_path / "annotations"
    feature_dir.mkdir()
    annotation_dir.mkdir()

    for video_id in ("train_a", "train_b", "held_out"):
        feature_path = feature_dir / f"features_{video_id}.json"
        feature_path.write_text(
            json.dumps([_feature_segment()]),
            encoding="utf-8",
        )
        gt_path = annotation_dir / f"gt_{video_id}.json"
        annotation = {
            "ground_truth": [{"start_sec": 0.0, "end_sec": 1.0}],
        }
        if video_id == "train_a":
            annotation["eval_range_sec"] = [0.0, 0.75]
        elif video_id == "held_out":
            annotation["eval_range_sec"] = [0.0, 0.5]
        gt_path.write_text(
            json.dumps(annotation),
            encoding="utf-8",
        )

    observed_groups = []
    observed_train_segments = []
    observed_test_segments = []

    class FakeTrainer:
        def __init__(self, model_path, label_overlap_basis):
            assert model_path
            assert label_overlap_basis == "ground_truth"

        def train(self, segments, ground_truth, **kwargs):
            assert segments
            assert ground_truth
            observed_train_segments.extend(segments)
            observed_groups.extend(kwargs["groups"])
            return {
                "best_epoch": 1,
                "best_val_loss": 0.1,
                "positive_ratio": 1.0,
            }

        def predict(self, segments):
            observed_test_segments.extend(segments)
            return [1.0] * len(segments)

    monkeypatch.setattr(train_mlp_fusion, "MLPFusionTrainer", FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_mlp_fusion.py",
            "--features",
            str(feature_dir),
            "--annotations",
            str(annotation_dir),
            "--test-features",
            str(feature_dir / "features_held_out.json"),
            "--test-annotations",
            str(annotation_dir / "gt_held_out.json"),
            "--output",
            str(tmp_path / "model.pth"),
        ],
    )

    train_mlp_fusion.main()

    assert observed_groups == ["train_a", "train_b"]
    assert observed_train_segments[0]["end"] == 0.75
    assert observed_test_segments[0]["end"] == 0.5
