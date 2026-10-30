import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.fusion.mlp_fusion as mlp_fusion
from src.fusion.mlp_fusion import (
    MLP_FEATURE_NAMES,
    MLP_FEATURE_VERSION,
    MLP_INPUT_DIM,
    MLPFusionTrainer,
    TEACHING_STAGES,
    _split_group_indices,
)


def test_missing_visual_reliability_is_encoded_as_unavailable():
    trainer = MLPFusionTrainer.__new__(MLPFusionTrainer)

    features = trainer._build_features(
        {
            "s_text": 0.6,
            "s_visual": 0.9,
            "s_prosodic": 0.5,
            "prosodic_available": True,
        }
    )

    assert features.shape == (MLP_INPUT_DIM,)
    assert features[2] == 0.0
    assert np.isfinite(features).all()


def test_missing_prosodic_score_is_not_silently_encoded_as_neutral():
    trainer = MLPFusionTrainer.__new__(MLPFusionTrainer)
    with pytest.raises(ValueError, match="s_prosodic 缺失"):
        trainer._build_features(
            {"s_text": 0.6, "prosodic_available": True}
        )


def test_unavailable_prosodic_score_is_masked_in_feature_vector():
    trainer = MLPFusionTrainer.__new__(MLPFusionTrainer)
    features = trainer._build_features(
        {
            "s_text": 0.6,
            "s_prosodic": 0.5,
            "prosodic_available": False,
        }
    )

    assert features[4] == 0.5
    assert features[5] == 0.0
    assert features.shape == (12,)


def test_unknown_teaching_stage_is_rejected():
    trainer = MLPFusionTrainer.__new__(MLPFusionTrainer)
    with pytest.raises(ValueError, match="不支援的 teaching_stage"):
        trainer._build_features(
            {
                "s_text": 0.6,
                "s_prosodic": 0.5,
                "prosodic_available": True,
                "teaching_stage": "semantic",
            }
        )


def test_label_overlap_default_matches_ground_truth_coverage():
    trainer = MLPFusionTrainer.__new__(MLPFusionTrainer)
    trainer.label_overlap_basis = "ground_truth"
    segment = {"start": 0.0, "end": 20.0}
    ground_truth = [(0.0, 2.0)]

    assert trainer._compute_label(segment, ground_truth) == 1.0

    trainer.label_overlap_basis = "segment"
    assert trainer._compute_label(segment, ground_truth) == 0.0


@pytest.mark.skipif(not mlp_fusion.TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_checkpoint_persists_and_validates_feature_schema(tmp_path):
    torch = mlp_fusion.torch
    path = tmp_path / "model.pth"
    trainer = MLPFusionTrainer(model_path=str(path))
    trainer._save_model()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert checkpoint["input_dim"] == MLP_INPUT_DIM
    assert checkpoint["feature_version"] == MLP_FEATURE_VERSION
    assert checkpoint["feature_names"] == list(MLP_FEATURE_NAMES)
    assert checkpoint["teaching_stages"] == TEACHING_STAGES
    MLPFusionTrainer(model_path=str(path))._load_model()


@pytest.mark.skipif(not mlp_fusion.TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_incompatible_checkpoint_schema_is_rejected(tmp_path):
    torch = mlp_fusion.torch
    path = tmp_path / "old-model.pth"
    trainer = MLPFusionTrainer(model_path=str(path))
    trainer._save_model()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["feature_version"] = "old-11d"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="feature_version 不相容"):
        MLPFusionTrainer(model_path=str(path))._load_model()


def test_mlp_training_requires_video_groups():
    trainer = MLPFusionTrainer.__new__(MLPFusionTrainer)

    with pytest.raises(ValueError, match="group"):
        trainer.train(
            segments=[{"start": 0.0, "end": 1.0}],
            ground_truth=[(0.0, 1.0)],
        )


def test_training_validation_groups_are_disjoint():
    groups = ["video-a", "video-a", "video-b", "video-c", "video-c"]

    train_idx, val_idx = _split_group_indices(groups, seed=7)
    train_groups = {groups[index] for index in train_idx}
    val_groups = {groups[index] for index in val_idx}

    assert train_groups.isdisjoint(val_groups)
    assert train_idx.size + val_idx.size == len(groups)
