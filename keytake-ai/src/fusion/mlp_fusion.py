"""
MLP 融合模組

透過監督式學習取代靜態加權，自 Ground Truth 學習多模態特徵的最佳組合權重。
支援優雅降級 (Graceful Degradation)：若 PyTorch 環境或模型檔不存在，
將自動退回 adaptive_fusion.fuse_scores。
"""

import os
import logging
import numpy as np
from typing import List, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from config import MLP_FUSION_MODEL_PATH, MLP_FUSION_EPOCHS, MLP_FUSION_LR
except ImportError:
    MLP_FUSION_MODEL_PATH = "weights/mlp_fusion_12d.pth"
    MLP_FUSION_EPOCHS = 100
    MLP_FUSION_LR = 0.001

logger = logging.getLogger(__name__)

TEACHING_STAGES = ["definition", "derivation", "example", "summary", "transition", "qa"]
_STAGE_TO_IDX = {stage: i for i, stage in enumerate(TEACHING_STAGES)}
MLP_FEATURE_NAMES = (
    "s_text",
    "s_visual",
    "visual_reliability",
    "gesture_intent_score",
    "s_prosodic",
    "prosodic_available",
    *(f"teaching_stage:{stage}" for stage in TEACHING_STAGES),
)
# 12 維 = 5 個連續分數 + 韻律可用遮罩 + 6 維教學階段 one-hot。
MLP_FEATURE_VERSION = "keytake-multimodal-12d-v3"
MLP_INPUT_DIM = len(MLP_FEATURE_NAMES)


def _split_group_indices(
    groups: List[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.array(sorted(set(groups)))
    if len(unique_groups) < 2:
        raise ValueError("至少需要兩部影片，才能進行以影片分組的訓練/驗證切分")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    val_group_count = max(1, int(round(len(unique_groups) * 0.2)))
    val_group_count = min(val_group_count, len(unique_groups) - 1)
    val_groups = set(unique_groups[:val_group_count])
    val_mask = np.array([group in val_groups for group in groups])
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


if TORCH_AVAILABLE:
    class MLPFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(MLP_INPUT_DIM, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
else:
    MLPFusionModel = None  # type: ignore[assignment]


class MLPFusionTrainer:
    def __init__(
        self,
        model_path: str = MLP_FUSION_MODEL_PATH,
        label_overlap_basis: str = "ground_truth",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安裝，無法使用 MLPFusionTrainer")
        if label_overlap_basis not in {"ground_truth", "segment"}:
            raise ValueError(
                "label_overlap_basis 必須是 ground_truth 或 segment"
            )
        self.model_path = model_path
        self.label_overlap_basis = label_overlap_basis
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPFusionModel().to(self.device)

    def train(
        self,
        segments: List[Dict],
        ground_truth: List[Tuple[float, float]],
        epochs: int = MLP_FUSION_EPOCHS,
        lr: float = MLP_FUSION_LR,
        seed: int = 42,
        groups: List[str] | None = None,
    ) -> dict:
        if not segments:
            raise ValueError("訓練片段不可為空")
        if not ground_truth:
            raise ValueError("Ground Truth 不可為空")
        if groups is None:
            raise ValueError("必須提供影片 group，避免相鄰片段洩漏至驗證集")
        if len(groups) != len(segments):
            raise ValueError("groups 數量必須與訓練片段數一致")
        if any(not isinstance(group, str) or not group.strip() for group in groups):
            raise ValueError("每個影片 group 都必須是非空字串")
        if epochs <= 0 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("epochs 與 lr 必須是正數")
        for start, end in ground_truth:
            if (
                not np.isfinite(start)
                or not np.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise ValueError("Ground Truth 時間範圍無效")

        torch.manual_seed(seed)

        features = np.array([self._build_features(seg) for seg in segments], dtype=np.float32)
        labels = np.array([self._compute_label(seg, ground_truth) for seg in segments], dtype=np.float32)

        n_total = len(features)
        # 以影片 ID 切分而非隨機切片段，避免同一影片的相鄰片段洩漏至驗證集。
        train_idx, val_idx = _split_group_indices(groups, seed)

        n_train = len(train_idx)
        X_train = torch.tensor(features[train_idx], device=self.device)
        y_train = torch.tensor(labels[train_idx], device=self.device).unsqueeze(1)
        X_val = torch.tensor(features[val_idx], device=self.device) if len(val_idx) > 0 else None
        y_val = torch.tensor(labels[val_idx], device=self.device).unsqueeze(1) if len(val_idx) > 0 else None

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        # Sigmoid 輸出是逐片段二元標籤分數；BCELoss 與 Adam 僅按訓練標籤學習。
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_epoch = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= n_train
            train_losses.append(epoch_loss)

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(X_val), y_val).item()
            else:
                # 無驗證集時以訓練損失代替，early stopping 退化為最後 epoch
                val_loss = epoch_loss
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_model()

            if (epoch + 1) % 20 == 0:
                logger.info(f"[MLPFusion] Epoch {epoch+1}/{epochs} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        logger.info(f"[MLPFusion] 訓練完成：best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
        return {
            "total_samples": n_total,
            "train_samples": n_train,
            "val_samples": len(val_idx),
            "train_groups": sorted(set(groups[index] for index in train_idx)),
            "val_groups": sorted(set(groups[index] for index in val_idx)),
            "label_overlap_basis": self.label_overlap_basis,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "positive_ratio": float(labels.sum() / len(labels)) if len(labels) > 0 else 0.0,
        }

    def predict(self, segments: List[Dict]) -> List[float]:
        if not segments:
            return []
        self._load_model()
        self.model.eval()

        features = np.array([self._build_features(seg) for seg in segments], dtype=np.float32)
        X = torch.tensor(features, device=self.device)

        with torch.no_grad():
            scores = self.model(X).squeeze(1).cpu().numpy()
        return scores.tolist()

    def _build_features(self, seg: Dict) -> np.ndarray:
        """Build the versioned feature vector in MLP_FEATURE_NAMES order."""
        feature = np.zeros(MLP_INPUT_DIM, dtype=np.float32)
        for index, key, default in (
            (0, "s_text", 0.0),
            (1, "s_visual", 0.0),
            (2, "visual_reliability", 0.0),
            (3, "gesture_intent_score", 0.0),
        ):
            value = float(seg.get(key, default))
            if not np.isfinite(value):
                raise ValueError(f"MLP 特徵 {key} 必須是有限數值")
            feature[index] = float(np.clip(value, 0.0, 1.0))

        if "prosodic_available" not in seg:
            raise ValueError("MLP 特徵 prosodic_available 缺失")
        prosodic_available = seg["prosodic_available"]
        if not isinstance(prosodic_available, (bool, np.bool_)):
            raise ValueError("MLP 特徵 prosodic_available 必須是布林值")
        if prosodic_available and "s_prosodic" not in seg:
            raise ValueError("MLP 特徵 s_prosodic 缺失，不能以中間分數代替")
        prosodic = float(seg.get("s_prosodic", 0.5))
        if not np.isfinite(prosodic):
            raise ValueError("MLP 特徵 s_prosodic 必須是有限數值")
        feature[4] = float(np.clip(prosodic, 0.0, 1.0))
        feature[5] = float(prosodic_available)

        stage = seg.get("teaching_stage", "unclassified")
        if stage not in (*TEACHING_STAGES, "unclassified"):
            raise ValueError(f"不支援的 teaching_stage：{stage!r}")
        if stage in _STAGE_TO_IDX:
            feature[6 + _STAGE_TO_IDX[stage]] = 1.0

        return feature

    def _compute_label(self, seg: Dict, ground_truth: List[Tuple[float, float]]) -> float:
        seg_start, seg_end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if (
            not np.isfinite(seg_start)
            or not np.isfinite(seg_end)
            or seg_start < 0
        ):
            raise ValueError("訓練片段時間必須是非負且有限數值")
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            raise ValueError("訓練片段時間範圍無效")

        for gt_start, gt_end in ground_truth:
            gt_start, gt_end = float(gt_start), float(gt_end)
            if (
                not np.isfinite(gt_start)
                or not np.isfinite(gt_end)
                or gt_start < 0
                or gt_end <= gt_start
            ):
                raise ValueError("Ground Truth 時間範圍無效")
            overlap = min(seg_end, gt_end) - max(seg_start, gt_start)
            denominator = (
                gt_end - gt_start
                if self.label_overlap_basis == "ground_truth"
                else seg_len
            )
            if max(0.0, overlap) / denominator >= 0.5:
                return 1.0
        return 0.0

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": MLP_INPUT_DIM,
                "feature_version": MLP_FEATURE_VERSION,
                "feature_names": list(MLP_FEATURE_NAMES),
                "teaching_stages": list(TEACHING_STAGES),
                "label_overlap_basis": self.label_overlap_basis,
            },
            self.model_path,
        )
        logger.debug(f"[MLPFusion] 模型已儲存至 {self.model_path}")

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到 MLP 融合模型權重：{self.model_path}，請先執行訓練或使用 fuse_scores_mlp() 退回解析式公式"
            )
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("MLP checkpoint 格式無效，需重新訓練")
        expected = {
            "input_dim": MLP_INPUT_DIM,
            "feature_version": MLP_FEATURE_VERSION,
            "feature_names": list(MLP_FEATURE_NAMES),
            "teaching_stages": list(TEACHING_STAGES),
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(
                    f"MLP checkpoint {key} 不相容，請重新抽取特徵並訓練"
                )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("MLP checkpoint 缺少 model_state_dict，請重新訓練")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        logger.debug(f"[MLPFusion] 模型已從 {self.model_path} 載入")


def fuse_scores_mlp(segments: List[Dict], model_path: str = MLP_FUSION_MODEL_PATH) -> List[float]:
    if TORCH_AVAILABLE and os.path.exists(model_path):
        try:
            trainer = MLPFusionTrainer(model_path=model_path)
            scores = trainer.predict(segments)
            logger.info(f"[MLPFusion] 使用 MLP 模型推論（{len(segments)} 個片段）")
            return scores
        except Exception as e:
            logger.warning(f"[MLPFusion] MLP 推論失敗，退回解析式公式：{e}")

    # Fallback 機制
    from src.fusion.adaptive_fusion import fuse_scores
    logger.info("[MLPFusion] 啟用 Fallback，退回解析式 fuse_scores 公式")
    
    return [
        fuse_scores(
            s_text=seg.get("s_text", 0.0),
            s_visual=seg.get("s_visual", 0.0),
            visual_reliability=seg.get("visual_reliability", 0.0),
        ) for seg in segments
    ]