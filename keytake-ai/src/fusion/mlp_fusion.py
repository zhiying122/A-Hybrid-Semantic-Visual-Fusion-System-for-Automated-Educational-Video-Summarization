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
    MLP_FUSION_MODEL_PATH = "weights/mlp_fusion.pth"
    MLP_FUSION_EPOCHS = 100
    MLP_FUSION_LR = 0.001

logger = logging.getLogger(__name__)

TEACHING_STAGES = ["definition", "derivation", "example", "summary", "transition", "qa"]
_STAGE_TO_IDX = {stage: i for i, stage in enumerate(TEACHING_STAGES)}


if TORCH_AVAILABLE:
    class MLPFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 輸入：s_text, s_visual, visual_reliability, gesture_intent_score,
            #       s_prosodic, one-hot teaching_stage (6 維) = 11 維
            self.network = nn.Sequential(
                nn.Linear(11, 32),
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
    def __init__(self, model_path: str = MLP_FUSION_MODEL_PATH):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安裝，無法使用 MLPFusionTrainer")
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPFusionModel().to(self.device)

    def train(
        self,
        segments: List[Dict],
        ground_truth: List[Tuple[float, float]],
        epochs: int = MLP_FUSION_EPOCHS,
        lr: float = MLP_FUSION_LR,
        seed: int = 42,
    ) -> dict:
        # 固定 seed 確保 train/val split 可重現
        np.random.seed(seed)
        torch.manual_seed(seed)

        features = np.array([self._build_features(seg) for seg in segments], dtype=np.float32)
        labels = np.array([self._compute_label(seg, ground_truth) for seg in segments], dtype=np.float32)

        n_total = len(features)
        n_train = max(1, int(n_total * 0.8))
        indices = np.random.permutation(n_total)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train = torch.tensor(features[train_idx], device=self.device)
        y_train = torch.tensor(labels[train_idx], device=self.device).unsqueeze(1)
        X_val = torch.tensor(features[val_idx], device=self.device) if len(val_idx) > 0 else None
        y_val = torch.tensor(labels[val_idx], device=self.device).unsqueeze(1) if len(val_idx) > 0 else None

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
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
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "positive_ratio": float(labels.sum() / len(labels)) if len(labels) > 0 else 0.0,
        }

    def predict(self, segments: List[Dict]) -> List[float]:
        self._load_model()
        self.model.eval()

        features = np.array([self._build_features(seg) for seg in segments], dtype=np.float32)
        X = torch.tensor(features, device=self.device)

        with torch.no_grad():
            scores = self.model(X).squeeze(1).cpu().numpy()
        return scores.tolist()

    def _build_features(self, seg: Dict) -> np.ndarray:
        """建構 11 維特徵向量：s_text, s_visual, visual_reliability, gesture_intent_score,
        s_prosodic, one-hot teaching_stage (6 維)。"""
        feature = np.zeros(11, dtype=np.float32)
        feature[0] = float(seg.get("s_text", 0.0))
        feature[1] = float(seg.get("s_visual", 0.0))
        feature[2] = float(seg.get("visual_reliability", 1.0))
        feature[3] = float(seg.get("gesture_intent_score", 0.0))
        feature[4] = float(seg.get("s_prosodic", 0.5))

        stage = seg.get("teaching_stage", "")
        if stage in _STAGE_TO_IDX:
            feature[5 + _STAGE_TO_IDX[stage]] = 1.0

        return feature

    def _compute_label(self, seg: Dict, ground_truth: List[Tuple[float, float]]) -> float:
        seg_start, seg_end = seg.get("start", 0.0), seg.get("end", 0.0)
        seg_len = seg_end - seg_start

        if seg_len <= 0:
            return 0.0

        for gt_start, gt_end in ground_truth:
            overlap = min(seg_end, gt_end) - max(seg_start, gt_start)
            if overlap / seg_len >= 0.5:
                return 1.0
        return 0.0

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.debug(f"[MLPFusion] 模型已儲存至 {self.model_path}")

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到 MLP 融合模型權重：{self.model_path}，請先執行訓練或使用 fuse_scores_mlp() 退回解析式公式"
            )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        logger.debug(f"[MLPFusion] 模型已從 {self.model_path} 載入")


def fuse_scores_mlp(segments: List[Dict], model_path: str = MLP_FUSION_MODEL_PATH) -> List[float]:
    if TORCH_AVAILABLE and os.path.exists(model_path):
        trainer = MLPFusionTrainer(model_path=model_path)
        try:
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
            visual_reliability=seg.get("visual_reliability", 1.0),
        ) for seg in segments
    ]