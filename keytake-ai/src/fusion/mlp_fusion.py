"""
MLP 融合模組：以監督式學習取代靜態 α/β 融合公式
─────────────────────────────────────────────────────────────────────
本模組實作一個可學習的多層感知器（MLP），用於取代 adaptive_fusion.py 中
手動調校的 fuse_scores 線性加權公式。

核心優勢：
  - 從 Ground Truth 標註資料中自動學習最佳特徵組合權重
  - 能夠捕捉非線性交互作用（例如：手勢意圖 × 教學階段 的交叉效果）
  - 自動適應不同教學風格與學科領域，無需重新手動調參
  - 當無訓練模型可用時，自動退回解析式公式，確保系統可用性

輸入特徵（10 維）：
  [s_text, s_visual, visual_reliability, gesture_intent_score,
   teaching_stage_onehot(6 dims: definition/derivation/example/summary/transition/qa)]

輸出：
  保留機率 ∈ [0, 1]（「保留此片段」的信心度）

對應計畫書 4.2 步驟四改進方案
"""

import os
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 設定值（從 config.py 匯入，若失敗則使用預設值）─────────────
try:
    from config import MLP_FUSION_MODEL_PATH
except ImportError:
    MLP_FUSION_MODEL_PATH = "weights/mlp_fusion.pth"

try:
    from config import MLP_FUSION_EPOCHS
except ImportError:
    MLP_FUSION_EPOCHS = 100

try:
    from config import MLP_FUSION_LR
except ImportError:
    MLP_FUSION_LR = 0.001

logger = logging.getLogger(__name__)

# ── 教學階段 one-hot 對應表 ──────────────────────────────────────
TEACHING_STAGES = ["definition", "derivation", "example", "summary", "transition", "qa"]


class MLPFusionModel(nn.Module):
    """
    可學習的 MLP 融合模型

    架構：Linear(10, 32) → ReLU → Dropout(0.3) → Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid

    輸入：10 維特徵向量
        - s_text:               語意分數 ∈ [0, 1]
        - s_visual:             視覺分數 ∈ [0, 1]
        - visual_reliability:   視覺可靠度 ∈ [0, 1]
        - gesture_intent_score: 手勢意圖分數 ∈ [0, 1]
        - teaching_stage:       教學階段 one-hot 編碼（6 維）

    輸出：保留機率 ∈ [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播，輸入 shape: (batch, 10)，輸出 shape: (batch, 1)"""
        return self.network(x)


class MLPFusionTrainer:
    """
    MLP 融合模型的訓練與推論管理器

    負責：
      - 從片段字典建構特徵向量
      - 根據 Ground Truth 生成二元標籤
      - 訓練模型並以驗證集 loss 選擇最佳權重
      - 載入模型並對新片段進行推論
    """

    def __init__(self, model_path: str = MLP_FUSION_MODEL_PATH):
        """
        初始化訓練器

        Args:
            model_path: 模型權重儲存/載入路徑
        """
        self.model_path = model_path
        self.model = MLPFusionModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        segments: list[dict],
        ground_truth: list[tuple[float, float]],
        epochs: int = MLP_FUSION_EPOCHS,
        lr: float = MLP_FUSION_LR,
    ) -> dict:
        """
        訓練 MLP 融合模型

        Args:
            segments:     片段列表，每個片段包含特徵欄位
            ground_truth: 人工標註的重點時間區間 [(start, end), ...]
            epochs:       訓練輪數
            lr:           學習率

        Returns:
            訓練統計字典，包含 train_loss、val_loss、best_epoch 等資訊

        標籤規則：
            若片段與任一 Ground Truth 區間的重疊比例 ≥ 50%，標記為 1（保留），否則為 0
        """
        # 建構特徵矩陣與標籤
        features = np.array([self._build_features(seg) for seg in segments], dtype=np.float32)
        labels = np.array(
            [self._compute_label(seg, ground_truth) for seg in segments], dtype=np.float32
        )

        # 80/20 訓練/驗證分割
        n_total = len(features)
        n_train = max(1, int(n_total * 0.8))
        indices = np.random.permutation(n_total)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train = torch.tensor(features[train_idx], device=self.device)
        y_train = torch.tensor(labels[train_idx], device=self.device).unsqueeze(1)
        X_val = torch.tensor(features[val_idx], device=self.device) if len(val_idx) > 0 else None
        y_val = torch.tensor(labels[val_idx], device=self.device).unsqueeze(1) if len(val_idx) > 0 else None

        # 建立 DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 損失函數與優化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 訓練迴圈
        best_val_loss = float("inf")
        best_epoch = 0
        train_losses = []
        val_losses = []

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= n_train
            train_losses.append(epoch_loss)

            # 驗證集評估
            val_loss = epoch_loss  # 若無驗證集，以訓練 loss 代替
            if X_val is not None and len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                self.model.train()
            val_losses.append(val_loss)

            # 儲存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self._save_model()

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"[MLPFusion] Epoch {epoch+1}/{epochs} "
                    f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}"
                )

        stats = {
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
        logger.info(f"[MLPFusion] 訓練完成：best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
        return stats

    def predict(self, segments: list[dict]) -> list[float]:
        """
        使用已訓練模型對片段進行融合分數預測

        Args:
            segments: 片段列表，每個片段包含特徵欄位

        Returns:
            每個片段的融合分數列表 ∈ [0, 1]

        Raises:
            FileNotFoundError: 若模型權重檔案不存在
        """
        self._load_model()
        self.model.eval()

        features = np.array([self._build_features(seg) for seg in segments], dtype=np.float32)
        X = torch.tensor(features, device=self.device)

        with torch.no_grad():
            scores = self.model(X).squeeze(1).cpu().numpy()

        return scores.tolist()

    def _build_features(self, seg: dict) -> np.ndarray:
        """
        從片段字典提取 10 維特徵向量

        特徵排列：
            [0] s_text               - 語意分數
            [1] s_visual             - 視覺分數
            [2] visual_reliability   - 視覺可靠度
            [3] gesture_intent_score - 手勢意圖分數
            [4:10] teaching_stage    - 教學階段 one-hot（6 維）

        Args:
            seg: 片段字典，應包含上述欄位（缺失時以 0.0 填充）

        Returns:
            10 維 numpy 陣列
        """
        # 連續特徵
        s_text = float(seg.get("s_text", 0.0))
        s_visual = float(seg.get("s_visual", 0.0))
        visual_reliability = float(seg.get("visual_reliability", 1.0))
        gesture_intent_score = float(seg.get("gesture_intent_score", 0.0))

        # 教學階段 one-hot 編碼
        stage = seg.get("teaching_stage", "")
        stage_onehot = np.zeros(6, dtype=np.float32)
        if stage in TEACHING_STAGES:
            stage_onehot[TEACHING_STAGES.index(stage)] = 1.0

        # 組合為 10 維向量
        feature = np.array(
            [s_text, s_visual, visual_reliability, gesture_intent_score],
            dtype=np.float32,
        )
        feature = np.concatenate([feature, stage_onehot])
        return feature

    def _compute_label(self, seg: dict, ground_truth: list[tuple[float, float]]) -> float:
        """
        計算片段標籤：與任一 GT 區間重疊 ≥ 50% 則為正樣本

        Args:
            seg:          片段字典（需含 start、end 欄位）
            ground_truth: Ground Truth 時間區間列表

        Returns:
            1.0（正樣本）或 0.0（負樣本）
        """
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        seg_len = seg_end - seg_start

        if seg_len <= 0:
            return 0.0

        for gt_start, gt_end in ground_truth:
            overlap = min(seg_end, gt_end) - max(seg_start, gt_start)
            if overlap / seg_len >= 0.5:
                return 1.0
        return 0.0

    def _save_model(self):
        """儲存模型權重至指定路徑（自動建立目錄）"""
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.debug(f"[MLPFusion] 模型已儲存至 {self.model_path}")

    def _load_model(self):
        """載入模型權重"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到 MLP 融合模型權重：{self.model_path}，"
                f"請先執行訓練或使用 fuse_scores_mlp() 自動退回解析式公式"
            )
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        logger.debug(f"[MLPFusion] 模型已從 {self.model_path} 載入")


def fuse_scores_mlp(
    segments: list[dict],
    model_path: str = MLP_FUSION_MODEL_PATH,
) -> list[float]:
    """
    便利函數：載入 MLP 模型並回傳各片段的融合分數

    若模型檔案不存在，自動退回 adaptive_fusion.fuse_scores 解析式公式，
    確保系統在未訓練時仍可正常運作。

    Args:
        segments:   片段列表
        model_path: 模型權重路徑

    Returns:
        各片段的融合分數列表 ∈ [0, 1]
    """
    if os.path.exists(model_path):
        # 使用已訓練的 MLP 模型
        trainer = MLPFusionTrainer(model_path=model_path)
        try:
            scores = trainer.predict(segments)
            logger.info(f"[MLPFusion] 使用 MLP 模型推論（{len(segments)} 個片段）")
            return scores
        except Exception as e:
            logger.warning(f"[MLPFusion] MLP 推論失敗，退回解析式公式：{e}")

    # 退回原始解析式融合公式
    from src.fusion.adaptive_fusion import fuse_scores

    logger.info("[MLPFusion] 模型不存在，退回解析式 fuse_scores 公式")
    scores = [
        fuse_scores(
            s_text=seg.get("s_text", 0.0),
            s_visual=seg.get("s_visual", 0.0),
            visual_reliability=seg.get("visual_reliability", 1.0),
        )
        for seg in segments
    ]
    return scores
