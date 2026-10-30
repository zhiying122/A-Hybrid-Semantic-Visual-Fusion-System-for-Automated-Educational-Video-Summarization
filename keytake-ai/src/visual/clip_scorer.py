"""
CLIP 視覺-文字對齊評分器

在 OCR 辨識品質不佳時，直接計算板書影像與口語關鍵字間的跨模態語意相似度，
作為 OCR→SBERT 路徑的補充或替代方案。全域採用 Singleton 模式管理。
"""

import logging
from typing import Optional
import numpy as np

try:
    from transformers import CLIPModel, CLIPProcessor
    import torch
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from config import CLIP_MODEL_NAME, CLIP_SCORE_THRESHOLD
except ImportError:
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CLIP_SCORE_THRESHOLD = 0.25

logger = logging.getLogger(__name__)


class CLIPScorer:
    """CLIP 視覺-文字語意相似度評分器（Singleton）。

    全域只保留一個實例，避免重複載入大型模型。
    第二次呼叫 CLIPScorer(model_name) 時 model_name 參數會被忽略。
    """

    _instance: Optional["CLIPScorer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None):
        if self._initialized:
            return

        self.model_name = model_name or CLIP_MODEL_NAME
        self.model: Optional["CLIPModel"] = None
        self.processor: Optional["CLIPProcessor"] = None
        self.available = False
        self.device = "cpu"
        
        self._load_model()
        self.__class__._initialized = self.available

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "CLIPScorer":
        return cls(model_name=model_name)

    def _load_model(self) -> None:
        if not CLIP_AVAILABLE:
            logger.warning("套件不足 (transformers, torch, Pillow)，CLIP 評分功能不可用。")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.available = True
            logger.info(f"[CLIPScorer] 成功載入模型 {self.model_name} on {self.device}")
        except Exception as e:
            logger.warning(f"[CLIPScorer] 模型載入失敗 ({e})，評分將返回 0.0")
            self.available = False

    def compute_clip_score(
        self,
        roi: np.ndarray,
        text: str,
        color_format: str = "bgr",
    ) -> float:
        """Return clipped cosine similarity, not a probability or importance score."""
        if not self.available:
            return 0.0

        if not isinstance(roi, np.ndarray) or roi.size == 0:
            logger.warning("[CLIPScorer] ROI 必須是非空的 NumPy 影像")
            return 0.0
        if roi.ndim != 3 or roi.shape[2] != 3:
            logger.warning("[CLIPScorer] ROI 必須是三通道彩色影像")
            return 0.0
        if color_format not in {"bgr", "rgb"}:
            raise ValueError("color_format 必須是 'bgr' 或 'rgb'")
        if not isinstance(text, str) or not text.strip():
            return 0.0

        try:
            rgb_image = roi[:, :, ::-1] if color_format == "bgr" else roi
            pil_image = Image.fromarray(np.ascontiguousarray(rgb_image))

            inputs = self.processor(
                text=[text.strip()],
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 兩個 encoder 輸出共同嵌入；cosine 只表示對齊程度，不能解讀成機率。
            with torch.no_grad():
                outputs = self.model(**inputs)

            cosine_sim = torch.nn.functional.cosine_similarity(
                outputs.image_embeds, outputs.text_embeds
            ).item()
            if not np.isfinite(cosine_sim):
                raise ValueError("CLIP 相似度不是有限數值")

            # 專案把負相似度裁成 0，方便後續 [0,1] 分數融合；因此丟失了負值資訊。
            return float(np.clip(cosine_sim, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"[CLIPScorer] 計算失敗: {e}")
            return 0.0

    def compute_clip_score_with_keywords(
        self,
        roi: np.ndarray,
        asr_keywords: list[str],
        color_format: str = "bgr",
    ) -> float:
        if not asr_keywords:
            return 0.0

        keywords = [
            keyword.strip()
            for keyword in asr_keywords
            if isinstance(keyword, str) and keyword.strip()
        ]
        if not keywords:
            return 0.0

        scores = [
            self.compute_clip_score(roi, keyword, color_format=color_format)
            for keyword in keywords[:20]
        ]
        return float(np.mean(scores)) if scores else 0.0

    @property
    def is_available(self) -> bool:
        return self.available

    @property
    def score_threshold(self) -> float:
        return CLIP_SCORE_THRESHOLD