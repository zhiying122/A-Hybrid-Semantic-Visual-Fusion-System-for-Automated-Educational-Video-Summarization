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
    _instance: Optional["CLIPScorer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """覆寫 __new__ 確保即使用 CLIPScorer() 呼叫，也嚴格遵守單例模式"""
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
        self.__class__._initialized = True

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

    def compute_clip_score(self, roi: np.ndarray, text: str) -> float:
        """計算 ROI 影像與文字的正規化 CLIP 分數 [0, 1]"""
        if not self.available or roi is None or roi.size == 0 or not text.strip():
            return 0.0

        try:
            pil_image = Image.fromarray(roi[:, :, ::-1])
            
            inputs = self.processor(
                text=[text.strip()],
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            # 確保輸入張量與模型在同一個 Device (CPU/GPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            cosine_sim = torch.nn.functional.cosine_similarity(
                outputs.image_embeds, outputs.text_embeds
            ).item()

            return max(0.0, min((cosine_sim + 1.0) / 2.0, 1.0))

        except Exception as e:
            logger.warning(f"[CLIPScorer] 計算失敗: {e}")
            return 0.0

    def compute_clip_score_with_keywords(self, roi: np.ndarray, asr_keywords: list[str]) -> float:
        if not asr_keywords:
            return 0.0
            
        combined_text = " ".join(kw.strip() for kw in asr_keywords if kw.strip())
        if not combined_text:
            return 0.0
            
        return self.compute_clip_score(roi, combined_text)

    @property
    def is_available(self) -> bool:
        return self.available

    @property
    def score_threshold(self) -> float:
        return CLIP_SCORE_THRESHOLD