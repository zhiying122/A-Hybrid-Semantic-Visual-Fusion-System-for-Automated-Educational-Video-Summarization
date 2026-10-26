"""SBERT 語意相似度計算器

計算 OCR 辨識文字與 ASR 關鍵字之間的語意相似度。
使用 paraphrase-multilingual-MiniLM-L12-v2，支援中文。
模型載入失敗時自動降級為字元重疊率（Jaccard）計算。
"""

import logging
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from config import SBERT_MODEL, SBERT_SIMILARITY_THRESHOLD
except ImportError:
    SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    SBERT_SIMILARITY_THRESHOLD = 0.6

logger = logging.getLogger(__name__)


def compute_char_overlap(text1: str, text2: str) -> float:
    """字元集合 Jaccard 相似度（降級備援）。

    注意：英文不區分大小寫，且不考慮字元頻率，僅供粗糙相似度估計使用。
    """
    if not text1 or not text2:
        return 0.0
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


class SBERTCalculator:
    """計算 OCR 文字與 ASR 關鍵字的語意相似度。

    模型不可用時降級為字元重疊率計算。
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or SBERT_MODEL
        self.model: Optional["SentenceTransformer"] = None
        self.fallback_mode = False
        self._load_model()

    def _load_model(self) -> None:
        if not SBERT_AVAILABLE:
            logger.warning("sentence-transformers 未安裝，啟用字元重疊率降級備援模式")
            self.fallback_mode = True
            return

        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"SBERT 模型載入失敗 ({e})，啟用降級備援模式")
            self.fallback_mode = True

    def compute_similarity(self, ocr_text: str, transcript_text: str) -> float:
        """計算 OCR 文字與轉錄文字的語意相似度。

        使用 cosine similarity，結果已在 [0, 1] 範圍。
        SBERT 的 paraphrase-multilingual 模型對語意相似句子輸出接近 1.0，
        不相關句子接近 0.0，不需要再做 (x+1)/2 的線性映射。
        """
        if not ocr_text or not transcript_text:
            return 0.0
        ocr_text = ocr_text.strip()
        transcript_text = transcript_text.strip()
        if not ocr_text or not transcript_text:
            return 0.0

        if self.fallback_mode or self.model is None:
            return compute_char_overlap(ocr_text, transcript_text)

        try:
            embeddings = self.model.encode(
                [ocr_text, transcript_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # normalize_embeddings=True 確保向量已 L2 正規化，dot product == cosine similarity
            cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
            return float(np.clip(cosine_sim, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"SBERT 計算失敗 ({e})，使用字元重疊率降級備援")
            return compute_char_overlap(ocr_text, transcript_text)

    def compute_similarity_with_keywords(
        self, ocr_text: str, asr_keywords: list[str]
    ) -> float:
        if not asr_keywords:
            return 0.0
        return self.compute_similarity(ocr_text, " ".join(asr_keywords))

    @property
    def is_fallback_mode(self) -> bool:
        return self.fallback_mode
