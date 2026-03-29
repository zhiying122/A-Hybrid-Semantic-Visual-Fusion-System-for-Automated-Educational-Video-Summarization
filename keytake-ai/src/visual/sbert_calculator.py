"""
SBERT 語意相似度計算器
- 使用 Sentence-BERT 計算 OCR 文字與 ASR 關鍵字的語意相似度
- 降級備援：當 SBERT 模型載入失敗時，使用字元重疊率計算
對應計畫書 4.2 步驟三
"""

import logging
from typing import Optional

import numpy as np

# 嘗試載入 sentence-transformers，若失敗則使用降級備援
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# 從 config 載入設定參數
try:
    from config import SBERT_MODEL, SBERT_SIMILARITY_THRESHOLD
except ImportError:
    SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    SBERT_SIMILARITY_THRESHOLD = 0.6

logger = logging.getLogger(__name__)


def compute_char_overlap(text1: str, text2: str) -> float:
    """
    計算兩個字串的字元重疊率（降級備援）
    
    Args:
        text1: 第一個字串
        text2: 第二個字串
        
    Returns:
        字元重疊率 [0.0, 1.0]
    """
    if not text1 or not text2:
        return 0.0
    
    set1 = set(text1)
    set2 = set(text2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


class SBERTCalculator:
    """
    SBERT 語意相似度計算器
    
    使用 Sentence-BERT 計算 OCR 文字與 ASR 關鍵字之間的語意相似度。
    當 SBERT 模型載入失敗時，自動降級為字元重疊率計算。
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化 SBERT 計算器
        
        Args:
            model_name: SBERT 模型名稱，預設使用 config.py 中的 SBERT_MODEL
        """
        self.model_name = model_name or SBERT_MODEL
        self.model: Optional["SentenceTransformer"] = None
        self.fallback_mode = False
        
        self._load_model()
    
    def _load_model(self) -> None:
        """載入 SBERT 模型，失敗時啟用降級備援模式"""
        if not SBERT_AVAILABLE:
            logger.warning(
                "sentence-transformers 未安裝，啟用字元重疊率降級備援模式"
            )
            self.fallback_mode = True
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"成功載入 SBERT 模型: {self.model_name}")
        except Exception as e:
            logger.warning(
                f"SBERT 模型載入失敗 ({e})，啟用字元重疊率降級備援模式"
            )
            self.fallback_mode = True
    
    def _compute_cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        計算兩個向量的餘弦相似度
        
        Args:
            embedding1: 第一個嵌入向量
            embedding2: 第二個嵌入向量
            
        Returns:
            餘弦相似度 [0.0, 1.0]（已正規化至非負範圍）
        """
        # 處理零向量情況
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 計算餘弦相似度
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # 將餘弦相似度從 [-1, 1] 正規化至 [0, 1]
        # 使用 (cosine_sim + 1) / 2 進行正規化
        normalized_sim = (cosine_sim + 1) / 2
        
        # 確保結果在 [0.0, 1.0] 範圍內
        return float(np.clip(normalized_sim, 0.0, 1.0))
    
    def _compute_fallback_similarity(
        self, 
        ocr_text: str, 
        transcript_text: str
    ) -> float:
        """
        使用字元重疊率計算相似度（降級備援）
        
        Args:
            ocr_text: OCR 辨識結果文字
            transcript_text: ASR 轉錄文字或關鍵字
            
        Returns:
            字元重疊率 [0.0, 1.0]
        """
        return compute_char_overlap(ocr_text, transcript_text)
    
    def compute_similarity(
        self, 
        ocr_text: str, 
        transcript_text: str
    ) -> float:
        """
        計算 OCR 文字與轉錄文字的語意相似度
        
        使用 SBERT 計算餘弦相似度。當 SBERT 不可用時，
        降級為字元重疊率計算。
        
        Args:
            ocr_text: OCR 辨識結果文字
            transcript_text: ASR 轉錄文字（可以是關鍵字列表合併的字串）
            
        Returns:
            語意相似度分數 [0.0, 1.0]
        """
        # 處理空字串情況
        if not ocr_text or not transcript_text:
            return 0.0
        
        # 清理輸入文字
        ocr_text = ocr_text.strip()
        transcript_text = transcript_text.strip()
        
        if not ocr_text or not transcript_text:
            return 0.0
        
        # 使用降級備援模式
        if self.fallback_mode or self.model is None:
            return self._compute_fallback_similarity(ocr_text, transcript_text)
        
        try:
            # 使用 SBERT 計算嵌入向量
            embeddings = self.model.encode(
                [ocr_text, transcript_text],
                convert_to_numpy=True
            )
            
            # 計算餘弦相似度
            similarity = self._compute_cosine_similarity(
                embeddings[0], 
                embeddings[1]
            )
            
            return similarity
            
        except Exception as e:
            logger.warning(
                f"SBERT 計算失敗 ({e})，使用字元重疊率降級備援"
            )
            return self._compute_fallback_similarity(ocr_text, transcript_text)
    
    def compute_similarity_with_keywords(
        self, 
        ocr_text: str, 
        asr_keywords: list[str]
    ) -> float:
        """
        計算 OCR 文字與 ASR 關鍵字列表的語意相似度
        
        將關鍵字列表合併為單一字串後計算相似度。
        
        Args:
            ocr_text: OCR 辨識結果文字
            asr_keywords: ASR 提取的關鍵字列表
            
        Returns:
            語意相似度分數 [0.0, 1.0]
        """
        if not asr_keywords:
            return 0.0
        
        # 將關鍵字列表合併為空格分隔的字串
        transcript_text = " ".join(asr_keywords)
        
        return self.compute_similarity(ocr_text, transcript_text)
    
    @property
    def is_fallback_mode(self) -> bool:
        """檢查是否處於降級備援模式"""
        return self.fallback_mode
