"""
步驟二（後半）：雙軌語意評分機制
1. TF-IDF 關鍵詞統計
2. Sentence-BERT 與教學提示語料庫的餘弦相似度
最終合併為語意分數 S_text
對應計畫書 4.2 步驟二
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from config import SBERT_MODEL, TFIDF_TOP_K, SBERT_SIMILARITY_THRESHOLD


class SemanticScorer:
    def __init__(self, prompt_corpus: list[str]):
        """
        prompt_corpus: 預先建構的「基礎教學提示語料庫」
        例如：["這邊非常重要", "導致這個結果的原因是", "請注意這個公式"]
        對應計畫書中蒐集 10~20 小時跨學科影片標註的引導語句
        """
        self.sbert = SentenceTransformer(SBERT_MODEL)
        self.corpus_embeddings = self.sbert.encode(prompt_corpus, convert_to_tensor=True)
        self.prompt_corpus = prompt_corpus

    def tfidf_scores(self, segments: list[dict]) -> np.ndarray:
        """
        對所有片段文字計算 TF-IDF，回傳每個片段的加總權重分數
        """
        texts = [seg["text"] for seg in segments]
        vectorizer = TfidfVectorizer(max_features=TFIDF_TOP_K)
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        # 每個片段的分數 = 該片段所有詞的 TF-IDF 加總
        scores = tfidf_matrix.sum(axis=1)
        # 正規化到 [0, 1]
        max_val = scores.max()
        return scores / max_val if max_val > 0 else scores

    def sbert_scores(self, segments: list[dict]) -> np.ndarray:
        """
        計算每個片段與提示語料庫的最大餘弦相似度
        捕捉「這邊很重要」等非關鍵詞但具語意重要性的語句
        """
        scores = np.zeros(len(segments))
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:  # 空字串直接給 0，避免 SBERT 報錯
                continue
            embedding = self.sbert.encode([text], convert_to_tensor=True)
            cosine = util.cos_sim(embedding, self.corpus_embeddings)
            scores[i] = float(cosine.max().cpu())
        return scores

    def score(self, segments: list[dict], alpha_tfidf: float = 0.5) -> list[dict]:
        """
        合併 TF-IDF 與 SBERT 分數，回傳帶有 S_text 的片段列表
        alpha_tfidf: TF-IDF 佔語意分數的比重（1-alpha_tfidf 為 SBERT）
        """
        tfidf = self.tfidf_scores(segments)
        sbert = self.sbert_scores(segments)
        combined = alpha_tfidf * tfidf + (1 - alpha_tfidf) * sbert

        for i, seg in enumerate(segments):
            seg["s_text"] = float(combined[i])
        return segments
