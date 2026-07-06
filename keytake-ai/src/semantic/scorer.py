"""
步驟二（後半）：三軌語意評分機制 (v3)
─────────────────────────────────────────────────────────────────────
v3 改進（回應「創新性不足」意見）：引入 LLM 做深度語意理解

三種模式（優先順序從上到下）：
  1. LLM 模式（推薦）：用 LLM 對每段文字做重要性判斷，理解最深
  2. SBERT 模式（中等）：與提示語料庫比對，泛化能力中等
  3. TF-IDF 模式（最弱）：純統計，對同領域影片容易失效

支援的 LLM 後端：
  - OpenAI（GPT-4o-mini / GPT-4o）：最準，費用約 $0.001/片段
  - 本地 Ollama（Llama3 / Gemma）：免費，需自行架設
  - Groq API（Llama3-8B-instant）：免費，速度快

使用方式：
  在 .env 設定環境變數：
    SEMANTIC_SCORER_MODE=llm    # llm / sbert / tfidf
    LLM_BACKEND=openai          # openai / ollama / groq
    OPENAI_API_KEY=sk-xxx       # OpenAI 才需要
    GROQ_API_KEY=gsk_xxx        # Groq 才需要
    OLLAMA_URL=http://localhost:11434  # Ollama 本地端點

對應計畫書 4.2 步驟二
"""

import os
import re
import time
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from config import SBERT_MODEL, TFIDF_TOP_K, SBERT_SIMILARITY_THRESHOLD


# ── 提示模板：給 LLM 用的評分指示 ──────────────────────────
LLM_PROMPT_TEMPLATE = """你是教學影片分析助手。以下是一段課堂逐字稿片段，請判斷它的「教學重要性」。

**重要（高分 0.7~1.0）：**
- 推導新公式或定理
- 定義核心概念
- 強調易錯點（「這裡很多人會搞錯」）
- 總結或結論（「總結一下」）
- 板書演示步驟

**不重要（低分 0.0~0.3）：**
- 寒暄或行政公告
- 過渡語（「那個」「嗯」「好」）
- 重複已講過的內容
- 閒聊或離題

片段：
「{text}」

請只回答一個 0 到 1 之間的數字，代表重要性分數。不要解釋。"""


class SemanticScorer:
    """
    三模式語意評分器（v3）

    模式選擇優先順序：
      1. 環境變數 SEMANTIC_SCORER_MODE（llm / sbert / tfidf）
      2. 若 LLM API key 有設定 → 自動用 llm 模式
      3. 降級為 sbert（預設）

    Args:
        prompt_corpus: SBERT 模式用的教學提示語料庫（向後兼容 v2）
        mode:          強制指定模式（覆蓋環境變數）
        llm_backend:   LLM 後端選擇（openai / ollama / groq）
    """

    def __init__(
        self,
        prompt_corpus: list[str],
        mode: Optional[str] = None,
        llm_backend: Optional[str] = None,
    ):
        # 決定運作模式
        self.mode = mode or os.getenv("SEMANTIC_SCORER_MODE", "sbert").lower()
        self.llm_backend = llm_backend or os.getenv("LLM_BACKEND", "openai").lower()

        # 如果環境變數有 API key，自動切換到 llm 模式
        if self.mode == "sbert" and self._has_llm_key():
            self.mode = "llm"
            print("[SemanticScorer] 偵測到 LLM API key，自動切換至 LLM 模式")

        # 初始化對應的後端
        self.prompt_corpus = prompt_corpus
        if self.mode == "llm":
            self._init_llm()
            print(f"[SemanticScorer] 模式：LLM（{self.llm_backend}）")
        elif self.mode == "sbert":
            self._init_sbert()
            print(f"[SemanticScorer] 模式：SBERT（提示語料庫 {len(prompt_corpus)} 句）")
        else:
            print(f"[SemanticScorer] 模式：TF-IDF（純統計）")

    def _has_llm_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY") or
                    os.getenv("GROQ_API_KEY") or
                    os.getenv("OLLAMA_URL"))

    def _init_llm(self):
        """初始化 LLM 客戶端"""
        if self.llm_backend == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY 未設定")
                self.llm_client = OpenAI(api_key=api_key)
                self.llm_model = "gpt-4o-mini"  # 便宜快速
            except Exception as e:
                print(f"[SemanticScorer] OpenAI 初始化失敗（{e}），降級為 SBERT")
                self.mode = "sbert"
                self._init_sbert()
        elif self.llm_backend == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY 未設定")
                self.llm_client = Groq(api_key=api_key)
                self.llm_model = "llama3-8b-8192"
            except Exception as e:
                print(f"[SemanticScorer] Groq 初始化失敗（{e}），降級為 SBERT")
                self.mode = "sbert"
                self._init_sbert()
        elif self.llm_backend == "ollama":
            try:
                import requests
                self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
                self.llm_model = os.getenv("OLLAMA_MODEL", "llama3")
                # 測試連線
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                if resp.status_code != 200:
                    raise ValueError("Ollama 伺服器未回應")
                self.llm_client = "ollama"  # 標記用 ollama
            except Exception as e:
                print(f"[SemanticScorer] Ollama 連線失敗（{e}），降級為 SBERT")
                self.mode = "sbert"
                self._init_sbert()

    def _init_sbert(self):
        """初始化 SBERT"""
        self.sbert = SentenceTransformer(SBERT_MODEL)
        self.corpus_embeddings = self.sbert.encode(self.prompt_corpus, convert_to_tensor=True)

    def score(self, segments: list[dict], alpha_tfidf: float = 0.5) -> list[dict]:
        """
        對所有片段評分，回傳帶有 S_text 的列表

        Args:
            segments:     Whisper 轉錄的片段列表
            alpha_tfidf:  TF-IDF 權重（只在 sbert 模式混合時用到）
        """
        if self.mode == "llm":
            scores = self._llm_scores(segments)
        elif self.mode == "sbert":
            tfidf = self.tfidf_scores(segments)
            sbert = self.sbert_scores(segments)
            scores = alpha_tfidf * tfidf + (1 - alpha_tfidf) * sbert
        else:  # tfidf
            scores = self.tfidf_scores(segments)

        for i, seg in enumerate(segments):
            seg["s_text"] = float(scores[i])
        return segments

    # ─────────────────────────────────────────────────────────────────
    # LLM 評分
    # ─────────────────────────────────────────────────────────────────

    def _llm_scores(self, segments: list[dict]) -> np.ndarray:
        """用 LLM 對每個片段評分"""
        scores = np.zeros(len(segments))
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text or len(text) < 5:
                continue
            try:
                score = self._query_llm(text)
                scores[i] = float(np.clip(score, 0.0, 1.0))
            except Exception as e:
                print(f"[SemanticScorer] LLM 評分失敗（片段 {i}）：{e}")
                scores[i] = 0.5  # 失敗時給中性分數
            time.sleep(0.05)  # 避免 rate limit
        return scores

    def _query_llm(self, text: str) -> float:
        """呼叫 LLM 取得單一片段的重要性分數"""
        prompt = LLM_PROMPT_TEMPLATE.format(text=text[:500])  # 截斷過長文字

        if self.llm_backend in ("openai", "groq"):
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            answer = resp.choices[0].message.content.strip()
        elif self.llm_backend == "ollama":
            import requests
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10},
                },
                timeout=10,
            )
            answer = resp.json()["response"].strip()
        else:
            return 0.5

        # 從回答中提取數字
        match = re.search(r"0?\.\d+|[01]", answer)
        if match:
            return float(match.group())
        return 0.5

    # ─────────────────────────────────────────────────────────────────
    # SBERT / TF-IDF 評分（向後兼容 v2）
    # ─────────────────────────────────────────────────────────────────

    def tfidf_scores(self, segments: list[dict]) -> np.ndarray:
        """對所有片段文字計算 TF-IDF，回傳每個片段的加總權重分數"""
        texts = [seg["text"] for seg in segments]
        if len(texts) < 2:
            return np.zeros(len(texts))
        vectorizer = TfidfVectorizer(max_features=TFIDF_TOP_K)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts).toarray()
            scores = tfidf_matrix.sum(axis=1)
            max_val = scores.max()
            return scores / max_val if max_val > 0 else scores
        except:
            return np.zeros(len(texts))

    def sbert_scores(self, segments: list[dict]) -> np.ndarray:
        """計算每個片段與提示語料庫的最大餘弦相似度"""
        scores = np.zeros(len(segments))
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue
            embedding = self.sbert.encode([text], convert_to_tensor=True)
            cosine = util.cos_sim(embedding, self.corpus_embeddings)
            scores[i] = float(cosine.max().cpu())
        return scores
