"""
步驟二（後半）：三軌語意評分機制 (v4)
─────────────────────────────────────────────────────────────────────
v4 重大升級：LLM 深度教學結構理解

v3 做法：LLM 只回傳一個 0~1 的數字
v4 做法：LLM 同時輸出：
  1. 重要性分數 (0~1)
  2. 教學階段類型 (definition / derivation / example / summary / transition / qa)
  3. 一句摘要說明（中文）
  4. 全課結構分析：generate_course_summary() 一次掌握整堂課脈絡

教學階段定義：
  - definition   : 定義核心概念（分數加乘 ×1.1）
  - derivation   : 公式推導或定理證明（最重要，分數加乘 ×1.2）
  - example      : 舉例說明（分數加乘 ×1.0）
  - summary      : 總結回顧（分數加乘 ×1.1）
  - transition   : 換場或過渡語（分數下壓 ×0.3）
  - qa           : 問答互動（分數加乘 ×0.8）

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

import json
import os
import re
import time
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from config import SBERT_MODEL, TFIDF_TOP_K, SBERT_SIMILARITY_THRESHOLD
from src.semantic.llm_cache import LLMCache


# ── 教學階段類型 ──────────────────────────────────────────
TEACHING_STAGE_TYPES = {"definition", "derivation", "example", "summary", "transition", "qa"}

# 各階段對應前端顯示標籤（中文）
STAGE_LABELS = {
    "definition":  "概念定義",
    "derivation":  "公式推導",
    "example":     "例題說明",
    "summary":     "重點總結",
    "transition":  "過渡換場",
    "qa":          "問答互動",
}

# 各階段對應前端顏色 tag
STAGE_TO_SEG_TYPE = {
    "definition":  "semantic",
    "derivation":  "combined",  # 推導最重要，標紫色（雙模態）
    "example":     "visual",
    "summary":     "combined",
    "transition":  "semantic",
    "qa":          "semantic",
}

# 各階段分數加乘係數
STAGE_WEIGHT = {
    "definition":  1.1,
    "derivation":  1.2,
    "example":     1.0,
    "summary":     1.1,
    "transition":  0.3,
    "qa":          0.8,
}


# ── 提示模板 v4：結構化 JSON 輸出 ────────────────────────
LLM_PROMPT_TEMPLATE = """你是教學影片智能分析助手。以下是一段課堂逐字稿片段。

請分析這段文字並以 JSON 格式回答，格式如下：
{{
  "score": <0到1之間的數字，代表教學重要性>,
  "stage": "<教學階段：definition/derivation/example/summary/transition/qa>",
  "summary": "<用一句話（15字以內）概括這段的重點內容>"
}}

教學階段說明：
- definition：定義核心概念、術語解釋
- derivation：公式推導、定理證明、步驟演算
- example：舉例說明、解題示範
- summary：總結回顧、重點整理
- transition：換場過渡語、行政公告、寒暄
- qa：師生問答互動

片段文字：
「{text}」

只回答 JSON，不要其他說明。"""


# ── 全課摘要提示模板 ────────────────────────────────────
COURSE_SUMMARY_PROMPT = """你是教學影片課程分析師。以下是一堂課的完整逐字稿片段清單（含時間戳），格式為「[時間] 文字」。

請分析這堂課並以 JSON 格式回答：
{{
  "title": "<推測這堂課的主題，例如：微積分 - 極限的定義與計算>",
  "summary": "<2~3句話描述這堂課的核心內容>",
  "key_concepts": ["<概念1>", "<概念2>", "<概念3>"],
  "structure": [
    {{"stage": "introduction", "description": "<引入部分的概述>"}},
    {{"stage": "main_content", "description": "<主要內容概述>"}},
    {{"stage": "conclusion", "description": "<總結部分概述>"}}
  ]
}}

課堂內容（前 30 個片段）：
{transcript}

只回答 JSON，不要其他說明。"""


class SemanticScorer:
    """
    三模式語意評分器（v4）

    v4 新增：
      - LLM 模式輸出結構化 JSON（分數 + 教學階段 + 一句摘要）
      - generate_course_summary()：全課結構分析
      - 各教學階段加乘係數

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

        # v5：初始化 LLM 快取
        self._cache = LLMCache()

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
        對所有片段評分，回傳帶有 s_text、teaching_stage、stage_label、segment_summary 的列表

        Args:
            segments:     Whisper 轉錄的片段列表
            alpha_tfidf:  TF-IDF 權重（只在 sbert 模式混合時用到）
        """
        if self.mode == "llm":
            scores, stages, summaries = self._llm_scores(segments)
        elif self.mode == "sbert":
            tfidf = self.tfidf_scores(segments)
            sbert = self.sbert_scores(segments)
            scores = alpha_tfidf * tfidf + (1 - alpha_tfidf) * sbert
            stages = ["semantic"] * len(segments)
            summaries = [seg.get("text", "")[:20] for seg in segments]
        else:  # tfidf
            scores = self.tfidf_scores(segments)
            stages = ["semantic"] * len(segments)
            summaries = [seg.get("text", "")[:20] for seg in segments]

        for i, seg in enumerate(segments):
            seg["s_text"] = float(scores[i])
            seg["teaching_stage"] = stages[i] if i < len(stages) else "transition"
            seg["stage_label"] = STAGE_LABELS.get(seg["teaching_stage"], seg["teaching_stage"])
            seg["segment_summary"] = summaries[i] if i < len(summaries) else ""
        return segments

    def generate_course_summary(self, segments: list[dict]) -> dict:
        """
        v4 新增：全課結構分析，回傳課程標題、摘要、關鍵概念

        Args:
            segments: 已評分的片段列表（帶有 text、start 欄位）

        Returns:
            {
              "title": str,
              "summary": str,
              "key_concepts": list[str],
              "structure": list[dict]
            }
        """
        if self.mode != "llm":
            # 非 LLM 模式：從高分片段拼湊
            top_segs = sorted(segments, key=lambda s: s.get("s_text", 0), reverse=True)[:5]
            return {
                "title": "教學影片摘要",
                "summary": " ".join(s.get("text", "")[:30] for s in top_segs[:3]),
                "key_concepts": [s.get("segment_summary", "") for s in top_segs[:5] if s.get("segment_summary")],
                "structure": [],
            }

        # 組合前 30 個片段的逐字稿（避免 token 過長）
        transcript_lines = []
        for seg in segments[:30]:
            m, s = divmod(int(seg.get("start", 0)), 60)
            text = seg.get("text", "").strip()
            if text:
                transcript_lines.append(f"[{m:02d}:{s:02d}] {text}")
        transcript = "\n".join(transcript_lines)

        prompt = COURSE_SUMMARY_PROMPT.format(transcript=transcript)

        try:
            raw = self._call_llm_raw(prompt, max_tokens=400)
            result = _parse_json_safe(raw)
            if result and "title" in result:
                print(f"[SemanticScorer] 全課分析完成：{result.get('title', '')}")
                return result
        except Exception as e:
            print(f"[SemanticScorer] 全課分析失敗：{e}")

        return {
            "title": "教學影片摘要",
            "summary": "",
            "key_concepts": [],
            "structure": [],
        }

    # ─────────────────────────────────────────────────────────────────
    # LLM 評分（v4：結構化 JSON 輸出）
    # ─────────────────────────────────────────────────────────────────

    def _llm_scores(self, segments: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
        """
        v4：LLM 對每個片段同時輸出分數、教學階段、摘要句

        Returns:
            (scores, stages, summaries)
        """
        scores = np.zeros(len(segments))
        stages = ["transition"] * len(segments)
        summaries = [""] * len(segments)

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text or len(text) < 5:
                stages[i] = "transition"
                continue
            try:
                result = self._query_llm_structured(text)
                raw_score = float(np.clip(result.get("score", 0.5), 0.0, 1.0))
                stage = result.get("stage", "transition")
                if stage not in TEACHING_STAGE_TYPES:
                    stage = "transition"

                # 依教學階段加乘分數
                boosted = raw_score * STAGE_WEIGHT.get(stage, 1.0)
                scores[i] = float(np.clip(boosted, 0.0, 1.0))
                stages[i] = stage
                summaries[i] = result.get("summary", text[:20])
            except Exception as e:
                print(f"[SemanticScorer] LLM 評分失敗（片段 {i}）：{e}")
                scores[i] = 0.5
                stages[i] = "transition"
                summaries[i] = text[:20]
            time.sleep(0.05)  # 避免 rate limit

        # v5：確保快取寫入磁碟
        self._cache.flush()

        return scores, stages, summaries

    def _query_llm_structured(self, text: str) -> dict:
        """v4：呼叫 LLM 取得結構化 JSON（分數 + 階段 + 摘要），v5：含快取"""
        # v5：先查快取
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        prompt = LLM_PROMPT_TEMPLATE.format(text=text[:400])
        raw = self._call_llm_raw(prompt, max_tokens=80)
        result = _parse_json_safe(raw)
        if result and "score" in result:
            # v5：存入快取
            self._cache.put(text, result)
            return result
        # 降級：嘗試只提取數字
        match = re.search(r"0?\.\d+|[01]", raw)
        fallback_result = {
            "score": float(match.group()) if match else 0.5,
            "stage": "transition",
            "summary": text[:20],
        }
        self._cache.put(text, fallback_result)
        return fallback_result

    def _call_llm_raw(self, prompt: str, max_tokens: int = 80) -> str:
        """底層：呼叫 LLM 後端取得原始回應字串"""
        if self.llm_backend in ("openai", "groq"):
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        elif self.llm_backend == "ollama":
            import requests
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": max_tokens},
                },
                timeout=15,
            )
            return resp.json()["response"].strip()
        return "{}"

    def _query_llm(self, text: str) -> float:
        """向後兼容舊版介面（只回傳分數）"""
        result = self._query_llm_structured(text)
        return result.get("score", 0.5)

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


# ── 模組級工具函式 ──────────────────────────────────────────

def _parse_json_safe(raw: str) -> dict:
    """
    安全解析 LLM 輸出的 JSON 字串。
    LLM 有時會在 JSON 前後加 ```json ... ``` 或其他雜訊。
    """
    if not raw:
        return {}
    # 嘗試直接解析
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 嘗試提取 {} 區塊
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
