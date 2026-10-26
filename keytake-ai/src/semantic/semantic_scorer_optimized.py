"""語意評分模組：支援 SBERT、LLM 與 TF-IDF 三種模式。"""

import json
import os
import re
import time
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer, util

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    util = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import SBERT_MODEL, TFIDF_TOP_K, SBERT_SIMILARITY_THRESHOLD
from src.semantic.llm_cache import LLMCache


TEACHING_STAGE_TYPES = {
    "definition", "derivation", "example", "summary", "transition", "qa"
}

STAGE_LABELS = {
    "definition": "概念定義",
    "derivation": "公式推導",
    "example": "例題說明",
    "summary": "重點總結",
    "transition": "過渡換場",
    "qa": "問答互動",
}

STAGE_TO_SEG_TYPE = {
    "definition": "semantic",
    "derivation": "combined",
    "example": "visual",
    "summary": "combined",
    "transition": "semantic",
    "qa": "semantic",
}

STAGE_WEIGHT = {
    "definition": 1.1,
    "derivation": 1.2,
    "example": 1.0,
    "summary": 1.1,
    "transition": 0.3,
    "qa": 0.8,
}

LLM_PROMPT_TEMPLATE = """你是教學影片智能分析助手。請分析以下課堂逐字稿片段，並只輸出 JSON：
{{
  "score": <0到1之間的教學重要性分數>,
  "stage": "<definition/derivation/example/summary/transition/qa>",
  "summary": "<用一句話概括重點，15字以內>"
}}

分類說明：
- definition：定義核心概念、術語解釋
- derivation：公式推導、定理證明、步驟演算
- example：舉例說明、解題示範
- summary：總結回顧、重點整理
- transition：換場過渡語、行政公告、寒暄
- qa：師生問答互動

逐字稿：
「{text}」
"""

COURSE_SUMMARY_PROMPT = """你是教學影片課程分析師。請根據以下課堂逐字稿，只輸出 JSON：
{{
  "title": "<課程主題>",
  "summary": "<2至3句話描述核心內容>",
  "key_concepts": ["<概念1>", "<概念2>"],
  "structure": [
    {{"stage": "introduction", "description": "<引入部分>"}},
    {{"stage": "main_content", "description": "<主要內容>"}},
    {{"stage": "conclusion", "description": "<總結部分>"}}
  ]
}}

課堂內容：
{transcript}
"""


class SemanticScorer:
    """對轉錄片段評分，並補上教學階段與片段摘要。"""

    def __init__(
        self,
        prompt_corpus: list[str],
        mode: Optional[str] = None,
        llm_backend: Optional[str] = None,
    ):
        self.prompt_corpus = [
            str(item).strip() for item in prompt_corpus if str(item).strip()
        ]
        self.mode = (mode or os.getenv("SEMANTIC_SCORER_MODE", "sbert")).lower()
        self.llm_backend = (
            llm_backend or os.getenv("LLM_BACKEND", "openai")
        ).lower()
        self._cache = LLMCache()
        self.sbert = None
        self.corpus_embeddings = None
        self.llm_client = None
        self.llm_model = None
        self.ollama_url = None

        if self.mode not in {"sbert", "llm", "tfidf"}:
            raise ValueError("mode 必須是 sbert、llm 或 tfidf")

        if self.mode == "sbert" and self._has_llm_key():
            self.mode = "llm"
            print("[SemanticScorer] 偵測到 LLM 設定，切換至 LLM 模式")

        if self.mode == "llm":
            self._init_llm()
        elif self.mode == "sbert":
            self._init_sbert()

        print(f"[SemanticScorer] 模式：{self.mode}")

    @staticmethod
    def _has_llm_key() -> bool:
        return bool(
            os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OLLAMA_URL")
        )

    def _fallback_from_llm(self, reason: Exception) -> None:
        print(f"[SemanticScorer] LLM 初始化失敗：{reason}")
        self._init_sbert()

    def _init_llm(self) -> None:
        try:
            if self.llm_backend == "openai":
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("未設定 OPENAI_API_KEY")
                self.llm_client = OpenAI(api_key=api_key)
                self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            elif self.llm_backend == "groq":
                from groq import Groq

                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("未設定 GROQ_API_KEY")
                self.llm_client = Groq(api_key=api_key)
                self.llm_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

            elif self.llm_backend == "ollama":
                import requests

                self.ollama_url = os.getenv(
                    "OLLAMA_URL", "http://localhost:11434"
                ).rstrip("/")
                self.llm_model = os.getenv("OLLAMA_MODEL", "llama3")
                response = requests.get(
                    f"{self.ollama_url}/api/tags", timeout=3
                )
                response.raise_for_status()
                self.llm_client = "ollama"
            else:
                raise ValueError(
                    "LLM_BACKEND 必須是 openai、groq 或 ollama"
                )
        except Exception as exc:
            self._fallback_from_llm(exc)

    def _init_sbert(self) -> None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("[SemanticScorer] sentence-transformers 未安裝，改用 TF-IDF")
            self.mode = "tfidf"
            return

        try:
            if not self.prompt_corpus:
                raise ValueError("SBERT 模式需要非空的 prompt_corpus")
            self.sbert = SentenceTransformer(SBERT_MODEL)
            self.corpus_embeddings = self.sbert.encode(
                self.prompt_corpus,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.mode = "sbert"
        except Exception as exc:
            print(f"[SemanticScorer] SBERT 初始化失敗，改用 TF-IDF：{exc}")
            self.sbert = None
            self.corpus_embeddings = None
            self.mode = "tfidf"

    def score(
        self, segments: list[dict], alpha_tfidf: float = 0.5
    ) -> list[dict]:
        """評分片段並原地加入 s_text、teaching_stage、stage_label、segment_summary。"""
        if not segments:
            return segments

        texts = [str(seg.get("text", "") or "").strip() for seg in segments]

        if self.mode == "llm":
            scores, stages, summaries = self._llm_scores(segments)
        elif self.mode == "sbert":
            alpha = float(np.clip(alpha_tfidf, 0.0, 1.0))
            tfidf = self.tfidf_scores(segments)
            semantic = self.sbert_scores(segments)
            scores = alpha * tfidf + (1.0 - alpha) * semantic
            stages = ["semantic"] * len(segments)
            summaries = [text[:20] for text in texts]
        else:
            scores = self.tfidf_scores(segments)
            stages = ["semantic"] * len(segments)
            summaries = [text[:20] for text in texts]

        for i, segment in enumerate(segments):
            score = scores[i] if i < len(scores) else 0.5
            segment["s_text"] = float(np.clip(score, 0.0, 1.0))
            stage = stages[i] if i < len(stages) else "transition"
            segment["teaching_stage"] = stage
            segment["stage_label"] = STAGE_LABELS.get(stage, stage)
            segment["segment_summary"] = (
                summaries[i] if i < len(summaries) else ""
            )
        return segments

    def generate_course_summary(self, segments: list[dict]) -> dict:
        """產生全課摘要；非 LLM 模式以高分片段組成簡易結果。"""
        if self.mode != "llm":
            top_segments = sorted(
                segments,
                key=lambda item: float(item.get("s_text", 0) or 0),
                reverse=True,
            )[:5]
            return {
                "title": "教學影片摘要",
                "summary": " ".join(
                    str(item.get("text", ""))[:30]
                    for item in top_segments[:3]
                ),
                "key_concepts": [
                    item.get("segment_summary", "")
                    for item in top_segments
                    if item.get("segment_summary")
                ],
                "structure": [],
            }

        lines = []
        for segment in segments[:30]:
            try:
                total_seconds = max(0, int(float(segment.get("start", 0))))
            except (TypeError, ValueError):
                total_seconds = 0
            minutes, seconds = divmod(total_seconds, 60)
            text = str(segment.get("text", "") or "").strip()
            if text:
                lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")

        if not lines:
            return self._empty_course_summary()

        try:
            raw = self._call_llm_raw(
                COURSE_SUMMARY_PROMPT.format(transcript="\n".join(lines)),
                max_tokens=400,
            )
            result = _parse_json_safe(raw)
            if result.get("title"):
                result.setdefault("summary", "")
                result.setdefault("key_concepts", [])
                result.setdefault("structure", [])
                return result
        except Exception as exc:
            print(f"[SemanticScorer] 全課分析失敗：{exc}")

        return self._empty_course_summary()

    @staticmethod
    def _empty_course_summary() -> dict:
        return {
            "title": "教學影片摘要",
            "summary": "",
            "key_concepts": [],
            "structure": [],
        }

    def _llm_scores(
        self, segments: list[dict]
    ) -> tuple[np.ndarray, list[str], list[str]]:
        scores = np.full(len(segments), 0.5, dtype=float)
        stages = ["transition"] * len(segments)
        summaries = [""] * len(segments)

        for i, segment in enumerate(segments):
            text = str(segment.get("text", "") or "").strip()
            if not text:
                continue
            if len(text) < 5:
                summaries[i] = text
                continue

            try:
                result = self._query_llm_structured(text)
                raw_score = self._safe_score(result.get("score", 0.5))
                stage = result.get("stage")
                if stage not in TEACHING_STAGE_TYPES:
                    stage = "transition"
                scores[i] = float(
                    np.clip(raw_score * STAGE_WEIGHT.get(stage, 1.0), 0, 1)
                )
                stages[i] = stage
                summary = result.get("summary", text[:20])
                summaries[i] = str(summary).strip()[:100] or text[:20]
            except Exception as exc:
                print(f"[SemanticScorer] 片段 {i} 評分失敗：{exc}")
                scores[i] = 0.5
                stages[i] = "transition"
                summaries[i] = text[:20]

        self._cache.flush()
        return scores, stages, summaries

    @staticmethod
    def _safe_score(value) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.5
        return float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.5

    def _query_llm_structured(self, text: str) -> dict:
        """先查快取，再呼叫 LLM；只快取已驗證的結構化結果。"""
        cached = self._cache.get(text)
        if isinstance(cached, dict) and "score" in cached:
            result = dict(cached)
            result["score"] = self._safe_score(result["score"])
            if result.get("stage") not in TEACHING_STAGE_TYPES:
                result["stage"] = "transition"
            return result

        prompt = LLM_PROMPT_TEMPLATE.format(text=text[:400])
        raw = self._call_llm_raw(prompt, max_tokens=120)
        result = _parse_json_safe(raw)

        if "score" not in result:
            match = re.search(r"(?<![\d.])(?:0?\.\d+|1(?:\.0+)?|0)(?![\d.])", raw or "")
            result = {"score": float(match.group()) if match else 0.5}

        result["score"] = self._safe_score(result.get("score", 0.5))
        if result.get("stage") not in TEACHING_STAGE_TYPES:
            result["stage"] = "transition"
        summary = result.get("summary", text[:20])
        result["summary"] = str(summary).strip()[:100] or text[:20]
        self._cache.put(text, result)
        return result

    def _call_llm_raw(self, prompt: str, max_tokens: int = 80) -> str:
        """呼叫目前設定的 LLM 後端。"""
        if self.llm_backend in {"openai", "groq"}:
            if self.llm_client is None:
                raise RuntimeError("LLM 客戶端尚未初始化")
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""

        if self.llm_backend == "ollama":
            import requests

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": max_tokens,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()

        raise RuntimeError(f"不支援的 LLM 後端：{self.llm_backend}")

    def _query_llm(self, text: str) -> float:
        """相容舊版介面，回傳單一重要性分數。"""
        return self._safe_score(self._query_llm_structured(text).get("score", 0.5))

    def tfidf_scores(self, segments: list[dict]) -> np.ndarray:
        """以 TF-IDF 特徵總量估計片段關鍵詞程度。"""
        texts = [str(seg.get("text", "") or "").strip() for seg in segments]
        if not texts or not any(texts):
            return np.zeros(len(texts), dtype=float)

        try:
            vectorizer = TfidfVectorizer(max_features=TFIDF_TOP_K)
            matrix = vectorizer.fit_transform(texts)
            scores = np.asarray(matrix.sum(axis=1)).ravel()
            maximum = float(scores.max()) if scores.size else 0.0
            return scores / maximum if maximum > 0 else np.zeros(len(texts))
        except ValueError:
            return np.zeros(len(texts), dtype=float)

    def sbert_scores(self, segments: list[dict]) -> np.ndarray:
        """批次計算片段與提示語料庫的最大餘弦相似度。"""
        if self.sbert is None or self.corpus_embeddings is None:
            return np.zeros(len(segments), dtype=float)

        texts = [str(seg.get("text", "") or "").strip() for seg in segments]
        scores = np.zeros(len(texts), dtype=float)
        valid_indices = [i for i, text in enumerate(texts) if text]
        if not valid_indices:
            return scores

        embeddings = self.sbert.encode(
            [texts[i] for i in valid_indices],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = util.cos_sim(embeddings, self.corpus_embeddings)
        maxima = similarities.max(dim=1).values.detach().cpu().numpy()
        for index, value in zip(valid_indices, maxima):
            scores[index] = float(np.clip(value, 0.0, 1.0))
        return scores


def _parse_json_safe(raw: str) -> dict:
    """解析 LLM JSON；允許 Markdown 程式碼區塊或純數字回應。"""
    if not raw or not isinstance(raw, str):
        return {}

    def normalize(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"score": float(value)}
        return {}

    try:
        result = normalize(json.loads(raw.strip()))
        if result:
            return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return normalize(json.loads(match.group()))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    match = re.search(r"(?<![\d.])(?:0?\.\d+|1(?:\.0+)?|0)(?![\d.])", raw)
    if match:
        try:
            return {"score": float(match.group())}
        except ValueError:
            pass
    return {}
