"""
SemanticScorer v3 — Unit Tests
──────────────────────────────────────────────────────────────────
驗證三種評分模式（LLM / SBERT / TF-IDF）的行為：
  1. TF-IDF 模式基本正確性
  2. SBERT 模式分數範圍
  3. LLM 模式：mock API 回應，確認分數解析正確
  4. LLM 失敗時的降級行為
  5. 邊界條件（空片段、單片段）
"""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_CORPUS = [
    "這邊非常重要", "請注意這個公式",
    "導致這個結果的原因是", "總結一下今天的重點",
]


def make_segments(texts: list[str]) -> list[dict]:
    segs = []
    t = 0.0
    for text in texts:
        segs.append({"start": t, "end": t + 10.0, "text": text})
        t += 10.0
    return segs


class TestTFIDFMode:

    def test_tfidf_scores_in_range(self):
        from src.semantic.scorer import SemanticScorer
        scorer = SemanticScorer(SAMPLE_CORPUS, mode="tfidf")
        segs = make_segments(["這是第一段", "這是第二段", "重要公式推導", "閒聊"])
        result = scorer.score(segs)
        for seg in result:
            assert 0.0 <= seg["s_text"] <= 1.0

    def test_tfidf_single_segment(self):
        from src.semantic.scorer import SemanticScorer
        scorer = SemanticScorer(SAMPLE_CORPUS, mode="tfidf")
        segs = make_segments(["只有一段"])
        result = scorer.score(segs)
        assert "s_text" in result[0]

    def test_tfidf_empty_text_gives_zero(self):
        from src.semantic.scorer import SemanticScorer
        scorer = SemanticScorer(SAMPLE_CORPUS, mode="tfidf")
        segs = [{"start": 0, "end": 5, "text": ""}]
        result = scorer.score(segs)
        assert result[0]["s_text"] == 0.0


class TestLLMMode:

    @pytest.fixture(autouse=True)
    def _isolated_llm_cache(self, tmp_path):
        """為每個 LLM 測試使用獨立的臨時快取，避免讀到或污染正式快取，
        確保測試結果具決定性。"""
        from src.semantic import llm_cache as _llm_cache_mod
        original_init = _llm_cache_mod.LLMCache.__init__

        def _patched_init(self, cache_dir="cache", max_age_days=None, flush_every=10):
            original_init(self, cache_dir=str(tmp_path / "cache"),
                          max_age_days=max_age_days, flush_every=flush_every)

        with patch.object(_llm_cache_mod.LLMCache, "__init__", _patched_init):
            yield

    def _make_mock_completion(self, score_str: str):
        mock_choice = MagicMock()
        mock_choice.message.content = score_str
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_llm_parses_float_correctly(self):
        """LLM 回傳 '0.85' 時應解析為 0.85"""
        from src.semantic.scorer import SemanticScorer

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            scorer = SemanticScorer(SAMPLE_CORPUS, mode="llm", llm_backend="openai")
            # 替換 LLM client
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = \
                self._make_mock_completion("0.85")
            scorer.llm_client = mock_client

            segs = make_segments(["這是非常重要的公式推導"])
            result = scorer.score(segs)
            assert abs(result[0]["s_text"] - 0.85) < 0.01

    def test_llm_clamps_score_above_1(self):
        """LLM 若回傳超過 1 的數字，應夾到 1.0"""
        from src.semantic.scorer import SemanticScorer

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            scorer = SemanticScorer(SAMPLE_CORPUS, mode="llm", llm_backend="openai")
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = \
                self._make_mock_completion("1.5")
            scorer.llm_client = mock_client

            segs = make_segments(["測試片段"])
            result = scorer.score(segs)
            assert result[0]["s_text"] <= 1.0

    def test_llm_fallback_on_api_error(self):
        """LLM API 拋出例外時，分數應為 0.5（中性）"""
        from src.semantic.scorer import SemanticScorer

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            scorer = SemanticScorer(SAMPLE_CORPUS, mode="llm", llm_backend="openai")
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            scorer.llm_client = mock_client

            segs = make_segments(["這段會讓 API 失敗"])
            result = scorer.score(segs)
            assert result[0]["s_text"] == 0.5

    def test_llm_handles_empty_text(self):
        """空文字應直接跳過，給 0 分"""
        from src.semantic.scorer import SemanticScorer

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            scorer = SemanticScorer(SAMPLE_CORPUS, mode="llm", llm_backend="openai")
            mock_client = MagicMock()
            scorer.llm_client = mock_client

            segs = [{"start": 0, "end": 5, "text": ""}]
            result = scorer.score(segs)
            # 空文字不應呼叫 LLM
            mock_client.chat.completions.create.assert_not_called()
            assert result[0]["s_text"] == 0.0

    def test_llm_parses_integer_1(self):
        """LLM 回傳 '1' 時應解析為 1.0"""
        from src.semantic.scorer import SemanticScorer

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            scorer = SemanticScorer(SAMPLE_CORPUS, mode="llm", llm_backend="openai")
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = \
                self._make_mock_completion("1")
            scorer.llm_client = mock_client

            segs = make_segments(["這是非常重要的一段課堂內容"])
            result = scorer.score(segs)
            assert result[0]["s_text"] == 1.0

    def test_llm_parses_garbled_response(self):
        """LLM 回傳無法解析的文字時，應給中性分數 0.5"""
        from src.semantic.scorer import SemanticScorer

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            scorer = SemanticScorer(SAMPLE_CORPUS, mode="llm", llm_backend="openai")
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = \
                self._make_mock_completion("這段課程內容很重要喔！")
            scorer.llm_client = mock_client

            segs = make_segments(["這是一個需要 LLM 評分的教學片段"])
            result = scorer.score(segs)
            assert result[0]["s_text"] == 0.5


class TestModeAutoDetect:

    def test_no_api_key_uses_sbert(self):
        """沒有 API key 時，應降級為 sbert 模式"""
        from src.semantic.scorer import SemanticScorer

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("OPENAI_API_KEY", "GROQ_API_KEY", "OLLAMA_URL",
                                  "SEMANTIC_SCORER_MODE")}
        with patch.dict(os.environ, clean_env, clear=True):
            scorer = SemanticScorer(SAMPLE_CORPUS)
            assert scorer.mode == "sbert"

    def test_explicit_mode_overrides_env(self):
        """明確指定 mode 應覆蓋環境變數"""
        from src.semantic.scorer import SemanticScorer

        with patch.dict(os.environ, {"SEMANTIC_SCORER_MODE": "llm"}):
            scorer = SemanticScorer(SAMPLE_CORPUS, mode="tfidf")
            assert scorer.mode == "tfidf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
