"""
學習素材產生器模組 (Study Materials Generator)
─────────────────────────────────────────────────────────────────────
本模組填補 KeyTake AI 與商業工具（如 YouLearn AI）之間的功能差距。
YouLearn AI 等商業產品能產生閃卡、學習筆記、心智圖等結構化學習素材，
而 KeyTake AI v3 僅產出剪輯影片。

透過 v4 新增的教學階段分析（teaching_stage），我們能自動將已分類的
片段（definition / derivation / example / summary / transition / qa）
轉化為多種結構化教育輸出：
  - 章節標題（Chapter Titles）
  - Markdown 學習筆記（Study Notes）
  - 心智圖結構（Mind Map）
  - Q&A 閃卡（Flashcards）

這讓 KeyTake AI 從「精華影片剪輯工具」升級為「完整學習輔助平台」，
不再只是一段短影片，而是一整組可用於複習的教材。

支援的 LLM 後端（與 SemanticScorer 相同邏輯）：
  - OpenAI（GPT-4o-mini / GPT-4o）
  - 本地 Ollama（Llama3 / Gemma）
  - Groq API（Llama3-8B-instant）
  - 無 LLM 時自動降級為規則式生成

對應計畫書 4.5 步驟五（輸出擴充）
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


# ── 章節標題 LLM 提示模板 ────────────────────────────────────
CHAPTER_TITLE_PROMPT = """你是教學影片結構分析師。以下是一組教學片段（已按時間排列），每個片段包含教學階段和摘要。

請為這組片段生成一個簡短的章節標題（10字以內），反映其教學主題。

片段資訊：
{segments_info}

只回答一個標題字串，不要引號或其他說明。"""


# ── 學習筆記 LLM 提示模板 ────────────────────────────────────
STUDY_NOTES_PROMPT = """你是教學筆記整理助手。根據以下章節資訊，請生成該章節的學習筆記段落。
要求：使用繁體中文，簡潔扼要，用重點條列式。

章節標題：{chapter_title}
片段摘要：
{summaries}

請用 Markdown 列點格式輸出重點筆記（3~5 點），不要重複標題。"""


# ── 閃卡 LLM 提示模板 ────────────────────────────────────────
FLASHCARD_PROMPT = """你是教學閃卡產生器。根據以下教學片段（教學階段為定義或推導），
請產生一張 Q&A 閃卡。

片段內容：「{text}」
教學階段：{stage}

以 JSON 格式回答：
{{"question": "<提問>", "answer": "<簡答>"}}

只回答 JSON，不要其他說明。"""


# ── 心智圖 LLM 提示模板 ────────────────────────────────────────
MIND_MAP_PROMPT = """你是教學心智圖設計師。根據以下課程資訊，建構心智圖結構。

課程標題：{title}
關鍵概念：{concepts}
章節列表：{chapters}

以 JSON 格式回答，格式如下：
{{
  "root": "<課程標題>",
  "children": [
    {{
      "label": "<概念1>",
      "children": [{{"label": "<相關片段/例子"}}]
    }}
  ]
}}

只回答 JSON，不要其他說明。"""


class StudyMaterialsGenerator:
    """
    學習素材產生器

    根據已完成教學階段分析的片段資料，產生多種結構化學習素材。
    支援 LLM 增強模式與純規則式降級模式。

    Args:
        llm_backend: LLM 後端選擇（openai / ollama / groq），
                     預設 None 則從環境變數自動偵測
    """

    def __init__(self, llm_backend: str = None):
        # 與 SemanticScorer 相同的 LLM 後端自動偵測邏輯
        self.llm_backend = llm_backend or os.getenv("LLM_BACKEND", "openai").lower()
        self.use_llm = self._has_llm_key()
        self.llm_client = None
        self.llm_model = None

        if self.use_llm:
            self._init_llm()
            print(f"[StudyMaterials] LLM 模式啟用（{self.llm_backend}）")
        else:
            print("[StudyMaterials] 規則式模式（無 LLM key）")

    def _has_llm_key(self) -> bool:
        """檢查是否有可用的 LLM API 金鑰"""
        if self.llm_backend == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        elif self.llm_backend == "groq":
            return bool(os.getenv("GROQ_API_KEY"))
        elif self.llm_backend == "ollama":
            return bool(os.getenv("OLLAMA_URL"))
        # 任何一個有設定就算有
        return bool(
            os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OLLAMA_URL")
        )

    def _init_llm(self):
        """初始化 LLM 客戶端（與 SemanticScorer 相同邏輯）"""
        if self.llm_backend == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY 未設定")
                self.llm_client = OpenAI(api_key=api_key)
                self.llm_model = "gpt-4o-mini"
            except Exception as e:
                print(f"[StudyMaterials] OpenAI 初始化失敗（{e}），降級為規則式")
                self.use_llm = False
        elif self.llm_backend == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY 未設定")
                self.llm_client = Groq(api_key=api_key)
                self.llm_model = "llama3-8b-8192"
            except Exception as e:
                print(f"[StudyMaterials] Groq 初始化失敗（{e}），降級為規則式")
                self.use_llm = False
        elif self.llm_backend == "ollama":
            try:
                import requests
                self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
                self.llm_model = os.getenv("OLLAMA_MODEL", "llama3")
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                if resp.status_code != 200:
                    raise ValueError("Ollama 伺服器未回應")
                self.llm_client = "ollama"
            except Exception as e:
                print(f"[StudyMaterials] Ollama 連線失敗（{e}），降級為規則式")
                self.use_llm = False

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """底層 LLM 呼叫，回傳原始回應字串"""
        if not self.use_llm:
            return ""

        if self.llm_backend in ("openai", "groq"):
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
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
                    "options": {"temperature": 0.3, "num_predict": max_tokens},
                },
                timeout=30,
            )
            return resp.json().get("response", "").strip()
        return ""

    # ─────────────────────────────────────────────────────────────────
    # 章節標題生成
    # ─────────────────────────────────────────────────────────────────

    def generate_chapter_titles(
        self, segments: list[dict], course_summary: dict
    ) -> list[dict]:
        """
        將連續片段依教學階段轉換分組，為每組生成章節標題。

        分組邏輯：當教學階段從一種類型轉換到另一種時，視為新章節的開始。
        連續相同階段的片段歸入同一章節。

        Args:
            segments:       已評分的片段列表（需含 teaching_stage, start, end）
            course_summary: generate_course_summary() 的輸出

        Returns:
            [{"chapter": int, "title": str, "start_sec": float,
              "end_sec": float, "segments": list}]
        """
        if not segments:
            return []

        # 按教學階段轉換點分組
        chapters = []
        current_group = [segments[0]]
        current_stage = segments[0].get("teaching_stage", "transition")

        for seg in segments[1:]:
            stage = seg.get("teaching_stage", "transition")
            if stage != current_stage:
                # 階段變化 → 結束當前分組
                chapters.append(current_group)
                current_group = [seg]
                current_stage = stage
            else:
                current_group.append(seg)

        # 別忘了最後一組
        if current_group:
            chapters.append(current_group)

        # 合併太短的章節（少於 2 個片段的過渡章節併入前一章）
        merged = []
        for group in chapters:
            if (
                len(group) < 2
                and group[0].get("teaching_stage") == "transition"
                and merged
            ):
                merged[-1].extend(group)
            else:
                merged.append(group)
        chapters = merged if merged else chapters

        # 為每個章節生成標題
        result = []
        for idx, group in enumerate(chapters, start=1):
            start_sec = group[0].get("start", 0.0)
            end_sec = group[-1].get("end", group[-1].get("start", 0.0))
            title = self._generate_single_chapter_title(group, course_summary)
            result.append({
                "chapter": idx,
                "title": title,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "segments": group,
            })

        return result

    def _generate_single_chapter_title(
        self, group: list[dict], course_summary: dict
    ) -> str:
        """為單一章節組生成標題"""
        if self.use_llm:
            # 組合片段摘要資訊給 LLM
            info_lines = []
            for seg in group[:10]:  # 最多取 10 個片段避免 token 過長
                stage = seg.get("teaching_stage", "unknown")
                summary = seg.get("segment_summary", seg.get("text", "")[:30])
                info_lines.append(f"- [{stage}] {summary}")
            segments_info = "\n".join(info_lines)
            prompt = CHAPTER_TITLE_PROMPT.format(segments_info=segments_info)
            title = self._call_llm(prompt, max_tokens=30)
            if title:
                # 清理 LLM 可能回傳的引號
                return title.strip().strip('"').strip("'")

        # 規則式降級：取主要教學階段 + 第一個片段摘要
        stage = group[0].get("teaching_stage", "transition")
        stage_labels = {
            "definition": "概念定義",
            "derivation": "公式推導",
            "example": "例題說明",
            "summary": "重點總結",
            "transition": "過渡",
            "qa": "問答互動",
        }
        label = stage_labels.get(stage, stage)
        first_summary = group[0].get("segment_summary", "")
        if first_summary:
            return f"{label}：{first_summary[:10]}"
        return label

    # ─────────────────────────────────────────────────────────────────
    # Markdown 學習筆記生成
    # ─────────────────────────────────────────────────────────────────

    def generate_study_notes(
        self, segments: list[dict], course_summary: dict
    ) -> str:
        """
        產生 Markdown 格式的學習筆記。

        結構：
          1. 課程標題
          2. 關鍵概念列表
          3. 各章節筆記
          4. 課程總結

        Args:
            segments:       已評分的片段列表
            course_summary: 全課摘要分析結果

        Returns:
            完整 Markdown 字串
        """
        title = course_summary.get("title", "教學影片筆記")
        summary_text = course_summary.get("summary", "")
        key_concepts = course_summary.get("key_concepts", [])

        # 生成章節結構
        chapters = self.generate_chapter_titles(segments, course_summary)

        # 組合 Markdown
        lines = []
        lines.append(f"# {title}")
        lines.append("")

        # 關鍵概念
        if key_concepts:
            lines.append("## 關鍵概念")
            lines.append("")
            for concept in key_concepts:
                lines.append(f"- {concept}")
            lines.append("")

        # 各章節筆記
        lines.append("## 課程筆記")
        lines.append("")

        for ch in chapters:
            lines.append(f"### 第 {ch['chapter']} 章：{ch['title']}")
            lines.append("")
            # 時間標記
            start_m, start_s = divmod(int(ch["start_sec"]), 60)
            end_m, end_s = divmod(int(ch["end_sec"]), 60)
            lines.append(
                f"*時間：{start_m:02d}:{start_s:02d} ~ {end_m:02d}:{end_s:02d}*"
            )
            lines.append("")

            # 章節內容：使用 LLM 或規則式
            chapter_notes = self._generate_chapter_notes(ch)
            lines.append(chapter_notes)
            lines.append("")

        # 課程總結
        if summary_text:
            lines.append("## 總結")
            lines.append("")
            lines.append(summary_text)
            lines.append("")

        return "\n".join(lines)

    def _generate_chapter_notes(self, chapter: dict) -> str:
        """為單一章節生成筆記內容"""
        group = chapter.get("segments", [])
        summaries = [
            seg.get("segment_summary", seg.get("text", "")[:40])
            for seg in group
            if seg.get("teaching_stage") != "transition"
        ]

        if self.use_llm and summaries:
            prompt = STUDY_NOTES_PROMPT.format(
                chapter_title=chapter.get("title", ""),
                summaries="\n".join(f"- {s}" for s in summaries[:10]),
            )
            notes = self._call_llm(prompt, max_tokens=300)
            if notes:
                return notes

        # 規則式降級：直接列出片段摘要
        if not summaries:
            return "- （過渡片段，無重點內容）"
        result_lines = []
        for s in summaries[:8]:
            if s.strip():
                result_lines.append(f"- {s}")
        return "\n".join(result_lines) if result_lines else "- （無摘要）"

    # ─────────────────────────────────────────────────────────────────
    # 心智圖生成
    # ─────────────────────────────────────────────────────────────────

    def generate_mind_map(
        self, segments: list[dict], course_summary: dict
    ) -> dict:
        """
        產生心智圖 JSON 樹狀結構。

        結構：
          - root: 課程標題
          - Level 1 children: 關鍵概念
          - Level 2 children: 各概念相關的片段/例子

        可由前端渲染為互動心智圖，或匯出為 Mermaid / PlantUML 格式。

        Args:
            segments:       已評分的片段列表
            course_summary: 全課摘要分析結果

        Returns:
            {"root": str, "children": [{"label": str, "children": [...]}]}
        """
        title = course_summary.get("title", "教學影片")
        key_concepts = course_summary.get("key_concepts", [])
        chapters = self.generate_chapter_titles(segments, course_summary)

        if self.use_llm and key_concepts:
            # 嘗試用 LLM 生成更好的心智圖結構
            concepts_str = ", ".join(key_concepts)
            chapters_str = ", ".join(ch["title"] for ch in chapters[:10])
            prompt = MIND_MAP_PROMPT.format(
                title=title,
                concepts=concepts_str,
                chapters=chapters_str,
            )
            raw = self._call_llm(prompt, max_tokens=500)
            result = _parse_json_safe(raw)
            if result and "root" in result:
                return result

        # 規則式降級：從 key_concepts 和章節片段建構樹
        children = []
        if key_concepts:
            # 每個關鍵概念作為一級節點
            for concept in key_concepts:
                # 找與概念相關的片段（簡易文字比對）
                related = []
                for seg in segments:
                    seg_text = seg.get("segment_summary", "") + seg.get("text", "")
                    if concept in seg_text and seg.get("teaching_stage") != "transition":
                        related.append({
                            "label": seg.get("segment_summary", seg.get("text", "")[:20])
                        })
                        if len(related) >= 3:
                            break
                children.append({
                    "label": concept,
                    "children": related if related else [{"label": "（參見課程內容）"}],
                })
        else:
            # 無 key_concepts 時，用章節作為一級節點
            for ch in chapters:
                ch_children = []
                for seg in ch.get("segments", [])[:3]:
                    if seg.get("teaching_stage") != "transition":
                        ch_children.append({
                            "label": seg.get("segment_summary", "")[:20]
                        })
                children.append({
                    "label": ch["title"],
                    "children": ch_children,
                })

        return {
            "root": title,
            "children": children,
        }

    # ─────────────────────────────────────────────────────────────────
    # Q&A 閃卡生成
    # ─────────────────────────────────────────────────────────────────

    def generate_flashcards(self, segments: list[dict]) -> list[dict]:
        """
        從 definition 和 derivation 類型的片段生成 Q&A 閃卡。

        每張閃卡包含一個問題和答案，並標記來源時間點。
        適合匯出至 Anki 或其他閃卡學習工具。

        Args:
            segments: 已評分的片段列表（需含 teaching_stage, text, start）

        Returns:
            [{"question": str, "answer": str, "source_time": str}]
        """
        # 篩選適合做閃卡的片段（定義與推導）
        candidates = [
            seg for seg in segments
            if seg.get("teaching_stage") in ("definition", "derivation")
            and len(seg.get("text", "")) > 10
        ]

        flashcards = []
        for seg in candidates:
            text = seg.get("text", "").strip()
            stage = seg.get("teaching_stage", "definition")
            start = seg.get("start", 0)
            # 格式化時間
            m, s = divmod(int(start), 60)
            source_time = f"{m:02d}:{s:02d}"

            card = self._generate_single_flashcard(text, stage, source_time)
            if card:
                flashcards.append(card)

        return flashcards

    def _generate_single_flashcard(
        self, text: str, stage: str, source_time: str
    ) -> Optional[dict]:
        """為單一片段生成一張閃卡"""
        if self.use_llm:
            prompt = FLASHCARD_PROMPT.format(text=text[:300], stage=stage)
            raw = self._call_llm(prompt, max_tokens=150)
            result = _parse_json_safe(raw)
            if result and "question" in result and "answer" in result:
                result["source_time"] = source_time
                return result

        # 規則式降級：從片段文字自動生成問答
        summary = text[:60].strip()
        if stage == "definition":
            # 定義型：「什麼是 X？」
            question = f"什麼是「{summary[:15]}」所定義的概念？"
            answer = summary
        elif stage == "derivation":
            # 推導型：「如何推導 X？」
            question = f"請說明以下推導的關鍵步驟：{summary[:15]}"
            answer = summary
        else:
            question = f"請解釋：{summary[:20]}"
            answer = summary

        return {
            "question": question,
            "answer": answer,
            "source_time": source_time,
        }

    # ─────────────────────────────────────────────────────────────────
    # 批次匯出所有素材
    # ─────────────────────────────────────────────────────────────────

    def export_all(
        self, segments: list[dict], course_summary: dict, output_dir: str
    ) -> dict:
        """
        一次產生所有學習素材並存檔至指定目錄。

        產出檔案：
          - chapters.json   : 章節標題結構
          - study_notes.md  : Markdown 學習筆記
          - mind_map.json   : 心智圖 JSON 結構
          - flashcards.json : Q&A 閃卡列表

        Args:
            segments:       已評分的片段列表
            course_summary: 全課摘要分析結果
            output_dir:     輸出目錄路徑

        Returns:
            {"chapters": str, "study_notes": str,
             "mind_map": str, "flashcards": str}
             各 value 為檔案完整路徑
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 生成章節標題
        chapters = self.generate_chapter_titles(segments, course_summary)
        # 序列化時移除 segments 欄位（避免 JSON 過大）
        chapters_export = [
            {k: v for k, v in ch.items() if k != "segments"}
            for ch in chapters
        ]
        chapters_file = out_path / "chapters.json"
        chapters_file.write_text(
            json.dumps(chapters_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 生成學習筆記
        notes = self.generate_study_notes(segments, course_summary)
        notes_file = out_path / "study_notes.md"
        notes_file.write_text(notes, encoding="utf-8")

        # 生成心智圖
        mind_map = self.generate_mind_map(segments, course_summary)
        mind_map_file = out_path / "mind_map.json"
        mind_map_file.write_text(
            json.dumps(mind_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 生成閃卡
        flashcards = self.generate_flashcards(segments)
        flashcards_file = out_path / "flashcards.json"
        flashcards_file.write_text(
            json.dumps(flashcards, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[StudyMaterials] 已匯出 {len(chapters)} 章節、"
              f"{len(flashcards)} 張閃卡至 {output_dir}")

        return {
            "chapters": str(chapters_file),
            "study_notes": str(notes_file),
            "mind_map": str(mind_map_file),
            "flashcards": str(flashcards_file),
        }


# ── 模組級工具函式 ──────────────────────────────────────────────

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
