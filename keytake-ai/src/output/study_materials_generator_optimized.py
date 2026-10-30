"""Generate structured study materials from analyzed teaching segments."""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional


CHAPTER_TITLE_PROMPT = """你是教學影片結構分析師。以下是一組依時間排列的教學片段，每個片段包含教學階段和摘要。
請產生一個簡短、準確的繁體中文章節標題，最多 10 個中文字。
片段資訊：
{segments_info}
只回答標題，不要引號或其他說明。"""

STUDY_NOTES_PROMPT = """你是教學筆記整理助手。請根據提供的章節資訊，以繁體中文產生 3 至 5 點 Markdown 條列筆記。
只整理來源中有根據的內容，不要補入未提及的事實，不要重複章節標題。
章節標題：{chapter_title}
片段摘要：
{summaries}
只輸出 Markdown 條列。"""

FLASHCARD_PROMPT = """你是教學閃卡產生器。根據下列教學片段，產生一張有助於理解與回憶的問答閃卡。
問題應明確且不能直接洩漏答案；答案應簡潔，並忠於來源內容。使用繁體中文。
教學階段：{stage}
片段內容：{text}
只輸出 JSON，格式為：{{"question": "問題", "answer": "答案"}}"""

MIND_MAP_PROMPT = """根據課程資訊產生繁體中文心智圖 JSON。只使用提供的概念與章節，不要自行杜撰。
課程標題：{title}
關鍵概念：{concepts}
章節列表：{chapters}
格式：
{{
  "root": "課程標題",
  "children": [
    {{
      "label": "概念",
      "children": [{{"label": "相關章節或例子"}}]
    }}
  ]
}}
只輸出有效 JSON,不要 Markdown 程式碼區塊或其他說明。"""


class StudyMaterialsGenerator:
    """Generate chapter titles, study notes, mind maps, and flashcards."""

    STAGE_LABELS = {
        "definition": "概念定義",
        "derivation": "公式推導",
        "example": "例題說明",
        "summary": "重點總結",
        "transition": "過渡",
        "qa": "問答互動",
    }

    def __init__(self, llm_backend: Optional[str] = None):
        self.llm_backend = (llm_backend or os.getenv("LLM_BACKEND", "openai")).lower()
        materials_mode = os.getenv("STUDY_MATERIALS_MODE")
        if materials_mode is not None:
            self.mode = materials_mode.strip().lower()
            if self.mode not in {"llm", "rules"}:
                raise ValueError("STUDY_MATERIALS_MODE 必須是 llm 或 rules")
        else:
            self.mode = os.getenv("SEMANTIC_SCORER_MODE", "sbert").strip().lower()
            if self.mode not in {"llm", "sbert", "tfidf"}:
                raise ValueError(
                    "SEMANTIC_SCORER_MODE 必須是 llm、sbert 或 tfidf"
                )
        self.use_llm = False
        self.llm_client = None
        self.llm_model = None
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        if self.mode == "llm":
            self._init_llm()

        mode = f"LLM 模式（{self.llm_backend}）" if self.use_llm else "規則式模式"
        print(f"[StudyMaterials] {mode}")

    def _init_llm(self) -> None:
        try:
            if self.llm_backend == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return
                from openai import OpenAI

                self.llm_client = OpenAI(
                    api_key=api_key,
                    timeout=float(os.getenv("LLM_TIMEOUT_SEC", "60")),
                )
                self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            elif self.llm_backend == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    return
                from groq import Groq

                self.llm_client = Groq(
                    api_key=api_key,
                    timeout=float(os.getenv("LLM_TIMEOUT_SEC", "60")),
                )
                self.llm_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

            elif self.llm_backend == "ollama":
                import requests

                response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                response.raise_for_status()
                self.llm_model = os.getenv("OLLAMA_MODEL", "llama3")
                self.llm_client = "ollama"
            else:
                print(f"[StudyMaterials] 不支援的 LLM_BACKEND:{self.llm_backend}")
                return

            self.use_llm = True
        except Exception as exc:
            self.llm_client = None
            self.use_llm = False
            print(f"[StudyMaterials] LLM 初始化失敗，改用規則式生成：{exc}")

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        if not self.use_llm:
            return ""

        try:
            if self.llm_backend in ("openai", "groq"):
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3,
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
                        "options": {"temperature": 0.3, "num_predict": max_tokens},
                    },
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                return payload.get("response", "").strip()

        except Exception as exc:
            print(f"[StudyMaterials] LLM 呼叫失敗，改用規則式生成：{exc}")
            self.use_llm = False
            self.llm_client = None

        return ""

    @staticmethod
    def _segment_text(segment: dict, limit: Optional[int] = None) -> str:
        text = segment.get("segment_summary") or segment.get("text") or ""
        text = str(text).strip()
        return text[:limit] if limit is not None else text

    @staticmethod
    def _format_time(seconds: Any) -> str:
        try:
            total = max(0, int(float(seconds or 0)))
        except (TypeError, ValueError):
            total = 0
        minutes, secs = divmod(total, 60)
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _clean_title(title: str, limit: int = 10) -> str:
        title = str(title or "").strip().strip("\"'「」『』`")
        title = re.sub(r"[\r\n]+", " ", title)
        title = re.sub(r"^(章節標題|標題)\s*[:：]\s*", "", title)
        return title[:limit].strip()

    @staticmethod
    def _valid_mind_map(data: Any) -> bool:
        if not isinstance(data, dict) or not isinstance(data.get("root"), str):
            return False
        children = data.get("children")
        if not isinstance(children, list):
            return False
        for node in children:
            if not isinstance(node, dict) or not isinstance(node.get("label"), str):
                return False
            nested = node.get("children", [])
            if not isinstance(nested, list):
                return False
            if any(not isinstance(item, dict) or not isinstance(item.get("label"), str)
                   for item in nested):
                return False
        return True

    def generate_chapter_titles(
        self, segments: list[dict], course_summary: dict
    ) -> list[dict]:
        if not segments:
            return []

        groups: list[list[dict]] = []
        current_group = [segments[0]]
        current_stage = segments[0].get("teaching_stage", "transition")

        for segment in segments[1:]:
            stage = segment.get("teaching_stage", "transition")
            if stage == current_stage:
                current_group.append(segment)
            else:
                groups.append(current_group)
                current_group = [segment]
                current_stage = stage
        groups.append(current_group)

        merged: list[list[dict]] = []
        pending_transition: list[dict] = []
        for group in groups:
            is_short_transition = (
                len(group) < 2
                and all(item.get("teaching_stage", "transition") == "transition" for item in group)
            )
            if is_short_transition:
                if merged:
                    merged[-1].extend(group)
                else:
                    pending_transition.extend(group)
                continue
            if pending_transition:
                group = pending_transition + group
                pending_transition = []
            merged.append(group)
        if pending_transition:
            if merged:
                merged[-1].extend(pending_transition)
            else:
                merged.append(pending_transition)

        chapters = []
        for index, group in enumerate(merged, start=1):
            start = self._safe_float(group[0].get("start", 0))
            end = self._safe_float(group[-1].get("end", group[-1].get("start", start)))
            if end < start:
                end = start
            chapters.append({
                "chapter": index,
                "title": self._generate_single_chapter_title(group, course_summary),
                "start_sec": start,
                "end_sec": end,
                "segments": group,
            })
        return chapters

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _generate_single_chapter_title(self, group: list[dict], course_summary: dict) -> str:
        if self.use_llm:
            lines = []
            for segment in group[:10]:
                stage = segment.get("teaching_stage", "unknown")
                summary = self._segment_text(segment, 100)
                if summary:
                    lines.append(f"- [{stage}] {summary}")
            if lines:
                prompt = CHAPTER_TITLE_PROMPT.format(segments_info="\n".join(lines))
                title = self._clean_title(self._call_llm(prompt, max_tokens=30))
                if title:
                    return title

        stage = group[0].get("teaching_stage", "transition")
        label = self.STAGE_LABELS.get(stage, str(stage))
        summary = self._segment_text(group[0], 10)
        return f"{label}：{summary}" if summary else label

    def generate_study_notes(self, segments: list[dict], course_summary: dict) -> str:
        title = str(course_summary.get("title") or "教學影片筆記")
        summary_text = str(course_summary.get("summary") or "").strip()
        concepts = course_summary.get("key_concepts") or []
        chapters = self.generate_chapter_titles(segments, course_summary)

        lines = [f"# {title}", ""]
        if concepts:
            lines.extend(["## 關鍵概念", ""])
            lines.extend(f"- {concept}" for concept in concepts if str(concept).strip())
            lines.append("")

        lines.extend(["## 課程筆記", ""])
        if not chapters:
            lines.extend(["目前沒有可用的教學片段。", ""])
        for chapter in chapters:
            lines.extend([
                f"### 第 {chapter['chapter']} 章：{chapter['title']}",
                "",
                f"*時間：{self._format_time(chapter['start_sec'])} ~ "
                f"{self._format_time(chapter['end_sec'])}*",
                "",
                self._generate_chapter_notes(chapter),
                "",
            ])

        if summary_text:
            lines.extend(["## 總結", "", summary_text, ""])
        return "\n".join(lines).rstrip() + "\n"

    def _generate_chapter_notes(self, chapter: dict) -> str:
        summaries = [
            self._segment_text(segment)
            for segment in chapter.get("segments", [])
            if segment.get("teaching_stage") != "transition"
        ]
        summaries = [summary for summary in summaries if summary]
        if not summaries:
            return "- （此章節沒有可整理的重點內容）"

        if self.use_llm:
            prompt = STUDY_NOTES_PROMPT.format(
                chapter_title=chapter.get("title", ""),
                summaries="\n".join(f"- {summary}" for summary in summaries[:10]),
            )
            notes = self._call_llm(prompt, max_tokens=300)
            if notes:
                return notes

        return "\n".join(f"- {summary}" for summary in summaries[:8])

    def generate_mind_map(self, segments: list[dict], course_summary: dict) -> dict:
        title = str(course_summary.get("title") or "教學影片")
        concepts = course_summary.get("key_concepts") or []
        chapters = self.generate_chapter_titles(segments, course_summary)

        if self.use_llm and concepts:
            prompt = MIND_MAP_PROMPT.format(
                title=title,
                concepts=", ".join(str(item) for item in concepts),
                chapters=", ".join(chapter["title"] for chapter in chapters[:10]),
            )
            result = _parse_json_safe(self._call_llm(prompt, max_tokens=500))
            if self._valid_mind_map(result):
                return result

        children = []
        if concepts:
            for concept in concepts:
                concept_text = str(concept)
                related = []
                for segment in segments:
                    text = f"{segment.get('segment_summary', '')} {segment.get('text', '')}"
                    if concept_text and concept_text in text and segment.get("teaching_stage") != "transition":
                        label = self._segment_text(segment, 80)
                        if label:
                            related.append({"label": label})
                        if len(related) >= 3:
                            break
                children.append({
                    "label": concept_text,
                    "children": related or [{"label": "（參見課程內容）"}],
                })
        else:
            for chapter in chapters:
                related = [
                    {"label": self._segment_text(segment, 80)}
                    for segment in chapter.get("segments", [])[:3]
                    if segment.get("teaching_stage") != "transition" and self._segment_text(segment)
                ]
                children.append({
                    "label": chapter["title"],
                    "children": related or [{"label": "（無可用片段摘要）"}],
                })

        return {"root": title, "children": children}

    def generate_flashcards(self, segments: list[dict]) -> list[dict]:
        flashcards = []
        for segment in segments:
            stage = segment.get("teaching_stage")
            text = str(segment.get("text") or "").strip()
            if stage not in ("definition", "derivation") or len(text) <= 10:
                continue

            source_time = self._format_time(segment.get("start", 0))
            card = self._generate_single_flashcard(text, stage, source_time)
            if card:
                flashcards.append(card)
        return flashcards

    def _generate_single_flashcard(
        self, text: str, stage: str, source_time: str
    ) -> Optional[dict]:
        if self.use_llm:
            prompt = FLASHCARD_PROMPT.format(text=text[:1000], stage=stage)
            result = _parse_json_safe(self._call_llm(prompt, max_tokens=150))
            if (
                isinstance(result, dict)
                and isinstance(result.get("question"), str)
                and isinstance(result.get("answer"), str)
                and result["question"].strip()
                and result["answer"].strip()
            ):
                return {
                    "question": result["question"].strip(),
                    "answer": result["answer"].strip(),
                    "source_time": source_time,
                }

        answer = text[:300].strip()
        if stage == "definition":
            question = "這段內容主要定義或說明什麼概念？"
        else:
            question = "這段推導的主要目的或關鍵步驟是什麼？"
        return {"question": question, "answer": answer, "source_time": source_time}

    def export_all(
        self, segments: list[dict], course_summary: dict, output_dir: str
    ) -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        chapters = self.generate_chapter_titles(segments, course_summary)
        chapters_export = [
            {key: value for key, value in chapter.items() if key != "segments"}
            for chapter in chapters
        ]
        notes = self._build_study_notes(segments, course_summary, chapters)
        mind_map = self._build_mind_map(segments, course_summary, chapters)
        flashcards = self.generate_flashcards(segments)

        files = {
            "chapters": output_path / "chapters.json",
            "study_notes": output_path / "study_notes.md",
            "mind_map": output_path / "mind_map.json",
            "flashcards": output_path / "flashcards.json",
        }
        files["chapters"].write_text(
            json.dumps(chapters_export, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        files["study_notes"].write_text(notes, encoding="utf-8")
        files["mind_map"].write_text(
            json.dumps(mind_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        files["flashcards"].write_text(
            json.dumps(flashcards, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(
            f"[StudyMaterials] 已匯出 {len(chapters)} 個章節、"
            f"{len(flashcards)} 張閃卡至 {output_path}"
        )
        return {key: str(path) for key, path in files.items()}

    def _build_study_notes(
        self, segments: list[dict], course_summary: dict, chapters: list[dict]
    ) -> str:
        title = str(course_summary.get("title") or "教學影片筆記")
        summary_text = str(course_summary.get("summary") or "").strip()
        concepts = course_summary.get("key_concepts") or []
        lines = [f"# {title}", ""]

        if concepts:
            lines.extend(["## 關鍵概念", ""])
            lines.extend(f"- {concept}" for concept in concepts if str(concept).strip())
            lines.append("")

        lines.extend(["## 課程筆記", ""])
        if not chapters:
            lines.extend(["目前沒有可用的教學片段。", ""])
        for chapter in chapters:
            lines.extend([
                f"### 第 {chapter['chapter']} 章：{chapter['title']}", "",
                f"*時間：{self._format_time(chapter['start_sec'])} ~ "
                f"{self._format_time(chapter['end_sec'])}*", "",
                self._generate_chapter_notes(chapter), "",
            ])
        if summary_text:
            lines.extend(["## 總結", "", summary_text, ""])
        return "\n".join(lines).rstrip() + "\n"

    def _build_mind_map(
        self, segments: list[dict], course_summary: dict, chapters: list[dict]
    ) -> dict:
        # Keep the public method reusable while allowing export_all to share chapter results.
        title = str(course_summary.get("title") or "教學影片")
        concepts = course_summary.get("key_concepts") or []
        if self.use_llm and concepts:
            prompt = MIND_MAP_PROMPT.format(
                title=title,
                concepts=", ".join(str(item) for item in concepts),
                chapters=", ".join(chapter["title"] for chapter in chapters[:10]),
            )
            result = _parse_json_safe(self._call_llm(prompt, max_tokens=500))
            if self._valid_mind_map(result):
                return result

        children = []
        if concepts:
            for concept in concepts:
                concept_text = str(concept)
                related = []
                for segment in segments:
                    text = f"{segment.get('segment_summary', '')} {segment.get('text', '')}"
                    if concept_text and concept_text in text and segment.get("teaching_stage") != "transition":
                        label = self._segment_text(segment, 80)
                        if label:
                            related.append({"label": label})
                        if len(related) >= 3:
                            break
                children.append({
                    "label": concept_text,
                    "children": related or [{"label": "（參見課程內容）"}],
                })
        else:
            for chapter in chapters:
                related = [
                    {"label": self._segment_text(segment, 80)}
                    for segment in chapter.get("segments", [])[:3]
                    if segment.get("teaching_stage") != "transition" and self._segment_text(segment)
                ]
                children.append({
                    "label": chapter["title"],
                    "children": related or [{"label": "（無可用片段摘要）"}],
                })
        return {"root": title, "children": children}


def _parse_json_safe(raw: str) -> dict:
    """Parse JSON returned by an LLM, including responses wrapped in code fences."""
    if not raw or not isinstance(raw, str):
        return {}
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # Try each possible object start to avoid greedy extraction swallowing extra text.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start():])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}
