"""端對端資料流驗證（不需要 AI 模型）"""
import json

STAGE_TO_SEG_TYPE = {
    "definition": "semantic", "derivation": "combined",
    "example": "visual", "summary": "combined",
    "transition": "semantic", "qa": "semantic",
}
STAGE_LABELS = {
    "definition": "概念定義", "derivation": "公式推導",
    "example": "例題說明", "summary": "重點總結",
    "transition": "過渡換場", "qa": "問答互動",
}

mock_segments = [
    {
        "start": 10.0, "end": 25.0,
        "text": "今天我們來學微積分的極限定義",
        "s_text": 0.9, "s_visual": 0.3,
        "teaching_stage": "definition",
        "stage_label": "概念定義",
        "segment_summary": "極限的定義",
    },
    {
        "start": 26.0, "end": 58.0,
        "text": "我們來推導這個公式",
        "s_text": 0.85, "s_visual": 0.8,
        "teaching_stage": "derivation",
        "stage_label": "公式推導",
        "segment_summary": "微分公式推導",
    },
]

mock_course_summary = {
    "title": "微積分 - 極限與微分",
    "summary": "本課介紹極限的嚴格定義，並推導基本微分公式。",
    "key_concepts": ["極限", "微分", "epsilon-delta", "連續性"],
    "structure": [{"stage": "main_content", "description": "推導微分公式"}],
}

index = []
for seg in mock_segments:
    m, s = divmod(int(seg["start"]), 60)
    label = seg.get("segment_summary", "").strip() or seg.get("text", "")[:30]
    teaching_stage = seg.get("teaching_stage", "")
    seg_type = STAGE_TO_SEG_TYPE.get(teaching_stage, "semantic")
    stage_label = STAGE_LABELS.get(teaching_stage, teaching_stage)
    index.append({
        "timestamp": f"{m:02d}:{s:02d}",
        "start_sec": seg["start"],
        "end_sec": seg["end"],
        "label": label[:40],
        "type": seg_type,
        "stage": teaching_stage,
        "stage_label": stage_label,
        "s_text": round(seg.get("s_text", 0.0), 3),
        "s_visual": round(seg.get("s_visual", 0.0), 3),
    })

output_data = {"segments": index, "course_summary": mock_course_summary}

assert output_data["segments"][0]["stage"] == "definition"
assert output_data["segments"][0]["type"] == "semantic"
assert output_data["segments"][0]["label"] == "極限的定義"
assert output_data["segments"][1]["stage"] == "derivation"
assert output_data["segments"][1]["type"] == "combined"
assert output_data["course_summary"]["title"] == "微積分 - 極限與微分"
assert len(output_data["course_summary"]["key_concepts"]) == 4

serialized = json.dumps(output_data, ensure_ascii=False)
restored = json.loads(serialized)
assert restored["segments"][0]["label"] == "極限的定義"

print("ALL PASS")
print(json.dumps(output_data, ensure_ascii=False, indent=2)[:500])
