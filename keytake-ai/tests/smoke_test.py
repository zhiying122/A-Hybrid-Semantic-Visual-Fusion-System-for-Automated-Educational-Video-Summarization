"""
Smoke Test — 用假資料驗證整個 pipeline 邏輯
不需要真實影片、不需要 GPU，純粹確認各模組串接不會炸掉

執行：
  cd keytake-ai
  python -m pytest tests/smoke_test.py -v
  # 或直接執行
  python tests/smoke_test.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 假資料工廠 ────────────────────────────────────────────────

def make_fake_segments(n: int = 10) -> list[dict]:
    """產生假的 Whisper 逐字稿片段"""
    templates = [
        "這邊非常重要，同學要特別注意",
        "我們來看這個推導過程",
        "好，接下來講一個比較輕鬆的例子",
        "這個公式很關鍵，請大家抄下來",
        "嗯，那個，我們繼續往下走",
        "導致這個結果的原因是矩陣不可逆",
        "好啦今天就到這裡，下課",
        "這個定義一定要記起來",
        "同學有問題嗎？沒有的話我們繼續",
        "總結一下今天的重點",
    ]
    segments = []
    t = 0.0
    for i in range(n):
        duration = np.random.uniform(5, 20)
        segments.append({
            "start": round(t, 2),
            "end": round(t + duration, 2),
            "text": templates[i % len(templates)]
        })
        t += duration + np.random.uniform(0.5, 2)
    return segments


def make_fake_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """產生假的影像幀（BGR）"""
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return frame


def make_fake_ground_truth(segments: list[dict]) -> list[tuple[float, float]]:
    """隨機選 30% 的片段作為假 Ground Truth"""
    gt = []
    for seg in segments:
        if np.random.random() < 0.3:
            gt.append((seg["start"], seg["end"]))
    return gt if gt else [(segments[0]["start"], segments[0]["end"])]


# ── 各模組測試 ────────────────────────────────────────────────

def test_semantic_scorer():
    """測試語意評分模組"""
    print("\n[Test] SemanticScorer...")
    from data.prompt_corpus.corpus import PROMPT_CORPUS
    from src.semantic.scorer import SemanticScorer

    scorer = SemanticScorer(PROMPT_CORPUS)
    segments = make_fake_segments(5)
    result = scorer.score(segments)

    assert all("s_text" in s for s in result), "缺少 s_text 欄位"
    assert all(0.0 <= s["s_text"] <= 1.0 for s in result), "s_text 超出 [0,1] 範圍"
    print(f"  ✓ 語意分數範圍正常，樣本：{[round(s['s_text'], 3) for s in result]}")


def test_hand_tracker():
    """測試手部追蹤模組（用假幀，不需要真實影片）"""
    print("\n[Test] HandTracker...")
    from src.visual.hand_tracker import HandTracker

    tracker = HandTracker()
    frame = make_fake_frame()
    result = tracker.process_frame(frame)

    assert "triggered" in result
    assert "s_visual_raw" in result
    assert 0.0 <= result["s_visual_raw"] <= 1.0
    print(f"  ✓ 手部追蹤回傳格式正常，triggered={result['triggered']}")


def test_srgan_fallback():
    """測試 SRGAN 降級備援（無 GPU 時應自動用 bicubic）"""
    print("\n[Test] SRGAN fallback...")
    from src.visual.srgan import enhance_roi, is_blurry

    roi = make_fake_frame(64, 64)
    enhanced = enhance_roi(roi)
    assert enhanced is not None
    assert enhanced.ndim == 3  # 應該是 3D 影像

    blurry = is_blurry(roi)
    assert isinstance(blurry, bool)
    print(f"  ✓ SRGAN 降級正常，輸出尺寸 {roi.shape[:2]} → {enhanced.shape[:2]}")


def test_adaptive_fusion():
    """測試多模態融合與滑動視窗"""
    print("\n[Test] AdaptiveFusion...")
    from src.fusion.adaptive_fusion import fuse_scores, semantic_sliding_window

    segments = make_fake_segments(10)
    for seg in segments:
        seg["s_text"] = np.random.uniform(0, 1)
        seg["s_visual"] = np.random.uniform(0, 1)

    scores = [fuse_scores(s["s_text"], s["s_visual"]) for s in segments]
    assert all(0.0 <= sc <= 1.0 for sc in scores), "融合分數超出範圍"

    selected = semantic_sliding_window(segments, scores)
    assert isinstance(selected, list)
    print(f"  ✓ 融合正常，{len(segments)} 段 → 選取 {len(selected)} 段")


def test_grid_search():
    """測試 Grid Search 權重搜尋"""
    print("\n[Test] GridSearch...")
    from src.fusion.adaptive_fusion import grid_search_weights

    segments = make_fake_segments(8)
    for seg in segments:
        seg["s_text"] = np.random.uniform(0, 1)
        seg["s_visual"] = np.random.uniform(0, 1)

    gt = make_fake_ground_truth(segments)
    alpha, beta = grid_search_weights(segments, gt)

    assert 0.0 <= alpha <= 1.0
    assert abs(alpha + beta - 1.0) < 1e-6
    print(f"  ✓ Grid Search 完成，最佳 α={alpha}, β={beta}")


def test_evaluator():
    """測試評估指標計算"""
    print("\n[Test] Evaluator...")
    from src.fusion.evaluator import compute_recall, compute_time_saving_rate, compute_false_alarm_rate

    segments = make_fake_segments(10)
    selected = segments[:3]
    gt = [(segments[0]["start"], segments[0]["end"]),
          (segments[2]["start"], segments[2]["end"])]
    total_duration = segments[-1]["end"]

    recall = compute_recall(selected, gt)
    tsr = compute_time_saving_rate(selected, total_duration)
    far = compute_false_alarm_rate(selected, gt, total_duration)

    assert 0.0 <= recall <= 1.0
    assert 0.0 <= tsr <= 1.0
    assert 0.0 <= far <= 1.0
    print(f"  ✓ Recall={recall:.3f}, TSR={tsr:.3f}, FAR={far:.3f}")


def test_video_exporter_index():
    """測試片段索引匯出（不需要真實影片）"""
    print("\n[Test] VideoExporter index...")
    import json
    import tempfile
    from src.output.video_exporter import export_index

    segments = make_fake_segments(5)
    for seg in segments:
        seg["text"] = "測試片段"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    export_index(segments, tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        index = json.load(f)

    assert len(index) == len(segments)
    assert all("timestamp" in item for item in index)
    os.remove(tmp_path)
    print(f"  ✓ 索引匯出正常，共 {len(index)} 筆")


# ── 執行所有測試 ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_semantic_scorer,
        test_hand_tracker,
        test_srgan_fallback,
        test_adaptive_fusion,
        test_grid_search,
        test_evaluator,
        test_video_exporter_index,
    ]

    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed.append((t.__name__, str(e)))
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'='*40}")
    print(f"結果：{passed}/{len(tests)} 通過")
    if failed:
        print("失敗項目：")
        for name, err in failed:
            print(f"  - {name}: {err}")
    else:
        print("所有測試通過 ✓")
