import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.platform.keytake_api_optimized as api


def test_legacy_celery_index_shape_is_normalized_for_frontend():
    result = api._normalize_task_result(
        {
            "segments": {
                "segments": [{"start_sec": 1.0, "end_sec": 2.0}],
                "course_summary": {"title": "A lesson"},
            },
            "timeline": [{"source_start": 1.0, "output_start": 0.0}],
        }
    )

    assert result["segments"] == [{"start_sec": 1.0, "end_sec": 2.0}]
    assert result["course_summary"] == {"title": "A lesson"}
    assert result["timeline"] == [{"source_start": 1.0, "output_start": 0.0}]


def test_sync_fallback_returns_exported_timeline_and_encoded_duration(
    tmp_path,
    monkeypatch,
):
    task_id = "12345678-1234-4234-8234-123456789abc"
    output_dir = tmp_path / "output"
    timeline = [
        {
            "source_start": 4.0,
            "source_end": 6.0,
            "output_start": 0.2,
            "output_end": 2.2,
        }
    ]
    pipeline_result = {
        "segments": [{"start": 4.0, "end": 6.0}],
        "course_summary": {"title": "Test"},
        "original_duration": 10.0,
        "summary_duration": 2.0,
        "time_saving_rate": 0.8,
    }

    def export_summary_video(_source, _segments, output, timeline_output_path):
        with open(output, "wb"):
            pass
        with open(timeline_output_path, "w", encoding="utf-8") as file:
            json.dump(timeline, file)

    def export_index(_segments, output, course_summary=None):
        with open(output, "w", encoding="utf-8") as file:
            json.dump(
                {"segments": [{"start_sec": 4.0, "end_sec": 6.0}],
                 "course_summary": course_summary},
                file,
            )

    task_store = {}
    monkeypatch.setattr(api, "_get_task_store", lambda: task_store)
    monkeypatch.setattr(api, "_task_output_dir", lambda _task: output_dir)
    monkeypatch.setitem(
        sys.modules,
        "main",
        SimpleNamespace(run_pipeline=lambda *_args, **_kwargs: pipeline_result),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.output.video_exporter_optimized",
        SimpleNamespace(
            export_index=export_index,
            export_summary_video=export_summary_video,
            probe_video_duration=lambda _path: 2.4,
        ),
    )

    api._run_sync(task_id, str(tmp_path / "source.mp4"))
    result = task_store[task_id]["result"]

    assert result["segments"] == [{"start_sec": 4.0, "end_sec": 6.0}]
    assert result["timeline"] == timeline
    assert result["summary_duration"] == 2.4
    assert result["time_saving_rate"] == pytest.approx(0.76)


def test_sync_fallback_skips_celery_publish(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "OUTPUT_DIR", output_dir)
    monkeypatch.setenv("KEYTAKE_SYNC_FALLBACK", "1")

    threads = []

    class StubThread:
        def __init__(self, target, args, daemon, name):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.name = name

        def start(self):
            threads.append(self)

    monkeypatch.setattr(api, "threading", SimpleNamespace(Thread=StubThread))
    publish_calls = []
    fake_task_module = SimpleNamespace(
        process_video_task=SimpleNamespace(
            apply_async=lambda **kwargs: publish_calls.append(kwargs)
        )
    )
    monkeypatch.setitem(sys.modules, "src.platform.celery_app", fake_task_module)

    with TestClient(api.app) as client:
        response = client.post(
            "/upload",
            files={"file": ("sample.mp4", b"video bytes", "video/mp4")},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["mode"] == "sync_fallback"
    assert publish_calls == []
    assert len(threads) == 1
    assert threads[0].target is api._run_sync
    assert threads[0].args[0] == payload["task_id"]
    assert threads[0].daemon is True
    assert (upload_dir / f"{payload['task_id']}.mp4").is_file()
