import json
from pathlib import Path

import codex_loop


def test_read_prompts_lines_filters_comments(tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(
        "# comment\n\nfirst prompt\n  second prompt  \n# ignored", encoding="utf-8"
    )

    lines = codex_loop.read_prompts_lines(prompts_file)

    assert lines == ["first prompt", "second prompt"]


def test_main_writes_manifest_and_respects_timeout(monkeypatch, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("first\nsecond\n", encoding="utf-8")

    class _FixedDatetime(codex_loop.datetime):
        @classmethod
        def now(cls):
            return cls(2024, 5, 1, 12, 0, 0)

    calls = []

    class _Proc:
        def __init__(self):
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    def _fake_run(cmd, capture_output, text, check, timeout):
        calls.append({"cmd": cmd, "timeout": timeout})
        return _Proc()

    monkeypatch.setattr(codex_loop, "datetime", _FixedDatetime)
    monkeypatch.setattr(codex_loop.subprocess, "run", _fake_run)

    output_root = tmp_path / "outputs"
    manifest_file = tmp_path / "manifest.json"

    exit_code = codex_loop.main(
        [
            "--prompts",
            str(prompts_file),
            "--output-dir",
            str(output_root),
            "--run-label",
            "nightly run",
            "--timeout",
            "12.5",
            "--manifest",
            str(manifest_file),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert all(call["timeout"] == 12.5 for call in calls)

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["run_label"] == "nightly run"
    assert len(manifest["results"]) == 2
    assert all(result["status"] == "ok" for result in manifest["results"])
    assert Path(manifest["output_dir"]).name == "20240501_120000_nightly-run"
    for result in manifest["results"]:
        assert result["output_file"].endswith(".txt")
    summary = manifest["summary"]
    assert summary["total_prompts"] == 2
    assert summary["ok"] == 2
    assert summary["error"] == 0
    assert summary["timeout"] == 0
    assert summary["had_failure"] is False
    history_log = output_root / "run_history.jsonl"
    assert manifest["history_log"] == str(history_log.resolve())
    history_lines = history_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 1
    history_entry = json.loads(history_lines[0])
    assert history_entry["manifest"] == str(manifest_file.resolve())
    assert history_entry["summary"]["total_prompts"] == 2


def test_main_fail_on_error_propagates_exit_code(monkeypatch, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("alpha\nbeta\n", encoding="utf-8")

    class _Proc:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    responses = [
        _Proc(returncode=0, stdout="ok alpha"),
        _Proc(returncode=2, stdout="bad beta", stderr="boom"),
    ]

    def _fake_run(cmd, capture_output, text, check, timeout):
        return responses.pop(0)

    monkeypatch.setattr(codex_loop.subprocess, "run", _fake_run)

    output_root = tmp_path / "out"
    manifest_file = tmp_path / "fail_manifest.json"
    exit_code = codex_loop.main(
        [
            "--prompts",
            str(prompts_file),
            "--output-dir",
            str(output_root),
            "--fail-on-error",
            "--manifest",
            str(manifest_file),
        ]
    )

    assert exit_code == 1
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    summary = manifest["summary"]
    assert summary["had_failure"] is True
    assert summary["ok"] == 1
    assert summary["error"] == 1
    assert summary["timeout"] == 0
    assert manifest["results"][1]["status"] == "error"


def test_interval_sec_inserts_sleep(monkeypatch, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("first\nsecond\nthird\n", encoding="utf-8")

    class _Proc:
        def __init__(self):
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    def _fake_run(cmd, capture_output, text, check, timeout):
        return _Proc()

    sleeps = []

    def _fake_sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(codex_loop.subprocess, "run", _fake_run)
    monkeypatch.setattr(codex_loop.time, "sleep", _fake_sleep)

    output_root = tmp_path / "run"
    manifest_file = tmp_path / "manifest_interval.json"
    exit_code = codex_loop.main(
        [
            "--prompts",
            str(prompts_file),
            "--output-dir",
            str(output_root),
            "--interval-sec",
            "1.5",
            "--manifest",
            str(manifest_file),
        ]
    )

    assert exit_code == 0
    assert sleeps == [1.5, 1.5]
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["interval_sec"] == 1.5
    assert manifest["summary"]["total_prompts"] == 3


def test_history_log_override_is_honored(monkeypatch, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("solo\n", encoding="utf-8")

    class _Proc:
        def __init__(self):
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    def _fake_run(cmd, capture_output, text, check, timeout):
        return _Proc()

    class _FixedDatetime(codex_loop.datetime):
        @classmethod
        def now(cls):
            return cls(2024, 6, 2, 9, 30, 0)

    monkeypatch.setattr(codex_loop.subprocess, "run", _fake_run)
    monkeypatch.setattr(codex_loop, "datetime", _FixedDatetime)

    history_log = tmp_path / "history" / "custom.jsonl"
    output_root = tmp_path / "out"

    exit_code = codex_loop.main(
        [
            "--prompts",
            str(prompts_file),
            "--output-dir",
            str(output_root),
            "--history-log",
            str(history_log),
        ]
    )

    assert exit_code == 0
    assert history_log.exists()
    entry_lines = history_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(entry_lines) == 1
    entry = json.loads(entry_lines[0])
    assert entry["summary"]["total_prompts"] == 1
    assert entry["manifest"].endswith("manifest.json")
    assert Path(entry["run_dir"]).name == "20240602_093000"
