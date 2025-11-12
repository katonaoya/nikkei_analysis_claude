import json
from datetime import datetime

import pytest

import codex_history_report as history_report


def _write_history(path, entries):
    with path.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry))
            fp.write("\n")


def test_load_history_entries_reads_all_lines(tmp_path):
    history_file = tmp_path / "run_history.jsonl"
    entries = [
        {"logged_at": "2025-11-07T00:00:00", "summary": {"total_prompts": 1}},
        {"logged_at": "2025-11-08T00:00:00", "summary": {"total_prompts": 2}},
    ]
    _write_history(history_file, entries)

    loaded = history_report.load_history_entries(history_file)

    assert len(loaded) == 2
    assert loaded[0]["summary"]["total_prompts"] == 1
    assert loaded[1]["summary"]["total_prompts"] == 2


def test_filter_entries_respects_label_since_and_failures():
    entries = [
        {
            "logged_at": "2025-11-07T10:00:00",
            "run_label": "nightly",
            "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
        },
        {
            "logged_at": "2025-11-08T12:00:00",
            "run_label": "nightly",
            "summary": {"total_prompts": 2, "ok": 1, "error": 1, "timeout": 0, "had_failure": True},
        },
        {
            "logged_at": "2025-11-09T09:00:00",
            "run_label": "weekly",
            "summary": {"total_prompts": 1, "ok": 1, "error": 0, "timeout": 0},
        },
    ]
    since = datetime.fromisoformat("2025-11-08T00:00:00")

    filtered = history_report.filter_entries(
        entries, run_label="nightly", since=since, failures_only=True
    )

    assert len(filtered) == 1
    assert filtered[0]["logged_at"] == "2025-11-08T12:00:00"


def test_filter_entries_handles_backfilled_mode():
    entries = [
        {"logged_at": "2025-11-09T00:00:00", "backfilled": True, "summary": {"total_prompts": 1}},
        {"logged_at": "2025-11-10T00:00:00", "summary": {"total_prompts": 1}},
    ]

    excluded = history_report.filter_entries(entries, backfilled_mode="exclude")
    assert len(excluded) == 1
    assert excluded[0]["logged_at"] == "2025-11-10T00:00:00"

    only = history_report.filter_entries(entries, backfilled_mode="only")
    assert len(only) == 1
    assert only[0]["logged_at"] == "2025-11-09T00:00:00"


def test_filter_entries_supports_prompts_file(tmp_path):
    prompts_a = tmp_path / "prompts_a.txt"
    prompts_b = tmp_path / "prompts_b.txt"
    prompts_a.write_text("A\n", encoding="utf-8")
    prompts_b.write_text("B\n", encoding="utf-8")
    entries = [
        {
            "logged_at": "2025-11-09T00:00:00",
            "prompts_file": str(prompts_a.resolve()),
            "summary": {"total_prompts": 2},
        },
        {
            "logged_at": "2025-11-10T00:00:00",
            "prompts_file": str(prompts_b.resolve()),
            "summary": {"total_prompts": 3},
        },
        {"logged_at": "2025-11-11T00:00:00", "summary": {"total_prompts": 1}},
    ]

    normalized = history_report.normalize_path_str(str(prompts_a))
    filtered = history_report.filter_entries(entries, prompts_file=normalized)

    assert len(filtered) == 1
    assert filtered[0]["prompts_file"].endswith("prompts_a.txt")


def test_resolve_since_supports_last_days_and_validation():
    now = datetime(2025, 11, 10, 12, 0, 0)
    since = history_report.resolve_since(None, 3, None, now=now)
    assert since == datetime(2025, 11, 7, 12, 0, 0)

    with pytest.raises(ValueError):
        history_report.resolve_since("2025-11-07", 1, 1)
    with pytest.raises(ValueError):
        history_report.resolve_since(None, 0, None, now=now)


def test_resolve_since_supports_last_hours():
    now = datetime(2025, 11, 10, 12, 0, 0)
    since = history_report.resolve_since(None, None, 6, now=now)
    assert since == datetime(2025, 11, 10, 6, 0, 0)

    with pytest.raises(ValueError):
        history_report.resolve_since(None, None, 0, now=now)


def test_compute_stats_and_format_entry(tmp_path):
    entry = {
        "logged_at": "2025-11-08T15:30:00",
        "run_label": "nightly",
        "manifest": "/tmp/manifest.json",
        "summary": {
            "total_prompts": 4,
            "ok": 3,
            "error": 1,
            "timeout": 0,
            "had_failure": True,
        },
    }
    stats = history_report.compute_stats([entry])
    assert stats.total_runs == 1
    assert stats.failures == 1
    assert stats.ok == 3

    formatted = history_report.format_entry(entry)
    assert "label=nightly" in formatted
    assert "FAIL" in formatted
    assert "/tmp/manifest.json" in formatted


def test_main_outputs_report_and_totals(tmp_path, capsys):
    history_file = tmp_path / "history.jsonl"
    entries = [
        {
            "logged_at": "2025-11-08T10:00:00",
            "run_label": "nightly",
            "manifest": "/tmp/a.json",
            "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
        },
        {
            "logged_at": "2025-11-09T11:00:00",
            "run_label": "nightly",
            "manifest": "/tmp/b.json",
            "summary": {
                "total_prompts": 3,
                "ok": 2,
                "error": 1,
                "timeout": 0,
                "had_failure": True,
            },
        },
    ]
    _write_history(history_file, entries)

    exit_code = history_report.main(["--history", str(history_file), "--limit", "1"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Showing 1 of 2 matching runs" in out
    assert "Totals: runs=2" in out


def test_main_returns_error_when_file_missing(tmp_path, capsys):
    missing = tmp_path / "none.jsonl"

    exit_code = history_report.main(["--history", str(missing)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "history file not found" in err


def test_main_require_matches_forces_nonzero_exit(tmp_path, capsys):
    history_file = tmp_path / "history.jsonl"
    _write_history(
        history_file,
        [
            {"logged_at": "2025-11-08T10:00:00", "run_label": "nightly", "summary": {"total_prompts": 1}},
        ],
    )

    exit_code = history_report.main(
        [
            "--history",
            str(history_file),
            "--label",
            "weekly",
            "--require-matches",
        ]
    )

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "No matching runs found" in out


def test_main_fail_when_failure_alerts(tmp_path, capsys):
    history_file = tmp_path / "history.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-08T10:00:00",
                "run_label": "nightly",
                "summary": {
                    "total_prompts": 2,
                    "ok": 1,
                    "error": 1,
                    "timeout": 0,
                    "had_failure": True,
                },
            },
        ],
    )

    exit_code = history_report.main(
        [
            "--history",
            str(history_file),
            "--fail-when-failure",
        ]
    )

    assert exit_code == 3
    captured = capsys.readouterr()
    assert "Totals: runs=1 failures=1" in captured.out
    assert "alert" in captured.err


def test_main_min_runs_guard(tmp_path, capsys):
    history_file = tmp_path / "runs.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-08T10:00:00",
                "run_label": "nightly",
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
            }
        ],
    )

    exit_code = history_report.main(
        [
            "--history",
            str(history_file),
            "--min-runs",
            "2",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Need at least 2 run(s)" in captured.err


def test_main_min_prompts_and_per_run_guard(tmp_path, capsys):
    history_file = tmp_path / "runs.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-08T10:00:00",
                "summary": {"total_prompts": 1, "ok": 1, "error": 0, "timeout": 0},
            },
            {
                "logged_at": "2025-11-08T11:00:00",
                "summary": {"total_prompts": 3, "ok": 3, "error": 0, "timeout": 0},
            },
        ],
    )

    exit_code = history_report.main(
        [
            "--history",
            str(history_file),
            "--min-prompts",
            "5",
            "--min-prompts-per-run",
            "2",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Need at least 5 prompt(s)" in captured.err
    assert "run(s) have < 2 prompt(s)" in captured.err


def test_main_backfilled_only_option(tmp_path, capsys):
    history_file = tmp_path / "runs.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-08T10:00:00",
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
            },
            {
                "logged_at": "2025-11-08T11:00:00",
                "summary": {"total_prompts": 3, "ok": 3, "error": 0, "timeout": 0},
                "backfilled": True,
            },
        ],
    )

    exit_code = history_report.main(
        [
            "--history",
            str(history_file),
            "--backfilled",
            "only",
            "--min-runs",
            "1",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Totals: runs=1" in out


def test_main_prompts_file_filter(tmp_path, capsys):
    history_file = tmp_path / "runs.jsonl"
    prompts = tmp_path / "monitored_prompts.txt"
    prompts.write_text("hello\n", encoding="utf-8")
    other_prompts = tmp_path / "other_prompts.txt"
    other_prompts.write_text("world\n", encoding="utf-8")
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-08T10:00:00",
                "prompts_file": str(prompts.resolve()),
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
            },
            {
                "logged_at": "2025-11-08T11:00:00",
                "prompts_file": str(other_prompts.resolve()),
                "summary": {"total_prompts": 3, "ok": 3, "error": 0, "timeout": 0},
            },
        ],
    )

    exit_code = history_report.main(
        [
            "--history",
            str(history_file),
            "--prompts-file",
            str(prompts),
            "--require-matches",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Totals: runs=1" in out
