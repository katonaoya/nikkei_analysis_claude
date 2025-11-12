import json
from pathlib import Path

import codex_history_backfill


def _write_manifest(
    run_root: Path,
    run_name: str,
    *,
    created_at: str,
    total_prompts: int,
) -> Path:
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "created_at": created_at,
        "prompts_file": "docment/codex_docs/prompts.txt",
        "codex_cmd": "codex",
        "extra_args": "--approvals full",
        "output_dir": str(run_dir),
        "run_label": run_name,
        "timeout": 30,
        "interval_sec": 0,
        "fail_on_error": True,
        "summary": {
            "total_prompts": total_prompts,
            "ok": total_prompts,
            "error": 0,
            "timeout": 0,
            "duration_sec": 1.23,
            "had_failure": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _read_history(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_backfill_creates_history_from_manifests(tmp_path):
    results_dir = tmp_path / "results" / "codex_cli_runs"
    history_log = tmp_path / "history.jsonl"
    manifest_a = _write_manifest(
        results_dir, "20240101_foo", created_at="2024-01-01T01:00:00", total_prompts=2
    )
    manifest_b = _write_manifest(
        results_dir, "20240102_bar", created_at="2024-01-02T02:00:00", total_prompts=3
    )

    exit_code = codex_history_backfill.main(
        [
            "--results-dir",
            str(results_dir),
            "--history-log",
            str(history_log),
        ]
    )

    assert exit_code == 0
    entries = _read_history(history_log)
    assert [entry["manifest"] for entry in entries] == [
        str(manifest_a.resolve()),
        str(manifest_b.resolve()),
    ]
    assert entries[0]["summary"]["total_prompts"] == 2
    assert entries[1]["run_label"] == "20240102_bar"
    assert entries[1]["backfilled"] is True


def test_backfill_skips_existing_entries(tmp_path):
    results_dir = tmp_path / "results" / "codex_cli_runs"
    history_log = tmp_path / "history.jsonl"
    manifest_a = _write_manifest(
        results_dir, "20240101_foo", created_at="2024-01-01T01:00:00", total_prompts=2
    )
    exit_code = codex_history_backfill.main(
        [
            "--results-dir",
            str(results_dir),
            "--history-log",
            str(history_log),
        ]
    )
    assert exit_code == 0

    manifest_b = _write_manifest(
        results_dir, "20240103_baz", created_at="2024-01-03T03:00:00", total_prompts=4
    )
    exit_code = codex_history_backfill.main(
        [
            "--results-dir",
            str(results_dir),
            "--history-log",
            str(history_log),
        ]
    )

    assert exit_code == 0
    entries = _read_history(history_log)
    assert len(entries) == 2
    assert entries[0]["manifest"] == str(manifest_a.resolve())
    assert entries[1]["manifest"] == str(manifest_b.resolve())


def test_backfill_rebuild_overwrites_history(tmp_path):
    results_dir = tmp_path / "results" / "codex_cli_runs"
    history_log = tmp_path / "history.jsonl"
    manifest_a = _write_manifest(
        results_dir, "20240101_foo", created_at="2024-01-01T01:00:00", total_prompts=2
    )
    manifest_b = _write_manifest(
        results_dir, "20240102_bar", created_at="2024-01-02T02:00:00", total_prompts=3
    )
    codex_history_backfill.main(
        [
            "--results-dir",
            str(results_dir),
            "--history-log",
            str(history_log),
        ]
    )
    history_log.write_text("stale\nentry\n", encoding="utf-8")
    manifest_c = _write_manifest(
        results_dir, "20240104_qux", created_at="2024-01-04T04:00:00", total_prompts=5
    )

    exit_code = codex_history_backfill.main(
        [
            "--results-dir",
            str(results_dir),
            "--history-log",
            str(history_log),
            "--rebuild",
        ]
    )

    assert exit_code == 0
    entries = _read_history(history_log)
    assert [entry["manifest"] for entry in entries] == [
        str(manifest_a.resolve()),
        str(manifest_b.resolve()),
        str(manifest_c.resolve()),
    ]
