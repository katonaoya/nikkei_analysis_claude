import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_SCRIPT = REPO_ROOT / "scripts" / "codex_history_watchdog.sh"


def _write_history(path, entries):
    with path.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry))
            fp.write("\n")


def _run_watchdog(args, history_file):
    cmd = [str(WATCHDOG_SCRIPT), "-H", str(history_file), *args]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_watchdog_passes_when_thresholds_met(tmp_path):
    history_file = tmp_path / "history.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-10T10:00:00",
                "summary": {
                    "total_prompts": 3,
                    "ok": 3,
                    "error": 0,
                    "timeout": 0,
                },
            }
        ],
    )

    result = _run_watchdog(
        ["-m", "1", "-p", "3", "-u", "3", "-R"],
        history_file,
    )

    assert result.returncode == 0
    assert "coverage OK" in result.stdout


def test_watchdog_reports_min_run_violation(tmp_path):
    history_file = tmp_path / "history.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-10T10:00:00",
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
            }
        ],
    )

    result = _run_watchdog(
        ["-m", "2"],
        history_file,
    )

    assert result.returncode == 2
    assert "Need at least 2 run(s)" in result.stderr
    assert "watchdog exit 2" in result.stdout


def test_watchdog_exits_when_failure_detected(tmp_path):
    history_file = tmp_path / "history.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-10T10:00:00",
                "run_label": "nightly",
                "summary": {
                    "total_prompts": 2,
                    "ok": 1,
                    "error": 1,
                    "timeout": 0,
                    "had_failure": True,
                },
            }
        ],
    )

    result = _run_watchdog(
        ["-F"],
        history_file,
    )

    assert result.returncode == 3
    assert "alert" in result.stderr
    assert "watchdog exit 3" in result.stdout


def test_watchdog_last_hours_filter(tmp_path):
    recent_history = tmp_path / "recent.jsonl"
    _write_history(
        recent_history,
        [
            {
                "logged_at": "2000-01-01T00:00:00",
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
            },
            {
                "logged_at": "2099-01-01T00:00:00",
                "summary": {"total_prompts": 3, "ok": 3, "error": 0, "timeout": 0},
            },
        ],
    )

    ok_result = _run_watchdog(
        ["-a", "6", "-R"],
        recent_history,
    )
    assert ok_result.returncode == 0

    stale_history = tmp_path / "stale.jsonl"
    _write_history(
        stale_history,
        [
            {
                "logged_at": "2000-01-01T00:00:00",
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
            }
        ],
    )

    stale_result = _run_watchdog(
        ["-a", "6", "-R"],
        stale_history,
    )
    assert stale_result.returncode == 2
    assert "No matching runs found" in stale_result.stdout


def test_watchdog_supports_backfilled_mode(tmp_path):
    history_file = tmp_path / "history.jsonl"
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-10T10:00:00",
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
                "backfilled": True,
            }
        ],
    )

    only_result = _run_watchdog(["-B", "only", "-m", "1"], history_file)
    assert only_result.returncode == 0

    exclude_result = _run_watchdog(["-B", "exclude", "-R"], history_file)
    assert exclude_result.returncode == 2
    assert "No matching runs found" in exclude_result.stdout


def test_watchdog_filters_by_prompts_file(tmp_path):
    history_file = tmp_path / "history.jsonl"
    prompts_a = tmp_path / "prompts_a.txt"
    prompts_b = tmp_path / "prompts_b.txt"
    prompts_a.write_text("A\n", encoding="utf-8")
    prompts_b.write_text("B\n", encoding="utf-8")
    _write_history(
        history_file,
        [
            {
                "logged_at": "2025-11-10T10:00:00",
                "prompts_file": str(prompts_a.resolve()),
                "summary": {"total_prompts": 2, "ok": 2, "error": 0, "timeout": 0},
            },
            {
                "logged_at": "2025-11-10T12:00:00",
                "prompts_file": str(prompts_b.resolve()),
                "summary": {"total_prompts": 3, "ok": 3, "error": 0, "timeout": 0},
            },
        ],
    )

    ok_result = _run_watchdog(["-P", str(prompts_a), "-R"], history_file)
    assert ok_result.returncode == 0

    miss_result = _run_watchdog(["-P", str(prompts_b.with_name("missing.txt")), "-R"], history_file)
    assert miss_result.returncode == 2
    assert "No matching runs found" in miss_result.stdout
