import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "codex_loop_watchdog_runner.sh"


def _make_stub(tmp_path, name, exit_env):
    log_file = tmp_path / f"{name}_log.txt"
    script_file = tmp_path / f"{name}_stub.sh"
    script_file.write_text(
        f"""#!/usr/bin/env bash
printf '{name}:%s\\n' "$*" >> "{log_file}"
exit "${{{exit_env}:-0}}"
""",
        encoding="utf-8",
    )
    script_file.chmod(0o755)
    return script_file, log_file


def _run_runner(loop_stub, watchdog_stub, args, env=None):
    runner_env = os.environ.copy()
    runner_env.update(
        {
            "CODEX_LOOP_BIN": str(loop_stub),
            "WATCHDOG_BIN": str(watchdog_stub),
        }
    )
    if env:
        runner_env.update(env)
    return subprocess.run(
        [str(RUNNER), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=runner_env,
    )


def test_runner_executes_loop_then_watchdog(tmp_path):
    loop_stub, loop_log = _make_stub(tmp_path, "loop", "LOOP_STUB_EXIT")
    watchdog_stub, watchdog_log = _make_stub(tmp_path, "watchdog", "WATCHDOG_STUB_EXIT")

    result = _run_runner(
        loop_stub,
        watchdog_stub,
        ["-p", "docment/codex_docs/codex_loop_prompts.sample.txt", "--", "-L", "nightly"],
        env={"LOOP_STUB_EXIT": "0", "WATCHDOG_STUB_EXIT": "0"},
    )

    assert result.returncode == 0
    assert "loop:-p docment/codex_docs/codex_loop_prompts.sample.txt" in loop_log.read_text(
        encoding="utf-8"
    )
    assert "watchdog:-L nightly" in watchdog_log.read_text(encoding="utf-8")


def test_runner_returns_loop_exit_and_can_skip_watchdog(tmp_path):
    loop_stub, loop_log = _make_stub(tmp_path, "loop", "LOOP_STUB_EXIT")
    watchdog_stub, watchdog_log = _make_stub(tmp_path, "watchdog", "WATCHDOG_STUB_EXIT")

    result = _run_runner(
        loop_stub,
        watchdog_stub,
        ["--", "-L", "nightly"],
        env={"LOOP_STUB_EXIT": "2", "WATCHDOG_SKIP_ON_FAILURE": "1"},
    )

    assert result.returncode == 2
    assert "loop:" in loop_log.read_text(encoding="utf-8")
    assert not watchdog_log.exists()


def test_runner_runs_watchdog_even_when_loop_fails_without_skip(tmp_path):
    loop_stub, loop_log = _make_stub(tmp_path, "loop", "LOOP_STUB_EXIT")
    watchdog_stub, watchdog_log = _make_stub(tmp_path, "watchdog", "WATCHDOG_STUB_EXIT")

    result = _run_runner(
        loop_stub,
        watchdog_stub,
        ["--", "-L", "nightly"],
        env={"LOOP_STUB_EXIT": "2"},
    )

    assert result.returncode == 2
    assert "loop:" in loop_log.read_text(encoding="utf-8")
    assert "watchdog:-L nightly" in watchdog_log.read_text(encoding="utf-8")


def test_runner_prefers_watchdog_exit_when_loop_succeeds(tmp_path):
    loop_stub, _ = _make_stub(tmp_path, "loop", "LOOP_STUB_EXIT")
    watchdog_stub, watchdog_log = _make_stub(tmp_path, "watchdog", "WATCHDOG_STUB_EXIT")

    result = _run_runner(
        loop_stub,
        watchdog_stub,
        ["--", "-R"],
        env={"LOOP_STUB_EXIT": "0", "WATCHDOG_STUB_EXIT": "3"},
    )

    assert result.returncode == 3
    assert "watchdog:-R" in watchdog_log.read_text(encoding="utf-8")
