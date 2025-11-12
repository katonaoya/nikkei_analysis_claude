#!/usr/bin/env python3
"""
codex_history_report.py

Utility script to inspect `codex_loop.py` run history JSONL files.
It prints a concise per-run summary plus aggregate stats so that cron /
launchd executions can be audited quickly.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = REPO_ROOT / "results" / "codex_cli_runs" / "run_history.jsonl"


def normalize_path_str(value: str) -> str:
    """
    Normalize a filesystem path for comparison. This resolves user-expansion and
    returns an absolute path without requiring the target to exist.
    """
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except Exception:
        # Fallback to absolute path normalization without resolution to keep best-effort matching.
        return str(Path(value).expanduser().absolute())


def load_history_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"history file not found: {path}")
    entries: List[Dict[str, Any]] = []
    for idx, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {idx}: {exc}") from exc
    return entries


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(f"{value}T00:00:00")


def resolve_since(
    since_str: Optional[str],
    last_days: Optional[int],
    last_hours: Optional[int],
    *,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    specified = [bool(since_str), last_days is not None, last_hours is not None]
    if sum(1 for flag in specified if flag) > 1:
        raise ValueError(
            "--since, --last-days, and --last-hours are mutually exclusive."
        )
    if since_str:
        return _parse_datetime(since_str)
    if last_days is not None:
        if last_days <= 0:
            raise ValueError("--last-days must be >= 1.")
        if now is None:
            now = datetime.now()
        return now - timedelta(days=last_days)
    if last_hours is not None:
        if last_hours <= 0:
            raise ValueError("--last-hours must be >= 1.")
        if now is None:
            now = datetime.now()
        return now - timedelta(hours=last_hours)
    return None


def filter_entries(
    entries: Iterable[Dict[str, Any]],
    *,
    run_label: Optional[str] = None,
    since: Optional[datetime] = None,
    failures_only: bool = False,
    backfilled_mode: str = "include",
    prompts_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        if run_label and (entry.get("run_label") or "") != run_label:
            continue
        if failures_only and not entry.get("summary", {}).get("had_failure"):
            continue
        entry_backfilled = bool(entry.get("backfilled"))
        if backfilled_mode == "exclude" and entry_backfilled:
            continue
        if backfilled_mode == "only" and not entry_backfilled:
            continue
        if prompts_file:
            entry_prompts = entry.get("prompts_file")
            if not entry_prompts:
                continue
            try:
                normalized_entry_prompts = normalize_path_str(str(entry_prompts))
            except Exception:
                continue
            if normalized_entry_prompts != prompts_file:
                continue
        if since is not None:
            logged_at = entry.get("logged_at")
            if not logged_at:
                continue
            try:
                logged_dt = datetime.fromisoformat(logged_at)
            except ValueError:
                continue
            if logged_dt < since:
                continue
        filtered.append(entry)
    return filtered


@dataclass
class HistoryStats:
    total_runs: int
    failures: int
    total_prompts: int
    ok: int
    error: int
    timeout: int


def compute_stats(entries: Iterable[Dict[str, Any]]) -> HistoryStats:
    total_runs = 0
    failures = 0
    total_prompts = 0
    ok = 0
    error = 0
    timeout = 0
    for entry in entries:
        total_runs += 1
        summary = entry.get("summary") or {}
        total_prompts += int(summary.get("total_prompts") or 0)
        ok += int(summary.get("ok") or 0)
        error += int(summary.get("error") or 0)
        timeout += int(summary.get("timeout") or 0)
        if summary.get("had_failure"):
            failures += 1
    return HistoryStats(
        total_runs=total_runs,
        failures=failures,
        total_prompts=total_prompts,
        ok=ok,
        error=error,
        timeout=timeout,
    )


def _readable_ts(raw_ts: str) -> str:
    try:
        return datetime.fromisoformat(raw_ts).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw_ts


def format_entry(entry: Dict[str, Any]) -> str:
    summary = entry.get("summary") or {}
    timestamp = _readable_ts(entry.get("logged_at", "?"))
    label = entry.get("run_label") or "-"
    ok = summary.get("ok", 0)
    error = summary.get("error", 0)
    timeout = summary.get("timeout", 0)
    total = summary.get("total_prompts", 0)
    failure_flag = "FAIL" if summary.get("had_failure") else "OK"
    manifest = entry.get("manifest", "")
    return (
        f"{timestamp} | label={label} | prompts={total} "
        f"(ok={ok} error={error} timeout={timeout}) | {failure_flag} | {manifest}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize codex_loop run history JSONL files."
    )
    parser.add_argument(
        "--history",
        default=str(DEFAULT_HISTORY),
        help=f"Path to the run history JSONL (default: {DEFAULT_HISTORY}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of rows to display (newest first). Use 0 to show all.",
    )
    parser.add_argument("--label", help="Only show entries with this run_label.")
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Only show runs where summary.had_failure is true.",
    )
    parser.add_argument(
        "--since",
        help="Show entries logged on/after this ISO timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        help="Show entries from the last N days (mutually exclusive with --since).",
    )
    parser.add_argument(
        "--last-hours",
        type=int,
        help="Show entries from the last N hours (mutually exclusive with --since/--last-days).",
    )
    parser.add_argument(
        "--require-matches",
        action="store_true",
        help="Return exit code 2 when no runs match the filters (useful for watchdogs).",
    )
    parser.add_argument(
        "--fail-when-failure",
        action="store_true",
        help="Return exit code 3 if any matching run has summary.had_failure=true.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        help="Require at least this many matching runs; otherwise exit code 2.",
    )
    parser.add_argument(
        "--min-prompts",
        type=int,
        help="Require at least this many cumulative prompts; otherwise exit code 2.",
    )
    parser.add_argument(
        "--min-prompts-per-run",
        type=int,
        help="Require each matching run to have at least this many prompts; otherwise exit code 2.",
    )
    parser.add_argument(
        "--backfilled",
        choices=("include", "exclude", "only"),
        default="include",
        help="Control whether backfilled entries are included (default), excluded, or exclusively selected.",
    )
    parser.add_argument(
        "--prompts-file",
        help="Only include runs where prompts_file matches this path (absolute comparison).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for attr, flag in [
        ("min_runs", "--min-runs"),
        ("min_prompts", "--min-prompts"),
        ("min_prompts_per_run", "--min-prompts-per-run"),
    ]:
        value = getattr(args, attr)
        if value is not None and value <= 0:
            parser.error(f"{flag} must be >= 1.")

    history_path = Path(args.history).expanduser()

    prompts_filter: Optional[str] = None
    if args.prompts_file:
        prompts_filter = normalize_path_str(args.prompts_file)

    try:
        entries = load_history_entries(history_path)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    try:
        since_dt = resolve_since(args.since, args.last_days, args.last_hours)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    filtered = filter_entries(
        entries,
        run_label=args.label,
        since=since_dt,
        failures_only=args.failures_only,
        backfilled_mode=args.backfilled,
        prompts_file=prompts_filter,
    )
    if not filtered:
        print("No matching runs found.")
        return 2 if args.require_matches else 0

    filtered.sort(key=lambda e: e.get("logged_at", ""), reverse=True)

    if args.limit and args.limit > 0:
        visible = filtered[: args.limit]
    else:
        visible = filtered

    print(f"Showing {len(visible)} of {len(filtered)} matching runs (newest first)")
    print("-" * 80)
    for entry in visible:
        print(format_entry(entry))

    stats = compute_stats(filtered)
    print("-" * 80)
    print(
        "Totals: runs={total_runs} failures={failures} prompts={total_prompts} "
        "ok={ok} error={error} timeout={timeout}".format(**stats.__dict__)
    )
    coverage_errors: List[str] = []
    if args.min_runs is not None and stats.total_runs < args.min_runs:
        coverage_errors.append(
            f"Need at least {args.min_runs} run(s) but only {stats.total_runs} matched."
        )
    if args.min_prompts is not None and stats.total_prompts < args.min_prompts:
        coverage_errors.append(
            f"Need at least {args.min_prompts} prompt(s) but only {stats.total_prompts} matched."
        )
    lacking_runs: List[Dict[str, Any]] = []
    if args.min_prompts_per_run is not None:
        threshold = args.min_prompts_per_run
        for entry in filtered:
            prompts = int(entry.get("summary", {}).get("total_prompts") or 0)
            if prompts < threshold:
                lacking_runs.append(entry)
        if lacking_runs:
            coverage_errors.append(
                f"{len(lacking_runs)} run(s) have < {threshold} prompt(s)."
            )
    coverage_exit = 0
    if coverage_errors:
        for msg in coverage_errors:
            print(f"[coverage] {msg}", file=sys.stderr)
        coverage_exit = 2

    if args.fail_when_failure and stats.failures:
        print(
            f"[alert] Detected {stats.failures} failing run(s) in filtered entries.",
            file=sys.stderr,
        )
        return 3
    if coverage_exit:
        return coverage_exit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
