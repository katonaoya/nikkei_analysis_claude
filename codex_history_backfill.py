#!/usr/bin/env python3
"""
codex_history_backfill.py

Reconstruct codex_loop run_history.jsonl entries by scanning existing
manifest.json files. Useful when the history log was not enabled during
past executions but manifests exist under results/codex_cli_runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "codex_cli_runs"
DEFAULT_HISTORY_LOG = DEFAULT_RESULTS_DIR / "run_history.jsonl"


@dataclass
class ManifestRecord:
    path: Path
    data: Dict[str, object]
    created_at: datetime


def parse_iso_datetime(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def load_manifest_record(path: Path) -> ManifestRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"manifest missing summary: {path}")
    created_at = parse_iso_datetime(str(data.get("created_at") or ""))
    if not created_at:
        created_at = datetime.fromtimestamp(path.stat().st_mtime)
    return ManifestRecord(path=path, data=data, created_at=created_at)


def find_manifests(results_dir: Path) -> List[Path]:
    manifests: List[Path] = []
    if not results_dir.exists():
        return manifests
    for candidate in sorted(results_dir.glob("*/manifest.json")):
        if candidate.is_file():
            manifests.append(candidate)
    return manifests


def load_existing_manifest_paths(history_log: Path) -> Set[Path]:
    manifests: Set[Path] = set()
    if not history_log.exists():
        return manifests
    for idx, raw_line in enumerate(history_log.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            print(f"[warn] history line {idx} is invalid JSON; skipping", file=sys.stderr)
            continue
        manifest = entry.get("manifest")
        if manifest:
            manifests.add(Path(str(manifest)).resolve())
    return manifests


def build_history_entry(record: ManifestRecord) -> Dict[str, object]:
    data = record.data
    summary = data.get("summary")
    run_dir = data.get("output_dir") or str(record.path.parent.resolve())
    logged_at = data.get("created_at") or record.created_at.isoformat()
    entry = {
        "logged_at": logged_at,
        "run_dir": str(Path(str(run_dir)).resolve()),
        "manifest": str(record.path.resolve()),
        "summary": summary,
        "run_label": data.get("run_label") or "",
        "prompts_file": data.get("prompts_file"),
        "codex_cmd": data.get("codex_cmd"),
        "extra_args": data.get("extra_args"),
        "timeout": data.get("timeout"),
        "interval_sec": data.get("interval_sec"),
        "fail_on_error": bool(data.get("fail_on_error", False)),
        "backfilled": True,
    }
    return entry


def write_entries(entries: Sequence[Dict[str, object]], history_log: Path, rebuild: bool) -> None:
    history_log.parent.mkdir(parents=True, exist_ok=True)
    if rebuild:
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(history_log.parent), prefix=history_log.name, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fp:
                for entry in entries:
                    tmp_fp.write(json.dumps(entry, ensure_ascii=False))
                    tmp_fp.write("\n")
            os.replace(tmp_name, history_log)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    else:
        with history_log.open("a", encoding="utf-8") as fp:
            for entry in entries:
                fp.write(json.dumps(entry, ensure_ascii=False))
                fp.write("\n")


def describe_actions(entries: Iterable[ManifestRecord]) -> str:
    names = [record.path.parent.name for record in entries]
    return ", ".join(names) if names else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill codex_loop run_history.jsonl from manifest.json files."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory that contains codex_loop run folders (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--history-log",
        default=str(DEFAULT_HISTORY_LOG),
        help=f"run_history.jsonl path to update (default: {DEFAULT_HISTORY_LOG}).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rewrite the history log from scratch instead of appending missing entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which manifests would be added without touching files.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir).expanduser()
    history_log = Path(args.history_log).expanduser()
    is_rebuild = bool(args.rebuild)

    print(f"📁 results dir: {results_dir}")
    manifests = find_manifests(results_dir)
    if not manifests:
        print("ℹ️ No manifest.json files detected; nothing to backfill.")
        return 0

    manifest_records: List[ManifestRecord] = []
    for manifest_path in manifests:
        try:
            manifest_records.append(load_manifest_record(manifest_path))
        except ValueError as exc:
            print(f"[warn] {exc}; skipping", file=sys.stderr)

    if not manifest_records:
        print("⚠️ No valid manifest files were processed.", file=sys.stderr)
        return 1

    manifest_records.sort(key=lambda rec: (rec.created_at, rec.path.name))

    if is_rebuild:
        targets = manifest_records
    else:
        existing = load_existing_manifest_paths(history_log)
        targets = [rec for rec in manifest_records if rec.path.resolve() not in existing]
        if existing:
            print(f"🧮 history already tracks {len(existing)} manifest(s).")

    if not targets:
        print("✅ run_history.jsonl is already up-to-date.")
        return 0

    if args.dry_run:
        listed = describe_actions(targets)
        print(f"🧪 dry-run: would {'rewrite' if is_rebuild else 'append'} entries for: {listed}")
        return 0

    entries = [build_history_entry(rec) for rec in targets]
    write_entries(entries, history_log, rebuild=is_rebuild)

    listed = describe_actions(targets)
    action = "rebuilt" if is_rebuild else "appended"
    print(f"✅ {action} {len(targets)} entr{'y' if len(targets)==1 else 'ies'}: {listed}")
    print(f"🧾 history log: {history_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
