#!/usr/bin/env bash
# Run codex_loop.sh and (optionally) codex_history_watchdog.sh in sequence.
# Usage:
#   scripts/codex_loop_watchdog_runner.sh [codex_loop_args ...]
#   scripts/codex_loop_watchdog_runner.sh <codex_loop_args ...> -- <watchdog_args ...>
# Pass `--watchdog-help` after `--` to inspect the watchdog CLI (the runner itself
# provides `--help-runner` for its own usage guide).

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CODEX_LOOP_BIN="${CODEX_LOOP_BIN:-$ROOT_DIR/scripts/codex_loop.sh}"
WATCHDOG_BIN="${WATCHDOG_BIN:-$ROOT_DIR/scripts/codex_history_watchdog.sh}"
WATCHDOG_SKIP_ON_FAILURE="${WATCHDOG_SKIP_ON_FAILURE:-}"

usage() {
  cat <<'EOF'
codex_loop_watchdog_runner.sh - orchestrate codex_loop + codex_history_watchdog

Usage:
  scripts/codex_loop_watchdog_runner.sh [codex_loop_args ...]
  scripts/codex_loop_watchdog_runner.sh <codex_loop_args ...> -- <watchdog_args ...>

Examples:
  # Run codex_loop only
  scripts/codex_loop_watchdog_runner.sh -p docment/codex_docs/codex_loop_prompts.sample.txt

  # Run codex_loop and then the watchdog with thresholds
  scripts/codex_loop_watchdog_runner.sh \
    -p docment/codex_docs/codex_loop_prompts.sample.txt \
    -o results/codex_cli_runs \
    -- \
    -L nightly -a 6 -m 1 -p 3 -u 3 -F -R

Environment overrides:
  CODEX_LOOP_BIN             Path to codex_loop.sh (default: scripts/codex_loop.sh)
  WATCHDOG_BIN               Path to codex_history_watchdog.sh (default: scripts/codex_history_watchdog.sh)
  WATCHDOG_SKIP_ON_FAILURE   If set, skip watchdog execution when codex_loop exits non-zero.

Pass --help-runner before any other argument to display this message.
EOF
}

describe_cmd() {
  local exe="$1"
  shift || true
  printf '%s' "$exe"
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
}

log() {
  printf '🧭 [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "${1:-}" == "--help-runner" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$CODEX_LOOP_BIN" ]]; then
  log "❌ codex_loop 実行ファイルが見つかりません: $CODEX_LOOP_BIN"
  exit 1
fi

loop_args=()
has_watchdog=0
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    has_watchdog=1
    shift
    break
  fi
  loop_args+=("$1")
  shift
done

watchdog_args=("$@")
watchdog_exit=0
watchdog_invoked=0

loop_cmd=("$CODEX_LOOP_BIN")
if ((${#loop_args[@]})); then
  loop_cmd+=("${loop_args[@]}")
fi

log "🚀 codex_loop コマンド実行: $(describe_cmd "${loop_cmd[@]}")"
set +e
"${loop_cmd[@]}"
loop_exit=$?
set -e
if [[ $loop_exit -eq 0 ]]; then
  log "✅ codex_loop.sh が正常終了しました。"
else
  log "⚠️ codex_loop.sh が exit=$loop_exit で終了しました。"
fi

if (( has_watchdog )); then
  if [[ ! -x "$WATCHDOG_BIN" ]]; then
    log "❌ watchdog スクリプトが見つかりません: $WATCHDOG_BIN"
    exit 1
  fi
  if [[ -n "$WATCHDOG_SKIP_ON_FAILURE" && $loop_exit -ne 0 ]]; then
    log "⏭️ codex_loop が失敗したため WATCHDOG_SKIP_ON_FAILURE=1 に従い watchdog をスキップします。"
  else
    watchdog_cmd=("$WATCHDOG_BIN")
    if ((${#watchdog_args[@]})); then
      watchdog_cmd+=("${watchdog_args[@]}")
    fi
    log "🛡️ watchdog コマンド実行: $(describe_cmd "${watchdog_cmd[@]}")"
    set +e
    "${watchdog_cmd[@]}"
    watchdog_exit=$?
    set -e
    watchdog_invoked=1
    if [[ $watchdog_exit -eq 0 ]]; then
      log "✅ codex_history_watchdog.sh が正常終了しました。"
    else
      log "⚠️ codex_history_watchdog.sh が exit=$watchdog_exit を返しました。"
    fi
  fi
fi

if [[ $loop_exit -ne 0 ]]; then
  exit "$loop_exit"
fi

if (( has_watchdog )) && (( watchdog_invoked )); then
  exit "$watchdog_exit"
fi

exit 0
