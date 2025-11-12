#!/usr/bin/env bash
# codex_history_report.py を watchdog 用の閾値監視コマンドとして呼び出すヘルパー。

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
HISTORY_REPORT="$ROOT_DIR/codex_history_report.py"
DEFAULT_HISTORY="$ROOT_DIR/results/codex_cli_runs/run_history.jsonl"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HISTORY_FILE="${HISTORY_FILE:-$DEFAULT_HISTORY}"
RUN_LABEL="${RUN_LABEL:-}"
LAST_DAYS="${LAST_DAYS:-}"
LAST_HOURS="${LAST_HOURS:-}"
SINCE="${SINCE:-}"
MIN_RUNS="${MIN_RUNS:-}"
MIN_PROMPTS="${MIN_PROMPTS:-}"
MIN_PROMPTS_PER_RUN="${MIN_PROMPTS_PER_RUN:-}"
BACKFILLED_MODE="${BACKFILLED_MODE:-}"
LIMIT="${LIMIT:-20}"
FAIL_WHEN_FAILURE="${FAIL_WHEN_FAILURE:-}"
REQUIRE_MATCHES="${REQUIRE_MATCHES:-}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
DRY_RUN=0

usage() {
  cat <<'EOF'
codex_history_watchdog.sh - codex_loop run history coverage monitor

オプション:
  -H PATH   監視対象の run_history.jsonl パス (デフォルト: results/codex_cli_runs/run_history.jsonl)
  -L LABEL  run_label でフィルタ
  -d DAYS   直近 N 日のみを対象 (--last-days)
  -a HOURS  直近 N 時間のみを対象 (--last-hours)
  -S TS     ISO 形式 (YYYY-MM-DD[THH:MM:SS]) で since を指定 (--since)
  -m NUM    最低 run 件数 (--min-runs)
  -p NUM    累計プロンプト件数 (--min-prompts)
  -u NUM    各 run の最低プロンプト件数 (--min-prompts-per-run)
  -B MODE  backfilled エントリの扱い (include|exclude|only) (--backfilled)
  -P PATH   prompts_file でフィルタ (--prompts-file)
  -k NUM    表示する行数 (--limit)。0 で全件
  -F        失敗 run があれば exit 3 (--fail-when-failure)
  -R        マッチが無い場合 exit 2 (--require-matches)
  -n        Dry-run。実行コマンドのみ表示
  -h        このヘルプを表示

環境変数:
  PYTHON_BIN, HISTORY_FILE, RUN_LABEL, LAST_DAYS, LAST_HOURS, SINCE,
  MIN_RUNS, MIN_PROMPTS, MIN_PROMPTS_PER_RUN, BACKFILLED_MODE, PROMPTS_FILE,
  LIMIT, FAIL_WHEN_FAILURE, REQUIRE_MATCHES を上書き可能。
EOF
}

log() {
  printf '📊 [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

to_abs() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

while getopts "hH:L:d:a:S:m:p:u:B:P:k:FRn" opt; do
  case "$opt" in
    h)
      usage
      exit 0
      ;;
    H)
      HISTORY_FILE="$OPTARG"
      ;;
    L)
      RUN_LABEL="$OPTARG"
      ;;
    d)
      LAST_DAYS="$OPTARG"
      ;;
    a)
      LAST_HOURS="$OPTARG"
      ;;
    S)
      SINCE="$OPTARG"
      ;;
    m)
      MIN_RUNS="$OPTARG"
      ;;
    p)
      MIN_PROMPTS="$OPTARG"
      ;;
    u)
      MIN_PROMPTS_PER_RUN="$OPTARG"
      ;;
    B)
      BACKFILLED_MODE="$OPTARG"
      ;;
    P)
      PROMPTS_FILE="$OPTARG"
      ;;
    k)
      LIMIT="$OPTARG"
      ;;
    F)
      FAIL_WHEN_FAILURE=1
      ;;
    R)
      REQUIRE_MATCHES=1
      ;;
    n)
      DRY_RUN=1
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ ! -f "$HISTORY_REPORT" ]]; then
  log "❌ codex_history_report.py が見つかりません: $HISTORY_REPORT"
  exit 1
fi

HISTORY_FILE="$(to_abs "$HISTORY_FILE")"

cmd=(
  "$PYTHON_BIN"
  "$HISTORY_REPORT"
  --history "$HISTORY_FILE"
  --limit "${LIMIT}"
)

if [[ -n "$RUN_LABEL" ]]; then
  cmd+=(--label "$RUN_LABEL")
fi
if [[ -n "$LAST_DAYS" ]]; then
  cmd+=(--last-days "$LAST_DAYS")
fi
if [[ -n "$LAST_HOURS" ]]; then
  cmd+=(--last-hours "$LAST_HOURS")
fi
if [[ -n "$SINCE" ]]; then
  cmd+=(--since "$SINCE")
fi
if [[ -n "$MIN_RUNS" ]]; then
  cmd+=(--min-runs "$MIN_RUNS")
fi
if [[ -n "$MIN_PROMPTS" ]]; then
  cmd+=(--min-prompts "$MIN_PROMPTS")
fi
if [[ -n "$MIN_PROMPTS_PER_RUN" ]]; then
  cmd+=(--min-prompts-per-run "$MIN_PROMPTS_PER_RUN")
fi
if [[ -n "$BACKFILLED_MODE" ]]; then
  cmd+=(--backfilled "$BACKFILLED_MODE")
fi
if [[ -n "$PROMPTS_FILE" ]]; then
  cmd+=(--prompts-file "$(to_abs "$PROMPTS_FILE")")
fi
if [[ -n "$FAIL_WHEN_FAILURE" ]]; then
  cmd+=(--fail-when-failure)
fi
if [[ -n "$REQUIRE_MATCHES" ]]; then
  cmd+=(--require-matches)
fi

log "監視対象: $HISTORY_FILE"
if [[ -n "$RUN_LABEL" ]]; then
  log "run_label filter: $RUN_LABEL"
fi
if [[ -n "$LAST_DAYS" ]]; then
  log "window: last ${LAST_DAYS} day(s)"
fi
if [[ -n "$LAST_HOURS" ]]; then
  log "window: last ${LAST_HOURS} hour(s)"
fi
if [[ -n "$SINCE" ]]; then
  log "since: $SINCE"
fi
if [[ -n "$MIN_RUNS" ]]; then
  log "min runs: $MIN_RUNS"
fi
if [[ -n "$MIN_PROMPTS" ]]; then
  log "min prompts: $MIN_PROMPTS"
fi
if [[ -n "$MIN_PROMPTS_PER_RUN" ]]; then
  log "min prompts/run: $MIN_PROMPTS_PER_RUN"
fi
if [[ -n "$BACKFILLED_MODE" ]]; then
  log "backfilled mode: $BACKFILLED_MODE"
fi
if [[ -n "$PROMPTS_FILE" ]]; then
  log "prompts file filter: $(to_abs "$PROMPTS_FILE")"
fi
if [[ -n "$FAIL_WHEN_FAILURE" ]]; then
  log "fail when failure: enabled"
fi
if [[ -n "$REQUIRE_MATCHES" ]]; then
  log "require matches: enabled"
fi
log "limit: ${LIMIT}"

if (( DRY_RUN )); then
  printf '🧪 Dry-run command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

set +e
"${cmd[@]}"
exit_code=$?
set -e

if [[ $exit_code -eq 0 ]]; then
  log "✅ coverage OK"
else
  log "❌ watchdog exit ${exit_code}"
fi

exit "$exit_code"
