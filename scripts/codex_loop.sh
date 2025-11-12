#!/usr/bin/env bash
# codex CLI を非対話モードで複数プロンプトへ順次実行するユーティリティ。
# 使い方:
#   scripts/codex_loop.sh -p path/to/prompts.txt
#   PROMPTS_FILE=docment/codex_docs/prompts.txt EXTRA_ARGS="--approvals full --max-tokens 2000" scripts/codex_loop.sh
#
# オプション:
#   -p <path>  : プロンプト一覧ファイル（1行1プロンプト）。省略時は sample ファイルを参照
#   -o <dir>   : codex_loop.py の output-dir。デフォルト: results/codex_cli_runs
#   -c <cmd>   : 実行する codex CLI コマンド。デフォルト: codex
#   -e <args>  : codex CLI へ渡す追加引数文字列 (`codex_loop.py --extra` 相当)
#   -l <dir>   : ログ出力先ディレクトリ。デフォルト: logs/codex_loop
#   -n         : Dry-run。実行コマンドを表示して終了
#   -h         : ヘルプを表示

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_PROMPTS_FILE="$ROOT_DIR/docment/codex_docs/codex_loop_prompts.sample.txt"
DEFAULT_OUTPUT_DIR="$ROOT_DIR/results/codex_cli_runs"
DEFAULT_LOG_DIR="$ROOT_DIR/logs/codex_loop"
CODEX_LOOP_ENTRY="$ROOT_DIR/codex_loop.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CODEX_CMD="${CODEX_CMD:-codex}"
EXTRA_ARGS="${EXTRA_ARGS:---approvals full}"
RUN_LABEL="${RUN_LABEL:-}"

PROMPTS_FILE="$DEFAULT_PROMPTS_FILE"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
LOG_DIR="$DEFAULT_LOG_DIR"
DRY_RUN=0

usage() {
  cat <<'EOF'
codex_loop.sh - codex CLI の非対話バッチ実行ヘルパー

必須準備:
  1. docment/codex_docs/ 以下に 1 行 1 プロンプトのテキストファイルを作成
  2. codex CLI が PATH にあり、--prompt オプションで当該ファイルを読めること

オプション:
  -p PATH   プロンプト一覧ファイル。省略時は codex_loop_prompts.sample.txt
  -o DIR    出力ディレクトリ (codex_loop.py --output-dir)。デフォルト: results/codex_cli_runs
  -c CMD    実行する codex コマンド名 or パス。デフォルト: codex
  -e ARGS   追加引数文字列 (`codex_loop.py --extra`)。例: '--approvals full --max-output 1'
  -l DIR    ログ保存ディレクトリ。デフォルト: logs/codex_loop
  -n        Dry-run。実行コマンドを表示して終了
  -h        このメッセージを表示

環境変数:
  PROMPTS_FILE, OUTPUT_DIR, LOG_DIR, CODEX_CMD, EXTRA_ARGS, PYTHON_BIN, RUN_LABEL を指定可能
EOF
}

log() {
  printf '📝 [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

to_repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT_DIR" "$path"
  fi
}

while getopts "hp:o:c:e:l:n" opt; do
  case "$opt" in
    h)
      usage
      exit 0
      ;;
    p)
      PROMPTS_FILE="$OPTARG"
      ;;
    o)
      OUTPUT_DIR="$OPTARG"
      ;;
    c)
      CODEX_CMD="$OPTARG"
      ;;
    e)
      EXTRA_ARGS="$OPTARG"
      ;;
    l)
      LOG_DIR="$OPTARG"
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

PROMPTS_FILE="$(to_repo_path "$PROMPTS_FILE")"
OUTPUT_DIR="$(to_repo_path "$OUTPUT_DIR")"
LOG_DIR="$(to_repo_path "$LOG_DIR")"

if [[ ! -f "$CODEX_LOOP_ENTRY" ]]; then
  log "❌ codex_loop.py が見つかりません: $CODEX_LOOP_ENTRY"
  exit 1
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  if [[ "$PROMPTS_FILE" == "$DEFAULT_PROMPTS_FILE" ]]; then
    cat <<'EOF' >"$PROMPTS_FILE"
# codex_loop 用のサンプルリスト。1 行 1 プロンプトで記述します。
# コメント行 (#) と空行は無視されます。
# 下記はダミー行です。必要な CLI プロンプトを追記してください。
docment/codex_docs/prompt.md
EOF
    log "📄 サンプルファイルを生成しました: $PROMPTS_FILE"
  fi
  log "⚠️ プロンプトファイルが存在しません。-p で指定するか、サンプルを編集してください。"
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

timestamp="$(date '+%Y%m%d_%H%M%S')"
if [[ -n "$RUN_LABEL" ]]; then
  RUN_PREFIX="${timestamp}_${RUN_LABEL}"
else
  RUN_PREFIX="${timestamp}"
fi
LOG_FILE="$LOG_DIR/${RUN_PREFIX}.log"

cmd=(
  "$PYTHON_BIN"
  "$CODEX_LOOP_ENTRY"
  --prompts "$PROMPTS_FILE"
  --codex-cmd "$CODEX_CMD"
  --extra "$EXTRA_ARGS"
  --output-dir "$OUTPUT_DIR"
)

log "🏁 実行準備完了。ログ: $LOG_FILE"
log "📁 prompts: $PROMPTS_FILE"
log "📦 output:  $OUTPUT_DIR"
log "🛠️ codex : $CODEX_CMD"
log "🎛  extra : $EXTRA_ARGS"

if (( DRY_RUN )); then
  printf '🧪 Dry-run command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

set +e
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e

if [[ $exit_code -eq 0 ]]; then
  log "✅ codex_loop.py が正常終了しました。ログ: $LOG_FILE"
else
  log "❌ codex_loop.py が異常終了しました (exit=${exit_code})。ログを確認してください。"
fi

exit "$exit_code"
