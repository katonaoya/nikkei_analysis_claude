#!/usr/bin/env python3
"""codex_docs の最新コンテキストを読み込み、ループごとの行動を同期させるランチャー。"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = Path(__file__).resolve().parent / "codex_docs"
PROMPT_FILE = DOCS_ROOT / "prompt.md"
REQUIREMENTS_FILE = DOCS_ROOT / "requirements_summary.md"
TASK_PROGRESS_FILE = DOCS_ROOT / "task_progress.md"
SEARCH_REQUESTS_FILE = DOCS_ROOT / "検索リクエスト.md"

BASE_CMD: List[str] = [
    "codex",
    "--search",
    "--sandbox",
    "workspace-write",
    "--ask-for-approval",
    "on-request",
    "-c",
    'sandbox_permissions=["network-outbound"]',
    "-c",
    "shell_environment_policy.inherit=all",
    "exec",
]


def read_section(path: Path) -> str:
    """Return the trimmed file contents or raise when the context is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required context file is missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def gather_context() -> Dict[str, str]:
    return {
        "prompt": read_section(PROMPT_FILE),
        "requirements": read_section(REQUIREMENTS_FILE),
        "task_progress": read_section(TASK_PROGRESS_FILE),
        "search_requests": read_section(SEARCH_REQUESTS_FILE),
    }


def build_codex_prompt(context: Dict[str, str]) -> str:
    workflow = dedent(
        f"""
        あなたは Codex (GPT-5) であり、{REPO_ROOT} からヘッドレス実行されています。
        各ループで必ず次のワークフローを実行してください:
        1. 以下の参照セクション（プロンプト/要件/task_progress/検索リクエスト）をすべて読み、状況と制約を整理する。
        2. 目的と制約を短く要約し、ファイル変更前に3ステップ以上の計画を文章で示す。
        3. 既存パイプラインを壊さず requirements に最も寄与する作業を実装する。
        4. 関連テストやリンタを実行し、未実行なら理由とリスクを明記する。
        5. 実装後は docment/codex_docs/task_progress.md 先頭へ、作業内容・実行コマンド/テスト・成果物・次アクションを追記する。
        6. レスポンスでは file:line 付きで変更内容を説明し、実行テストと具体的な次ステップ案を提示する。
        7. ブロッカーや調査ニーズがあれば docment/codex_docs/検索リクエスト.md を更新し、解消状況を管理する。
        8. AGENTS.md の規約（Python3.9/Black/絵文字ロガー/秘密保持など）を常に順守する。
        """
    ).strip()

    sections = [
        ("参照 • prompt.md", context["prompt"]),
        ("参照 • requirements_summary.md", context["requirements"]),
        ("参照 • task_progress.md", context["task_progress"]),
        ("参照 • 検索リクエスト.md", context["search_requests"]),
    ]
    reference_text = "\n\n".join(f"## {title}\n{body}" for title, body in sections)

    mission = dedent(
        """
        # ミッション
        このループで株式レコメンドシステムを要件達成へ近づけるために、以下を指針に行動すること:
        - 翌営業日 +1% 上昇の Precision を 60%以上（目標 70〜75%）で維持しつつ、日次3〜5銘柄の推奨を欠かさない。
        - 日経225 過去10年以上の実績データを使った堅牢な時系列CVを前提にし、閾値だけに頼った見かけ精度向上を避ける。
        - 信頼性フェーズの次アクション確定や Kabutan/Yahoo!/高頻度データのライセンス・取得ログ実装を優先的に前進させる。
        - ループ毎に必ず具体的なコード/ドキュメント改善を届ける。ブロックされた場合は課題・必要調査・次善策を明確に残す。
        """
    ).strip()

    return f"{workflow}\n\n{reference_text}\n\n{mission}"


def run_cmd(args: List[str]) -> None:
    subprocess.run(args, check=True, env=os.environ.copy(), cwd=str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex を連続実行するラッパースクリプト")
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Codex を連続実行する回数（デフォルト: 1）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = max(1, args.loops)
    for idx in range(1, total + 1):
        print(f"🔁 Codex 実行 {idx}/{total}")
        context = gather_context()
        prompt = build_codex_prompt(context)
        run_cmd(BASE_CMD + [prompt])


if __name__ == "__main__":
    main()
