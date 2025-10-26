import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def read_prompts_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="codex CLI をループで呼ぶ最小ラッパー")
    parser.add_argument("--prompts", required=True, help="各行が1プロンプトのテキストファイル")
    parser.add_argument(
        "--codex-cmd",
        default="codex",
        help="実行するcodexコマンド名（パス）。例: codex",
    )
    parser.add_argument(
        "--extra",
        default="--approvals full",
        help="codex CLI にそのまま渡す追加引数（空白区切り）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("results") / "codex_cli_runs"),
        help="出力先ディレクトリ",
    )

    args = parser.parse_args(argv)

    prompts_file = Path(args.prompts)
    prompts = read_prompts_lines(prompts_file)
    if not prompts:
        print("[警告] プロンプトが1件も見つかりません。", file=sys.stderr)
        return 0

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    texts_dir = out_dir / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 出力先: {out_dir}")
    print(f"🛠️ codex: {args.codex_cmd}")
    print(f"⚙️ 追加引数: {args.extra}")

    extra_tokens = shlex.split(args.extra) if args.extra else []

    for i, prompt in enumerate(prompts, start=1):
        print(f"🚀 {i}/{len(prompts)} 実行中…")
        cmd = [args.codex_cmd, "--prompt", prompt, *extra_tokens]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            print(
                f"[エラー] codex コマンドが見つかりません: {args.codex_cmd}",
                file=sys.stderr,
            )
            return 2

        output_path = texts_dir / f"{i:04d}.txt"
        if proc.returncode == 0:
            output_path.write_text(proc.stdout or "", encoding="utf-8")
        else:
            output_path.write_text(
                (proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""),
                encoding="utf-8",
            )
            print(f"❌ 失敗: returncode={proc.returncode}", file=sys.stderr)

    print("✅ 完了。出力は texts/ を確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


