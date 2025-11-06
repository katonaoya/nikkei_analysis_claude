#!/usr/bin/env python3
"""Launch Codex interactively with search enabled."""
import os
import subprocess
from typing import Sequence

PROMPT = "こんにちは。今日の東京の天気を教えて"
CMD: Sequence[str] = (
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
    PROMPT,
)


def main() -> None:
    """Run Codex interactively in the current terminal."""
    env = os.environ.copy()
    subprocess.run(CMD, check=True, env=env)


if __name__ == "__main__":
    main()
