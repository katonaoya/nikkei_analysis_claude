# 2025-11-07 ループログ
- 作業内容: BL-002 を current に移管し、`scripts/codex_loop.sh` とサンプル `codex_loop_prompts.sample.txt` を追加して codex CLI 非対話ループの土台を構築。
- 実行コマンド/テスト: `bash -n scripts/codex_loop.sh`。シンタックス検証のみで実実行は未実施（codex CLI への外部アクセスが不要なためリスク低）。
- 成果物: `scripts/codex_loop.sh`, `docment/codex_docs/codex_loop_prompts.sample.txt`, backlog/current/task_progress ログ更新。
- 次アクション: DRY-RUN運用手順を documentation へ追加し、codex_loop のスケジューリング戦略 (cron or launchd) を洗い出す。

# 2025-02-17 ループログ
- 作業内容: BL-001 task_progress 運用ドリルを完了し、current/completed/backlog の同期と手順ドキュメントを整備。
- 実行コマンド/テスト: `apply_patch` で各Markdownを更新。テストはドキュメントのみのため未実施（リスク低）。
- 成果物: `docment/codex_docs/task_progress_flow.md`、ログ更新（`task_progress/current.md`, `task_progress/completed.md`, `planning/backlog.md`）。
- 次アクション: 次ループで BL-002 もしくは特徴量関連タスクへ着手し、テスト実行フローを組み込む。

# タスク進捗インデックス

進行中と完了済みのタスク管理を以下の2ファイルに分割しました。必ず両方を更新してください。

- `docment/codex_docs/task_progress/current.md` : 進行中タスクのみを記載。着手直後から記録し、完了時に行を削除して完了ログへ転記します。
- `docment/codex_docs/task_progress/completed.md` : 完了済みタスクの履歴。実施内容・実行コマンド/テスト・成果物・学び/フォローアップを詳細に書き残します。

## 運用ポリシー
1. ループ開始時は必ず `current.md` で未完タスクを確認し、必要なら状態を更新する。
2. 作業が完了したら、詳細な報告（実装内容、実行コマンドやテスト結果、生成物パス、残課題）を `completed.md` に追記し、`current.md` から該当行を削除する。
3. 途中でブロックされた場合も `current.md` に理由と解除条件を明記し、次のエージェントが即時対応できるようにする。
4. 大規模タスクは小さなサブタスクに分解し、各サブタスクを独立して記録・完了報告する。
