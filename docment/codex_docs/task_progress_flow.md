# task_progress 運用ドリル

codex_docs で定義されたログ運用を手早く確認できるよう、current → completed の流れを具体的に記述する。ブロックや未完了を放置しないため、各ループでこの手順に沿って進める。

## 1. 事前チェック
- `docment/codex_docs/prompt.md` と `planning/` (`roadmap.md` / `milestones.md` / `backlog.md`) を読み、依存タスクや優先度を把握する。
- `docment/codex_docs/task_progress/current.md` を確認し、既存タスクの状態を更新する。未記入なら対象タスクを backlog から移管する。

## 2. current.md の更新
1. backlog から着手するタスクを選定し、`current.md` に行を追加する。
2. `日付/タスク/状態/実施内容/リスク/次アクション` をすべて埋める。
3. 例: `apply_patch` で行を追加し、`rg "BL-001" docment/codex_docs/task_progress/current.md` で反映を確認。

## 3. 作業と検証
- 実装や調査を進め、実行したコマンド（`pytest`, `python script.py`, など）を記録する。
- テストを実行できない場合は理由とリスクを `current.md` と最終レポート双方に明記する。

## 4. completed.md への転記
1. タスク完了後、`completed.md` に詳細行を追加する。
2. `実施内容・利用コマンド/テスト` に実際のコマンド文字列を記載し、`成果物` に関連ファイルへのパスを箇条書きで残す。
3. `学び・フォローアップ` には、次のループで引き継ぐ注意点や残課題をまとめる。

## 5. current.md からの削除
- completed.md への転記後、`current.md` の対応行を削除し、代わりにバックログの状態を更新して「完了済み or 移管解除」を明記する。

## 6. 監査と共有
- `git diff docment/codex_docs/task_progress` でログ変更を再確認。
- レスポンスでは file:line 付きで更新箇所を報告し、次ステップ候補を提示する。

このドリルを毎ループで実施することで、codex_docs の計画・進捗・報告が常に同期した状態を維持できる。
