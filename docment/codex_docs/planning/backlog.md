# バックログ

マイルストーンを達成するための候補タスクを列挙します。実際に着手する際は該当行を `task_progress/current.md` へコピーし、ここではステータスをリンク参照に置き換えてください。

| ID | タスク概要 | 目的 / 期待成果 | 予定コマンド / 検証 | 優先度 | 状態 | 備考 |
| --- | --- | --- | --- | --- | --- | --- |
| BL-005 | Precision低下区間（min 0.7570）の要因調査 | 直近6区間での最低 Precision 期間の特徴量・銘柄・市況を把握し改善施策を立案 | `python - <<'PY' ...` で対象期間のデータを抽出し、`analysis/precision_dive.py`（新規）で影響度解析 | 高 | 📝 未着手 | `performance_history.md` 2025-11-07の最新行を参照 |
| BL-004 | Kabutan / 高頻度データ統合PoC | 精度下振れ時の情報量を補強し、直近Precisionを底上げ | データ取得スクリプト試作→ `systems/enhanced_precision_system_v3.py` へ特徴量追加→再評価 | 高 | 📝 未着手 | MS-002 連動。ライセンス調査とログ実装が前提 |
| BL-003 | 特徴量候補リストの洗い出し | Precision 60% 達成に必要な追加特徴量を網羅 | `analysis/feature_inventory.py` (作成予定) のドラフト | 中 | 📝 未着手 | MS-002 に紐づく |
| BL-002 | 非対話ループ自動起動スクリプト案 | Codex CLI を定期実行する仕組みを設計 | `scripts/codex_loop.sh` のモック + 実行ステップメモ | 高 | 🔄 進行中 (current.md 2025-11-07) | MS-001 に紐づく |
| BL-001 | task_progress 運用ドリル | `current.md` ~ `completed.md` 更新フローの確認 | ダミータスクを登録→完了移行する手順書 | 中 | ✅ 完了 (2025-02-17) | 詳細: `docment/codex_docs/task_progress/completed.md` |

> **使い方**
> - 優先度は `高 / 中 / 低` のいずれかで記載。
> - 具体的なコマンドやスクリプト名が未確定でも、想定ツールや検証方法を記入する。
> - タスクを `current.md` に移したら、状態を `移管済` とし、備考に該当行のリンクや日付を残す。
