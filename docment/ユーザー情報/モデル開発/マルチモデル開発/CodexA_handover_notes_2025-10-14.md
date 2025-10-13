# Codex-A ハンドオーバーノート (2025-10-14 10:05)

## 1. 完了状況
- `analysis/build_multi_model_candidate_dataset.py --append-output --down-thresholds=-0.01,-0.015,-0.02` により、120 営業日分（2025-04-17〜2025-10-10）の候補データを構築。
- 閾値ベースライン: `threshold_up=0.18`, `threshold_down=0.52`, `threshold_risk=0.45`、重み `1.0/0.5/0.4`。
  - Precision 41.46%、Average Return +0.59%、Coverage 55%（フォールバック無し）。
  - `analysis/multi_model_threshold_grid_extended.csv` にフォールバック比率（0.3 / 0.4 / none）まで含むグリッド結果を保存。
- 日次メトリクスログ (`analysis/log_multi_model_metrics.py → production_data/multi_model_metrics.csv`) を追加し、リポート末尾で直近14営業日の Precision / AvgReturn / Coverage を表示。古い月次データは `production_data/multi_model_metrics_archive/` に自動退避。
- 下落モデルのキャリブレーション指標は `analysis/downside_calibration_report.py` で算出し、最新結果を `analysis/downside_calibration_metrics.csv` に保存。
- Codex-B との最新アラインメント: `docment/ユーザー情報/モデル開発/マルチモデル開発/CodexA_CodexB_alignment_2025-10-14.md` を参照。
- `systems/downside_risk_system_v1.py --down-thresholds=-0.01,-0.015,-0.02 --retrain` を実行し、`production_data/downside_predictions.parquet` を複数ラベル (`down_target_1pct` など) 付きに更新済み。
- `daily_trading_automation.py` にキャリブレーション出力ステップを追加し、日次で `analysis/downside_calibration_metrics_latest.csv` を更新。
- フォールバック設定に `max_per_sector=1` を追加し、セクターごとのフォールバック採用数を制御可能にした。

## 2. 未完了タスク / TODO
1. **下落モデルの再学習**
   - 新特徴量（業種モメンタムの強化、指数差分、ギャップ方向ラベルなど）を投入した再学習を予定。Brier/ECE を改善し Precision を底上げする。 
   - 再学習後、120 営業日候補を再生成し、閾値サーチ・フォールバック設計を再実施。

2. **閾値・フォールバックの再調整**
   - Precision ≥45%、Coverage ≥60% を満たす組み合わせを探索するため、`threshold_up` 0.19〜0.22、`fallback_min_passed_ratio` 0.45〜0.55 などを中心に再サーチ。 
   - セクター別フォールバック制限案など Codex-B の優先度確認後に実装。
   - 現行フォールバック案（max=1, min=2, ratio=0.45）では Precision 23.35%、Coverage 100%（`analysis/multi_model_precision_report.py` 出力）に留まるため、再学習後に再評価。

3. **ログ・キャリブレーション運用**
   - `production_data/multi_model_metrics.csv` は最新月のみ保持し、それ以前を `production_data/multi_model_metrics_archive/` へ退避。バックアップ方針（例: 月次で Git 管理 or S3 保存）を決定。 
   - `analysis/downside_calibration_report.py` は日次パイプラインで実行済み。Brier/ECE の監視閾値（例: Brier ≤ 0.20）と再学習トリガーを設定。

## 3. 次のステップ提案
1. 下落モデル追加特徴量の再学習を実施、Precision / Brier / ECE をまとめたレポートを作成。
2. `analysis/multi_model_threshold_optimizer.py` を再スイープし、Precision / Coverage ギャップの改善状況を Codex-B と共有。
3. 日次ログ運用とキャリブレーション監視のポリシーを Codex-B と合意し、`daily_trading_automation.py` に必要なローテーション処理を追加。

## 4. 参考資料
- `analysis/multi_model_threshold_grid_extended.csv`
- `analysis/multi_model_threshold_grid_fallback_refined.csv`
- `analysis/downside_calibration_metrics.csv`
- `production_data/multi_model_metrics.csv`
- `production_reports/2025-10/multi_model/`

---
これらのタスクを進めつつ、Precision 向上と Coverage 確保のバランスを再評価していきます。
