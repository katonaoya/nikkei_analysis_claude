# Codex-A ハンドオーバーまとめ (2025-10-14)

## 1. 完了した実装・設定
- **候補データ基盤**
  - `analysis/build_multi_model_candidate_dataset.py --append-output --down-thresholds=-0.01,-0.015,-0.02` により、120 営業日（2025-04-17〜2025-10-10）の候補データを構築。
  - 既存ファイルへ追記しつつ最新 `lookback_days` のみ保持する仕組みを導入。
- **下落モデル**
  - 11 種の追加特徴量（30日モメンタム、10/20日ボラ、ATR、ギャップ率、業種モメンタム等）を `systems/downside_risk_system_v1.py` に統合。
  - `--down-thresholds` オプションで複数ラベル（`down_target_1pct` など）を出力。
  - 再学習済みモデルから最新の `production_data/downside_predictions.parquet` を生成。
- **キャリブレーション / モニタリング**
  - `analysis/downside_calibration_report.py` を追加し、Brier/ECE を `analysis/downside_calibration_metrics_latest.csv` に自動保存。
  - `analysis/log_multi_model_metrics.py` で日次 Precision / AvgReturn / Coverage を記録し、最新月のみ `production_data/multi_model_metrics.csv` に保持、過去分を `production_data/multi_model_metrics_archive/` に退避。
- **閾値探索**
  - `analysis/multi_model_threshold_optimizer.py` を拡張し、`--fallback-min-passed-ratio-grid`・`--fallback-max-per-sector`・`precision_gap`/`coverage_gap` などをサポート。
  - 120 営業日データで再スイープし、結果を `analysis/multi_model_threshold_grid_extended.csv` に出力。
- **フォールバック制御**
  - `select_top_candidates` に `fallback_max_per_sector` を導入し、セクター別フォールバック件数を制限可能に。
  - `config/multi_model_recommendation.json` に `fallback.max_per_sector=1` を追加。
- **パイプライン更新**
  - `daily_trading_automation.py` は全 10 ステップ構成となり、候補生成 → 指標ログ → キャリブレーション → レポート生成まで自動化。
- **レポート**
  - `production_reports/2025-10/multi_model/2025-10-10_multi.md` のフォーマットを更新し、直近14営業日の Precision / AvgReturn / Coverage を掲載。

## 2. 主要ファイル
- モデル / データ
  - `production_data/downside_predictions.parquet`
  - `production_data/multi_model_candidates.parquet`
  - `production_data/multi_model_metrics.csv`
  - `production_data/multi_model_metrics_archive/`
  - `analysis/downside_calibration_metrics_latest.csv`
  - `analysis/multi_model_threshold_grid_extended.csv`
- スクリプト
  - `systems/downside_risk_system_v1.py`
  - `analysis/downside_calibration_report.py`
  - `analysis/log_multi_model_metrics.py`
  - `analysis/multi_model_threshold_optimizer.py`
  - `analysis/multi_model_precision_report.py`
  - `reports/daily_stock_recommendation_multi.py`
  - `daily_trading_automation.py`
- ドキュメント
  - `CodexA_handover_notes_2025-10-14.md`
  - `CodexA_CodexB_alignment_2025-10-14.md`
  - 本ファイル `CodexA_handover_full_2025-10-14.md`

## 3. 現状の主な指標
- フォールバック無し (`threshold_up=0.18`, `down=0.52`, `risk=0.45`, 重み 1.0/0.5/0.4)
  - Precision **41.46%**, Average Return **+0.59%**, Coverage **55%**。
- フォールバックあり (`max_fallback=1`, `min_passed_all=2`, `min_passed_ratio=0.45`, `max_per_sector=1`)
  - Precision **23.35%**, Average Return **+0.12%**, Coverage **100%**。
- 下落モデルキャリブレーション（最新）
  - Brier Score **0.2600**、ECE **0.5096**。

## 4. 未完了タスク / 次エージェントへの引き継ぎ
1. **下落モデルの追加特徴量再学習**
   - 業種モメンタム・指数差分・ギャップ方向ラベルなどを含むリファイン版を学習し、Precision/Brier/ECE を再評価する。
   - 再学習後、120 営業日の候補データを再生成し、閾値・フォールバックを再探索する。
2. **Precision ≥45% / Coverage ≥60% の両立検討**
   - `threshold_up` 0.19〜0.22、`fallback_min_passed_ratio` 0.45〜0.55、`fallback_max_per_sector` の調整案を中心に、再学習データで再サーチする。
   - `analysis/multi_model_threshold_optimizer.py` の `precision_gap` / `coverage_gap` を確認しながら ベスト案を絞る。
3. **モニタリングと運用ポリシーの確定**
   - `production_data/multi_model_metrics.csv`／`production_data/multi_model_metrics_archive/` のバックアップ・保管先を決定（例: 月次で Git 管理 or 外部ストレージ）。
   - `analysis/downside_calibration_report.py` の監視閾値（例: Brier > 0.20）とリトレーニングトリガーを定義する。
4. **レポート拡張**
   - 必要に応じて、日次レポートへフォールバック採用率やセクター別分析など追加指標を表示する。

## 5. 参考コマンド
```bash
# 下落モデル再学習（複数閾値出力）
PYTHONPATH=. python systems/downside_risk_system_v1.py \
  --predict-date 2025-10-10 --retrain \
  --down-thresholds=-0.01,-0.015,-0.02

# 閾値グリッド（フォールバック比率＆セクター制限込み）
PYTHONPATH=. python analysis/multi_model_threshold_optimizer.py \
  --input production_data/multi_model_candidates.parquet \
  --threshold-up-grid 0.18,0.19,0.20,0.21,0.22 \
  --threshold-down-grid 0.48,0.50,0.52 \
  --risk-grid 0.40,0.45 \
  --metric precision --metric-weights precision:0.6,avg_net_return:0.3,coverage_rate:0.1 \
  --allow-fallback --fallback-max 1 --fallback-min-passed 2 \
  --fallback-min-passed-ratio-grid 0.35,0.40,0.45,0.50 \
  --fallback-max-per-sector 1 --export-csv analysis/multi_model_threshold_grid_extended.csv

# Precision レポート（フォールバック指定）
PYTHONPATH=. python analysis/multi_model_precision_report.py \
  --input production_data/multi_model_candidates.parquet \
  --fallback-max 1 --fallback-min-passed 2 \
  --fallback-min-passed-ratio 0.45 --fallback-max-per-sector 1

# 日次メトリクス（最新月保持＋月次アーカイブ）
PYTHONPATH=. python analysis/log_multi_model_metrics.py \
  --input production_data/multi_model_candidates.parquet \
  --allow-fallback --fallback-max 1 --fallback-min-passed 2 \
  --fallback-min-passed-ratio 0.45 --fallback-max-per-sector 1 \
  --days 120 --archive-dir production_data/multi_model_metrics_archive --keep-months 1
```

## 6. ドキュメント確認先
- `CodexA_handover_notes_2025-10-14.md` — 簡易サマリ
- `CodexA_CodexB_alignment_2025-10-14.md` — Codex-B との協議ポイント
- `CodexB_hand_over_2025-10-14.md` — Codex-B 側ハンドオーバー資料

---
次エージェントは上記の TODO に優先度を付け、Precision 向上と Coverage 確保のバランスを再検証してください。
