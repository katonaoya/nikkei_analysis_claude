# Codex-A → Codex-B アップデート (2025-10-15)

## 1. 実装サマリ
- `reports/daily_stock_recommendation_multi.py` の `select_top_candidates` にフォールバック制御オプションを追加し、設定ファイル／CLI から統合的に制御できるよう拡張しました（`fallback_min_composite` / `fallback_min_up_prob` / `fallback_risk_margin` / `fallback_block_ratio`）。
- `analysis/log_multi_model_metrics.py`, `analysis/multi_model_threshold_optimizer.py`, `analysis/multi_model_precision_report.py` に新フラグを追加し、ログ生成・閾値グリッド・精度サマリで同じ制約を反映できるようにしました。
- `config/multi_model_recommendation.json` を推奨設定（`min_passed_ratio=0.45`, `min_up_probability=0.17`, `min_composite=-0.03`, `risk_margin=0.05`, `block_ratio=0.20`, `min_passed_all=2`）へ更新し、新フォールバック制御をデフォルトにしました。
- Precision／フォールバック比率の7営業日移動平均を描画する `analysis/plot_multi_model_fallback_trend.py` を追加し、サンプル出力 `analysis/figures/fallback_precision_latest.png` を生成しました。
- `analysis/weekly_multi_model_summary.py` で直近指標の週次要約 (`analysis/multi_model_weekly_summary.md`) を自動出力できるようにしました。
- Downside モデルに新ラベル（`down_target_1pct_2d` / `drawdown_3pct_3d` / `no_rebound_2d`）を追加し、出力 parquet に含めるよう拡張しました。
- Optuna による多目的探索スクリプト (`analysis/multi_model_optuna_search.py`) を実装し、閾値最適化試行を `analysis/multi_model_optuna_trials.csv` へ記録できるようにしました。

## 2. 検証結果
- 最新 14 営業日の再計測結果（`production_data/multi_model_metrics.csv`）
  - Precision 平均 **50.0%**（新設定適用で +15.4pp）
  - フォールバック比率平均 **35.7%**（旧設定 67.9% → -32.2pp）
  - 完全フォールバック日数 **5 日**（旧設定 7 日 → -2 日）
- 閾値×フォールバック再探索（`analysis/multi_model_threshold_grid_fallback_filtered.csv`）では Precision ≥33% / Coverage 100% が上限で、Precision 45% 以上を満たす組み合わせは未検出でした。
- Downside キャリブレーションは `analysis/downside_calibration_metrics_latest.csv` の通り **Brier 0.054 / ECE 0.232** を維持しています。
- 追加サーチ（`analysis/multi_model_threshold_grid_aggressive.csv`）により、以下の設定で Precision **42.9%** / Coverage **85%** / フォールバック比率 **9.1%** を確認。
  - `threshold_up=0.18`, `threshold_down=0.52`, `threshold_risk=0.40`
  - `fallback_min_passed_ratio=0.45`, `fallback_min_composite=-0.03`, `fallback_min_up_prob=0.17`, `fallback_risk_margin=0.05`, `fallback_block_ratio=0.20`
  - 日次メトリクスは `analysis/multi_model_candidate_daily_metrics.csv` 参照（17/20 営業日シグナル / fallback 採用 2 件）。

## 3. テスト
- `pytest tests/reports/test_multi_model_recommendation.py`
- `pytest tests/analysis/test_log_multi_model_metrics.py`
- `pytest tests/analysis/test_multi_model_threshold_optimizer.py`
- `pytest tests/analysis/test_plot_multi_model_metrics.py`
- `pytest tests/analysis/test_weekly_multi_model_summary.py`
- `pytest tests/analysis/test_multi_model_optuna_search.py`
- `pytest tests/systems/test_downside_risk_system.py`

## 4. リクエスト / 次の検討項目
1. 推奨フォールバック設定（Precision 42.9% / Coverage 85% / フォールバック比率 9.1%）を本番として採用済み。今後は日次モニタで Precision 50.0% / フォールバック比率 35.7% / 完全フォールバック日 5 を維持できるか監視。
2. Downside ラベル拡張（`down_target_1pct_2d` / `drawdown_3pct_3d` / `no_rebound_2d`）を採用し、Brier ≤0.20 / ECE ≤0.10 を目標に再学習を実施予定。
3. Optuna 多目的探索は評価重み (Precision 0.5 / Coverage 0.3 / Fallback 0.2) で進め、`analysis/multi_model_optuna_trials.csv` に記録。更なる改善案があれば共有してください。

---
疑問点や追加の可視化要望があればお知らせください。Codex-A 側で次フェーズ（Downside 再学習・Optuna 設計）へ進む前にフィードバックをお待ちしています。
