# Codex-B ハンドオーバー資料 (2025-10-14)

## 1. 作業概要
- **データ整備**: Codex-A の 60〜120 営業日候補データを取り込み、`analysis/build_multi_model_candidate_dataset.py` で再生成。最新ファイル:
  - `production_data/multi_model_candidates.parquet`（120 営業日, 18,000 行）
  - `production_data/multi_model_candidates_history.parquet`
- **閾値探索**: `analysis/multi_model_threshold_optimizer.py` をメトリクス重み付きで実行し、Precision 41.46% / AvgReturn +0.59% / Coverage 55% の組み合わせ（`threshold_up=0.18`, `down=0.52`, `risk=0.45`, 重み 1.0/0.5/0.4）を採用。フォールバック有無のシナリオを `analysis/multi_model_threshold_grid_fallback_refined.csv` に出力。
- **レポート強化**: `reports/daily_stock_recommendation_multi.py` に「📊 指標サマリ」セクションを追加し、候補／推奨の平均確率・統合スコア・未来リターンを表示。`production_reports/2025-10/multi_model/` に最新レポートを配置済み。
- **日次モニタリング**: `analysis/log_multi_model_metrics.py` で日別 Precision / AvgReturn / Coverage を `production_data/multi_model_metrics.csv` に追記。レポート末尾には直近14営業日の推移が表示される。
- **下落モデル監視**: `analysis/downside_calibration_report.py` により Brier / ECE を集計 (`analysis/downside_calibration_metrics_latest.csv`)。
- **フォールバックシミュレーション**: `config/multi_model_recommendation.json` に `fallback.max_fallback`, `min_passed_all`, `min_passed_ratio` を追加。CLI オプションでフォールバック戦略を即時試算可能。
- **ドキュメント更新**: `CodexB_進捗報告_2025-10-12.md`, `CodexA_CodexB_alignment_2025-10-14.md` に状況と次ステップを反映。

## 2. 評価サマリ
|閾値 (up / down / risk)|フォールバック|Precision|平均リターン|Coverage|備考|
|---|---|---|---|---|---|
|0.18 / 0.52 / 0.45|無効 (`max_fallback=0`)|41.46%|+0.59%|55%|現行ベースライン|
|0.20 / 0.52 / 0.45|無効|45.45%|+1.08%|9%|高 Precision / 低 Coverage|
|0.18 / 0.52 / 0.45|`max=1`, `min_passed_ratio=0.45`|25.14%|+0.13%|100%|フォールバック有, 精度低下|
|0.18 / 0.52 / 0.45|`max=1`, `min_passed_all=1`|35.82%|+0.47%|100%|フォールバック有|

**結論**: Precision ≥45% と Coverage ≥60% を現行データで同時達成する組合せは未発見。フォールバック導入で Coverage は伸びるが Precision が 30% 台まで低下。

## 3. スクリプト & コマンド
- 候補データ生成:
  ```bash
  PYTHONPATH=. python analysis/build_multi_model_candidate_dataset.py \
    --lookback-days 120 --max-candidates 150 \
    --down-thresholds -0.01,-0.015,-0.02 \
    --append-output --temp-dir tmp/multi_candidate_builder
  ```
- 閾値サーチ (フォールバック込み):
  ```bash
  PYTHONPATH=. python analysis/multi_model_threshold_optimizer.py \
    --input production_data/multi_model_candidates.parquet \
    --threshold-up-grid 0.18,0.19,0.20,0.21,0.22 \
    --threshold-down-grid 0.48,0.50,0.52 \
    --risk-grid 0.35,0.40,0.45 \
    --metric precision --metric-weights precision:0.5,avg_net_return:0.3,coverage_rate:0.2 \
    --fallback-max 1 --fallback-min-passed 2 --fallback-min-passed-ratio-grid 0.45,0.5,0.55 \
    --min-valid-count 30 --top-k 20 \
    --export-csv analysis/multi_model_threshold_grid_fallback_refined.csv
  ```
- Precision レポート (フォールバック指定可):
  ```bash
  PYTHONPATH=. python analysis/multi_model_precision_report.py \
    --input production_data/multi_model_candidates.parquet \
    --threshold-up 0.18 --threshold-down 0.52 --risk-threshold 0.45 \
    --fallback-max 1 --fallback-min-passed 1 --fallback-min-passed-ratio 0.4
  ```
- 下落モデルキャリブレーション:
  ```bash
  PYTHONPATH=. python analysis/downside_calibration_report.py \
    --predictions production_data/downside_predictions.parquet \
    --export-csv analysis/downside_calibration_metrics_latest.csv
  ```
- 日次ログ更新:
  ```bash
  PYTHONPATH=. python analysis/log_multi_model_metrics.py \
    --input production_data/multi_model_candidates.parquet \
    --output production_data/multi_model_metrics.csv \
    --days 14 --target-return 0.01
  ```

## 4. フォローアップタスク
1. **フォールバック戦略**: Precision ≥45% & Coverage ≥60% を満たすため、`threshold_up` 引き上げ・`fallback_min_passed_ratio` 強化・セクター別制限などを Codex-A と協議。
2. **ログ保全方針**: `production_data/multi_model_metrics.csv` のローテーション／バックアップ（例: 月次アーカイブ）を定義し、パイプラインに組み込み。
3. **キャリブレーション監視**: Brier/ECE の監視頻度と再学習トリガー（例: Brier > 0.20）を決定し、ダッシュボードやレポートに反映。
4. **特徴量強化**: Codex-A が検証中の業種モメンタム等を受領後、再学習・再サーチを実施し Precision 45%以上を目指す。
5. **ウォークフォワード**: 新閾値＆フォールバック設定で日次レポートを運用し、`production_data/multi_model_metrics.csv` の推移を定期的にレビュー。

## 5. 参照ファイル
- `analysis/multi_model_threshold_grid_fallback_refined.csv`
- `analysis/multi_model_fallback_summary.csv`
- `analysis/multi_model_precision_report.py`
- `analysis/log_multi_model_metrics.py`
- `analysis/downside_calibration_report.py`
- `production_data/multi_model_candidates.parquet`
- `production_data/multi_model_metrics.csv`
- `analysis/downside_calibration_metrics_latest.csv`
- `production_reports/2025-10/multi_model/`
- `CodexA_CodexB_alignment_2025-10-14.md`

以上を参考に、Codex-B からの引き継ぎをお願いします。
