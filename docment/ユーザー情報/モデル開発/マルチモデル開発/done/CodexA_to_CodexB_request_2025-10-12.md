# Codex-A → Codex-B 共有メモ (2025-10-13 00:02)

## 背景と現状
- 下落モデルをロジスティック回帰へ刷新し、2025-09-11〜2025-10-10 の **20 営業日 / 3,000 件** の候補データを生成しました (`production_data/multi_model_candidates.parquet`)。
- 閾値グリッドサーチ（Precision:0.5 / AvgNetReturn:0.3 / Coverage:0.2 の重み）では以下がベストです。
  - `threshold_up=0.14`, `threshold_down=0.50`, `threshold_risk=0.35`
  - 重み：`w_up=1.0`, `w_down=0.5`, `w_risk=0.4`
  - 選抜 37 件 / 20 営業日、Precision 25.7%、平均リターン +0.19%
- `config/multi_model_recommendation.json` を上記値に更新しました。
- `systems/downside_risk_system_v1.py --predict-date 2025-10-10 --retrain` を実行済みで、`production_data/downside_predictions.parquet` / `risk_predictions.parquet` は最新状態です。

## Codex-B へのお願い
1. **日次パイプラインの連携強化**
   - `daily_trading_automation.py` 内で `systems/downside_risk_system_v1.py` 実行後、`production_data/downside_predictions.parquet` / `risk_predictions.parquet` が確実に残るよう確認をお願いします（必要なら退避やチェック機構を追加）。
   - マルチモデルレポート (`reports/daily_stock_recommendation_multi.py`) 実行前にファイル存在チェックを行い、欠損時は再実行 or 警告ログを出すよう調整してください。

2. **新閾値の展開と検証**
   - 更新済みの `config/multi_model_recommendation.json`（up=0.14, down=0.50, risk=0.35 / min_probability=0.15）を本番パイプラインに反映してください。
   - 反映後、`PYTHONPATH=. python reports/daily_stock_recommendation_multi.py --date <最新営業日>` を複数日実行し、推奨件数・Precision 等のログ（特に `total_candidates` / `passed_all` 件数）を共有いただきたいです。

3. **候補データの継続蓄積**
   - 現状 20 営業日分ですが、日次で更新して 3〜6 ヶ月分を保持できるよう運用面の整備をご検討ください。

## 参考コマンド
```bash
# 候補データ生成（Codex-A 実施済み例）
PYTHONPATH=. python analysis/build_multi_model_candidate_dataset.py \
  --lookback-days 20 --max-candidates 150 --temp-dir tmp/multi_candidate_builder

# 閾値サーチ（重み付き）
PYTHONPATH=. python analysis/multi_model_threshold_optimizer.py \
  --input production_data/multi_model_candidates.parquet \
  --threshold-up-grid 0.12,0.13,0.14,0.15,0.16 \
  --threshold-down-grid 0.46,0.48,0.50 \
  --risk-grid 0.35,0.40,0.45,0.50 \
  --weight-up 1.0 --weight-down 0.5 --weight-risk 0.4 \
  --top-n 5 --target-return 0.01 --transaction-cost 0.002 \
  --metric precision --metric-weights precision:0.5,avg_net_return:0.3,coverage_rate:0.2 \
  --min-valid-count 30 --top-k 20 \
  --export-csv analysis/multi_model_threshold_grid_weighted.csv
```

ご確認・ご対応をよろしくお願いいたします。
