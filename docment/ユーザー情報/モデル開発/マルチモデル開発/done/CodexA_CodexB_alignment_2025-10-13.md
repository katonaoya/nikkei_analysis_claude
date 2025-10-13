# Codex-A ↔ Codex-B Alignment メモ (2025-10-13 00:36)

## 1. 現状サマリ
- **候補データ**: `analysis/build_multi_model_candidate_dataset.py --lookback-days 60 --max-candidates 150`
  - 期間: 2025-07-15 〜 2025-10-10（60 営業日）
  - 件数: 9,000 行 (`production_data/multi_model_candidates.parquet`)
- **下落モデル**: ロジスティック回帰 + 標準化。特徴量に 10/20 日モメンタム、相対リターン、出来高比率などを追加。
- **閾値サーチ（重み付き）**
  - コマンド: `PYTHONPATH=. python analysis/multi_model_threshold_optimizer.py --input production_data/multi_model_candidates.parquet --threshold-up-grid 0.12,0.13,0.14,0.15,0.16,0.18 --threshold-down-grid 0.46,0.48,0.50 --risk-grid 0.30,0.35,0.40 --weight-up 1.0 --weight-down 0.5 --weight-risk 0.4 --top-n 5 --target-return 0.01 --transaction-cost 0.002 --metric precision --metric-weights precision:0.6,avg_net_return:0.3,coverage_rate:0.1 --min-valid-count 40 --top-k 10 --export-csv analysis/multi_model_threshold_grid_weighted_60d.csv`
  - ベスト: `threshold_up=0.18`, `threshold_down=0.50`, `threshold_risk=0.40`
    - Precision 41.46%
    - 平均リターン +0.59%
    - Coverage 55%（fallback 無し）
    - 選抜 41 件 / 60 営業日中 33 日
  - fallback(max=1, min_passed_all=1) を併用すると Precision 35.82% / 平均リターン +0.47% / Coverage 100%（選抜 68 件）。
  - fallback(max=1, min_passed_all=3) では Precision 33.70% / 平均リターン +0.44% / Coverage 100%（選抜 93 件）。
  - 現状はフォールバック無効（`max_fallback=0`）で運用し、Precision 41.46% / Coverage 55% を維持。
- **サマリツール**: `analysis/multi_model_precision_report.py` を追加し、Precision / AvgReturn / Coverage を CLI ですぐに確認可能。
- **日次パイプライン**: `daily_trading_automation.py` に候補データ更新ステップを追加し、`analysis/build_multi_model_candidate_dataset.py --lookback-days 20` を日次で実行するよう改修済み。

## 2. 協議したいポイント
1. **Coverage 改善**
   - Precision 41.5% は達成したものの、Coverage が 55% に留まっています。
   - 閾値 (`up`, `down`, `risk`) の微調整や fallback ルールの導入を検討して、Coverage を 65〜70% まで押し上げたい。
   - 120 営業日データでは `threshold_up=0.18 / threshold_down=0.52 / threshold_risk=0.45` (フォールバック無効) で Precision 40.3% / Coverage 47.5%。フォールバック (max=1, min_passed=2, ratio=0.4) を併用した場合 Precision 25.1% / Coverage 100%。新しいフォールバック比率 (`fallback.min_passed_ratio`) の調整を含めて再度トレードオフを検討したいです。
2. **日次運用フロー**
    - `daily_trading_automation.py` で候補データ再生成が走るようにしましたが、Codex-B でも `production_data/downside_predictions.parquet` / `risk_predictions.parquet` が確実に残るよう、ファイル退避や存在チェックをご検討いただきたいです。
    - 運用上の log 出力（`production_reports/..._multi.md`）に Precision/AvgReturn/手数などを週次で集計できるような仕組みの案をすり合わせたいです。
    - 新しく `analysis/log_multi_model_metrics.py` を追加し、`production_data/multi_model_metrics.csv` へ Precision/AvgReturn/Coverage を日次で自動追記するようにしました。レポートにも直近14営業日の推移を表示するため、ログファイルの保全・バックアップについて議論したいです。
3. **長期検証**
   - 3〜6 ヶ月分の候補データ翻案を目指しています。`--append-output` を追加し、120 営業日ぶん（18,000 行）の再生成が可能になりました。日次で `multi_model_candidates.parquet` を追記する運用フローについて意見交換したいです。
4. **今後の改善ロードマップ**
   - Precision を 60% 以上に引き上げるため、追加特徴量（業種モメンタム、指数差分など）やラベル改善（`down_target_return` の複数パターン）を Codex-A が継続的に検証予定です。Codex-B 側で必要な差分やインターフェース調整があれば提示してください。

## 3. 提案するミーティングアジェンダ
1. 最新閾値 (0.18 / 0.50 / 0.40) のレポート結果共有と、日次パイプラインへの反映状況の確認。
2. Coverage 向上のための施策検討（fallback 追加・閾値緩和のシミュレーション結果共有）。
3. 候補データの蓄積方法（追記 vs 再生成）とモニタリング指標の運用方法。
4. 今後の開発タスク分担（Precision 向上、ウォークフォワード自動化など）の整理。

## 4. 参考資料
- `analysis/multi_model_threshold_grid_weighted_60d.csv`
- `analysis/multi_model_precision_report.py` の出力（Precision 41.46%、AvgReturn +0.59%、Coverage 55%）
- `production_reports/2025-10/multi_model/2025-10-10_multi.md`（新閾値でのレポート例）

---
ご確認のうえ、フィードバックや追加議題があればお知らせください。
