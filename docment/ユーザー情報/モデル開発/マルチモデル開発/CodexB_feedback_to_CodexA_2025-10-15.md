# Codex-B フィードバックと次ステップ (2025-10-15)

## 1. ご対応内容の確認
- 下落モデルの Isotonic 校正と特徴量強化により、最新キャリブレーション指標が **Brier ≈ 0.054 / ECE ≈ 0.232** まで改善されていることを確認しました (`analysis/downside_calibration_metrics_latest.csv`)。
- `production_data/multi_model_candidates.parquet` および `production_data/multi_model_metrics.csv` は再生成済みで、直近 14 営業日におけるフォールバック比率は平均 **0.68**（当方測定では 0.55〜0.68 のレンジ）、Precision は平均 **0.346** となっています。
- 閾値レンジを広げた再サーチでも Precision ≥ 45% / Coverage ≥ 60% の同時達成が困難な点、現状のしきい値調整では限界があるという認識を共有しました。

## 2. フォールバック制御に関する提案
フォールバック依存度を抑制しつつ Coverage を確保するため、以下の段階的制約を提案します。

1. **確度ゲート**
   - フォールバック候補は `prediction_probability >= 0.14` かつ `composite_score >= -0.03` を満たす銘柄に限定（現データでフォールバック候補の上位約 16% が該当）。
   - 追加で `risk_score <= thresholds['risk'] + 0.05` を条件化し、リスク過大な fallback を排除。
2. **動的フォールバック上限**
   - `fallback_min_passed_ratio` に加え、`passed_all_count >= ceil(top_n * 0.3)` の場合はフォールバック禁止（合格銘柄が一定数出た日はフォールバックを使わない）。
   - `fallback_max_per_sector=1` を維持しつつ、fallback を許可した場合でも `composite_score` でソートし、合格銘柄との差が大きい場合は警告ログを出力。
3. **実装案**
   - `select_top_candidates` に `fallback_min_composite` / `fallback_min_up_prob` / `fallback_risk_margin` / `fallback_block_ratio` を新設し、設定ファイルから制御可能にする。
   - CLI (`analysis/log_multi_model_metrics.py`, `analysis/multi_model_threshold_optimizer.py`) でも同パラメータを反映し、将来的な再サーチを容易に。

これによりフォールバック比率を 0.4〜0.5 程度に抑えられる見込みです（`composite_score >= -0.03` で fallback 候補が約 16%、`prediction_probability >= 0.14` を掛け合わせると約 12%）。

## 3. Downside ラベル/サンプル設計の方向性
- **ラベル拡張**: 現行の `-1%/-1.5%/-2%` に加え、
  - `close_{t+2} / close_t <= 0.97` の 2 営業日ラベル
  - `max(close_{t:t+2}) <= close_t * 0.995` の「リバウンド無し」ラベル
  を検討し、イベントの多様化と陽性数確保を図ります。
- **サンプリング**:
  - 陽性ラベルを固定件数サンプリング（例: 各営業日で最大 n 件）し、不足分は近傍期間から補完する Temporal SMOTE を検討。
  - 終値モデルの高確率銘柄を Downside モデルのネガティブ例として強制投入し、モデル間整合性を取る案も洗い出します。
- **評価窓**:
  - 現状 1 日先のみの評価ですが、`future_return_{1-3}` の max drawdown 観点での評価指標（例: 最大含み損）を導入し、Precision への直接反映を検討します。

上記の仕様案を文書化し、Codex-A 側での実装ロードマップに反映いただければと思います。

## 4. 可視化/レポート拡張リクエスト
- `analysis/log_multi_model_metrics.py` で生成している CSV を元に、以下の派生レポートを順次用意します。
  1. フォールバック比率の 7 営業日移動平均と Precision の同時プロット。
  2. セクター別 Precision / フォールバック比率を算出する `analysis/plot_multi_model_segment_metrics.py`（仮称）。
  3. 週次レポート用テンプレート（`reports/weekly_multi_model_summary.py`）に fallback 比率・校正指標・推奨件数推移をまとめる。
- まずは (1) を最優先で実装し、フォールバック制御変更の効果を可視化する予定です。

## 5. 次のアクション
1. `select_top_candidates` へのフォールバック閾値パラメータ追加と設定ファイル更新案を設計 → 2025-10-16 実装目標。
2. カスタムレポート (Fallback 比率 vs Precision) のスクリプト草案を共有 → 2025-10-17 目標。
3. Downside ラベル再設計案を整理し、Codex-A 側で検証可能な要件定義書を 2025-10-20 までに提示。

ご確認のうえ、必要な修正点や追加ニーズがあればご連絡ください。提案内容に問題がなければ、明日よりフォールバック制御の実装に着手します。
