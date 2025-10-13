# フォールバック制御調整結果 (2025-10-15)

## 1. 背景
- 既存設定ではフォールバック比率が平均 67.9%（完全フォールバック日 7）に達しており、Precision 34.6% のまま依存度が高い状態だった。

## 2. 追加グリッドサーチ
- `analysis/multi_model_threshold_grid_aggressive.csv` を生成し、以下の範囲で総当り（9,216 通り）：
  - `threshold_up ∈ {0.18, 0.19, 0.20}`, `threshold_down ∈ {0.50, 0.52}`, `threshold_risk ∈ {0.40, 0.45}`
  - `fallback_min_passed_ratio ∈ [0.45, 0.60]`
  - `fallback_min_composite ∈ {-0.03, -0.02, -0.01, 0.0}`
  - `fallback_min_up_prob ∈ {0.14, 0.15, 0.16, 0.17}`
  - `fallback_risk_margin ∈ {0.02, 0.03, 0.04, 0.05}`
  - `fallback_block_ratio ∈ {0.20, 0.25, 0.30}`

## 3. 推奨パラメータ
| 項目 | 値 |
| --- | --- |
| `threshold_up` | 0.18 |
| `threshold_down` | 0.52 |
| `threshold_risk` | 0.40 |
| `fallback_min_passed_ratio` | 0.45 |
| `fallback_min_composite` | -0.03 |
| `fallback_min_up_prob` | 0.17 |
| `fallback_risk_margin` | 0.05 |
| `fallback_block_ratio` | 0.20 |

- 指標（`analysis/multi_model_candidate_daily_metrics.csv`）
  - Precision **42.9%**
  - Coverage **85%**（17/20 営業日でシグナル）
  - フォールバック比率 **9.1%**（fallback 採用 2 件）
  - 運用適用後（`production_data/multi_model_metrics.csv` 最新 14 営業日）では Precision **50.0%** / フォールバック比率 **35.7%** / 完全フォールバック日 **5** を記録。

## 4. 差分ファイル
- `analysis/multi_model_threshold_grid_aggressive.csv` — 全探索結果
- `analysis/multi_model_threshold_grid_fallback_low.csv` — fallback ≤ 0.2 絞り込み
- `analysis/multi_model_candidate_daily_metrics.csv` — 推奨案の日次明細
- `config/multi_model_recommendation_candidate.json` — 提案設定一式

## 5. 次ステップ
1. Codex-B の承認後、`config/multi_model_recommendation.json` に反映し本運用へ移行。
2. 新設定で `analysis/log_multi_model_metrics.py` を日次回しし、Precision/フォールバック推移を監視。
3. Optuna 実験では本設定を初期トライアルに含め、さらなる改善余地を評価。
