# Optuna 多目的最適化計画 (Precision / Coverage / Fallback)

## 1. 目的
- Precision、Coverage、Fallback Ratio の三指標を同時に最適化し、目標値 (Precision ≥ 0.45, Coverage ≥ 0.60, Fallback Ratio ≤ 0.40) を達成する設定を探索。

## 2. 探索パラメータ
| パラメータ | 範囲 / 選択肢 |
| --- | --- |
| `threshold_up` | 0.18〜0.22 |
| `threshold_down` | 0.48〜0.52 |
| `threshold_risk` | 0.38〜0.45 |
| `fallback_min_passed_ratio` | 0.40〜0.65 |
| `fallback_min_composite` | -0.05〜0.00 |
| `fallback_min_up_prob` | 0.14〜0.20 |
| `fallback_risk_margin` | 0.02〜0.06 |
| `fallback_block_ratio` | 0.15〜0.35 |

## 3. 目的関数候補
重量付けしたスコアを最大化：

```
score = w_precision * precision + w_coverage * coverage - w_fallback * fallback_ratio
```

決定値: `w_precision=0.5`, `w_coverage=0.3`, `w_fallback=0.2`。制約未達時にはペナルティ 0.2 を減点。

## 4. データセット・検証
- 入力：`production_data/multi_model_candidates.parquet`（最新 120 営業日）
- 評価：ウォークフォワード 4 分割（各 30 営業日）
- 指標出力：`analysis/multi_model_optuna_trials.csv` に trial ごとのパラメータと指標を記録。

## 5. TODO
1. `analysis/multi_model_optuna_search.py` を定期ジョブ化し、最新データで再探索できるようスケジュール設定。
2. ベストトライアル（Precision 0.36 / Coverage 1.00 / Fallback 0.23）を初期値とした追加探索を実施し、Precision ≥0.45 達成を狙う。
3. 探索結果から有望な設定を比較レポート化し、フォールバック制御の次回改訂に反映。
