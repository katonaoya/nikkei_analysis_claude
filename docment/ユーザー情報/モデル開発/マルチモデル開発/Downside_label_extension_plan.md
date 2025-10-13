# Downside ラベル拡張・サンプリング設計案 (Codex-A Draft)

## 1. 目的
- Downside モデルの陽性サンプル不足を補い、Brier/ECE を下げつつ Precision を底上げする。
- フォールバック制御に必要なリスク判定の信頼度を高め、fallback 依存度を更に下げる。

## 2. 提案ラベル
| ラベル名 | 定義 | 期待陽性率 | ノート |
| --- | --- | --- | --- |
| `down_target_1pct` (既存) | `Close_{t+1} / Close_t <= 0.99` | ~18% | 基準。検証継続。 |
| `down_target_1pct_2d` (新規) | `Close_{t+2} / Close_t <= 0.99` | ~27% | 2 営業日 horizon を追加し、遅延下落を捕捉。 |
| `drawdown_3pct_3d` (新規) | `max(close_{t:t+3}) <= Close_t * 0.97` | ~15% | 3 日以内に 3% 以上下落するケース。 |
| `no_rebound_2d` (新規) | `max(close_{t:t+2}) <= Close_t * 1.002` | ~22% | 反発が弱いケースを除外し、連続陰線を捉える。 |

## 3. サンプリング / 学習スキーム
1. **Temporal SMOTE**: 最新 6 ヶ月を対象に陽性ラベルを最大 2 倍まで増幅。近傍 3 営業日を距離定義。
2. **Hard Negative Mining**: 終値モデル (`P_up >= 0.55`) で高確率だが下落した銘柄を Downside の負例として強制投入。
3. **交差検証**: 60 営業日ローリングで 5-fold 時系列 CV。Brier・ECE・Precision@TopK を記録。
4. **評価基準**: Brier ≤ 0.20、ECE ≤ 0.10 を再学習完了条件とし、最新 3 ヶ月のホールドアウトで確認。

## 4. 評価指標
- Brier / ECE（各ラベル）
- Precision / Recall（上位 20% 確率しきい値）
- Calibration slope / intercept
- Multi-model 連携指標：新 Downside を `fallback_block_ratio` 判定に使った際の fallback 採用数・Precision 変化。

## 5. 次工程
1. ラベル生成スクリプトの拡張 (`systems/downside_risk_system_v1.py`) とテスト (`tests/systems/test_downside_risk_system.py`) を準備。
2. 新特徴量（業種差分、VWAP 乖離など）を採用しつつ、再学習ジョブを実行。
3. 学習結果と評価メトリクスを `analysis/downside_label_extension_report.md` として共有。

---
フィードバックを頂き次第、実装詳細とスケジュールを確定させます。
