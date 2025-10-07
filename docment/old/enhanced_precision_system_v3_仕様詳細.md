# Enhanced Precision System V3 仕様詳細

## 1. システム概要
- **目的**: 日経225構成銘柄を対象に、翌営業日の前日終値比 +1%以上の高値更新（ブレイクアウト）確率を算出。
- **モデル**: LightGBM による二値分類 (`LGBMClassifier`)。`SelectKBest` と `RobustScaler` を組み合わせた前処理パイプラインで 78.5% 精度モードを再現。
- **実行ポイント**: `systems/enhanced_precision_system_v3.py` の `EnhancedPrecisionSystemV3` クラスが学習・検証・保存を一括実行。
- **ターゲット閾値**: 翌日高値 / 前日終値 > 1.01 を陽性ラベル (`Target=1`) と定義。

## 2. データソースと入出力
### 2.1 株価データ
- **取得元**: J-Quants API (`data_management/nikkei225_complete_parallel_fetcher.py`)。
- **対象範囲**: 日経225 全 225 銘柄、概ね過去 10 年分。
- **保存形式**: `data/processed/nikkei225_complete_YYYYMMDD_HHMMSS.parquet`。
- **自動選択**: モデル実行時にタイムスタンプ最新の parquet を探索し利用 (`_find_latest_stock_file`)。

### 2.2 外部・補助データ
- **統合候補**: `data/processed/enhanced_integrated_data.parquet`、`data/external_extended/external_integrated_*.parquet` 等を `_find_latest_external_file` で探索。
- **組み込み条件**: 行数が 10,000 件未満の場合のみマージ（USDJPY, VIX, Nikkei225, S&P500 など主要指標列を抽出）。閾値超過の場合は株価データのみで継続。
- **補助スクリプト**: `data_management/enhanced_data_integration.py` がファンダメンタル（J-Quants）・マーケットデータ（Yahoo Finance）・相互作用特徴量を生成し `enhanced_integrated_data.parquet` を書き出す。

### 2.3 出力物
- **モデル**: `models/enhanced_v3/enhanced_model_v3_{accuracy:.4f}acc_{YYYYMMDD_HHMMSS}.joblib`
  - 格納内容: 学習済み LightGBM モデル、`RobustScaler`、`SelectKBest`、使用特徴量リスト、主要評価指標、学習/検証サンプル数。
- **検証結果**: `models/enhanced_v3/enhanced_results_v3_{YYYYMMDD_HHMMSS}.joblib`
  - ウォークフォワード統計、データサイズ、特徴量数、外部データ統合有無を保持。

## 3. パイプライン構成
### 3.1 日次自動実行フロー
`daily_trading_automation.py`
1. **STEP1**: J-Quants 全銘柄データ取得 (`nikkei225_complete_parallel_fetcher.py`)。
2. **STEP2**: 外部データ統合 (`enhanced_data_integration.py`)。
3. **STEP3**: Enhanced V3 モデル学習・検証 (`systems/enhanced_precision_system_v3.py`)。
4. **STEP4**: 推奨銘柄レポート生成 (`reports/daily_stock_recommendation_v3.py`)。

### 3.2 推奨レポート生成
- 最新の joblib をロードし、対象営業日の特徴量を再計算。
- モデルの `feature_cols` と保存済み前処理を適用。
- 予測確率 60%以上の銘柄を Markdown (`production_reports/YYYY-MM/DD.md`) に出力。

## 4. 特徴量エンジニアリング
### 4.1 共通処理
- 銘柄単位で日付昇順に並び替え、50 レコード未満の銘柄を除外。
- 欠損値・無限大を NaN へ変換し、`fillna(method='ffill').fillna(0)` で補完。

### 4.2 価格ベース特徴量
| 分類 | 詳細 |
| --- | --- |
| リターン系 | `Returns` (終値パーセント変化)、`Volatility_20` (20 日標準偏差) |
| 価格対比 | `High_Low_Ratio`、`MA_5` / `MA_20`、`MA_5_ratio` / `MA_20_ratio` |
| テクニカル | `RSI_14`（標準的 RS 計算）、`MACD`（12,26,9 EMA） |
| 出来高 | `Volume_MA_20`、`Volume_ratio` (= Volume / Volume_MA_20) |

### 4.3 外部指標 (条件付き)
- 列名に `usdjpy` または `vix` を含む場合、十分な観測値があればパーセント変化列を追加。
- `enhanced_data_integration.py` を経由する場合、ファンダメンタル指標（PER, PBR, ROE 等）や市場指数派生量（Nikkei/TOPIX, VIX regime など）も parquet 内に格納。

## 5. モデリング & バリデーション
### 5.1 最終学習
- サンプリング: データが 100,000 行超の場合は 100,000 行へ縮小。
- 特徴量選択: 欠損が少ない上位 25 列へ制限。`SelectKBest(f_classif)` で最大 30 特徴量を選択。
- スケーリング: `RobustScaler` で外れ値影響を抑制。
- 学習: LightGBM (`subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=reg_lambda=0.1`)。
- 評価: Accuracy / Precision / Recall / F1 をログ出力。

### 5.2 ウォークフォワード最適化
- 初期学習期間: ユニーク日付の 1/3 を上限に設定。
- ステップ幅: 42 営業日（約 2 ヶ月）。
- 期間ごとに同一パイプラインを適用し、指標を集約。

## 6. 運用・依存関係
- **Python**: 3.9 推奨。LightGBM, scikit-learn, pandas, numpy, joblib などが必須。
- **環境変数**: `.env` に J-Quants 認証情報 (`JQUANTS_MAIL_ADDRESS`, `JQUANTS_PASSWORD`) を設定。
- **ログ**: `daily_automation.log` に日次処理ログ、モデル実行時は INFO レベルで主要指標が記録される。
- **カレンダー**: `utils/market_calendar.JapanMarketCalendar` で各ステップの対象日を決定。

## 7. 既存ドキュメントとのギャップ
- `ベストプラクティス_0910.md` 記載の拡張仕様（外部指標 10 種、RSI/Volatility の複数窓、21 日ステップの WFO など）は現行コードではメモリ制約を考慮して一部無効化。
- フル機能復旧には `load_and_integrate_data` の外部データ制限解除と、レポート側の特徴量再現ロジック拡張が必要。

## 8. 既知の課題と改善余地
1. **Parquet 読み込み失敗**: `enhanced_integrated_data.parquet` を pandas で直接読むと sandbox 環境で `Floating point exception` が発生。pyarrow など別手段でのスキーマ確認が必要。
2. **外部データ利用率**: 10,000 行以上のファイルをスキップするため、拡張統合スクリプトで生成した全指標が学習に利用されていないケースがある。
3. **クラス不均衡**: ブレイクアウト条件の陽性率は約数 % で学習毎に変動。必要に応じてクラス重みやしきい値調整、サンプリング検討。
4. **モデル資産管理**: `models/enhanced_v3` 配下に精度別ファイルが蓄積するため、定期的なクリーンアップポリシーが望ましい。

## 9. 参照ファイル一覧
| 区分 | パス |
| --- | --- |
| モデル本体 | `systems/enhanced_precision_system_v3.py` |
| 日次バッチ | `daily_trading_automation.py` |
| データ取得 | `data_management/nikkei225_complete_parallel_fetcher.py` |
| 外部統合 | `data_management/enhanced_data_integration.py` |
| レポート | `reports/daily_stock_recommendation_v3.py` |
| 仕様ドキュメント(旧) | `docment/ユーザー情報/ベストプラクティス_0910.md` |

---
最終更新: 2025-09-23 20:33
