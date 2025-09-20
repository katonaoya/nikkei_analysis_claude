# Stock Data Collection System

J-Quants APIを使用した株価データ収集システム

## 概要

このシステムは、J-Quants APIから日本株の過去データを取得し、機械学習用のデータセットを構築するためのツールです。

## 主要機能

### 1. データ取得
- ✅ J-Quants API認証（メール/パスワード → リフレッシュトークン → IDトークン）
- ✅ 日次株価データ取得（OHLCV + 調整後価格）
- ✅ 日経225銘柄の一括取得
- ✅ バッチ処理による効率的なAPI利用
- ✅ レート制限対応（API負荷軽減）

### 2. データ管理
- ✅ Parquet形式での高効率データ保存
- ✅ データキャッシュ機能
- ✅ 収集状況のログ記録
- ✅ エラー処理とリトライ機能

### 3. ユーティリティ
- ✅ データ概要レポート生成
- ✅ 日次データ更新スクリプト
- ✅ データ品質チェック

## ディレクトリ構造

```
data/
├── raw/                     # 生データ（Parquet形式）
├── processed/               # 前処理済みデータ
├── cache/                   # キャッシュファイル
├── models/                  # 訓練済みモデル
└── predictions/             # 予測結果

scripts/
├── collect_historical_data.py  # 過去データ一括取得
├── daily_update.py             # 日次データ更新
└── data_overview.py            # データ概要レポート

src/data/
├── jquants_client.py           # J-Quants APIクライアント
├── stock_data_fetcher.py       # 統一データ取得インターフェース
├── data_preprocessor.py        # データ前処理
└── external_data_fetcher.py    # 外部データ取得
```

## 使用方法

### 1. 環境設定

`.env`ファイルにJ-Quantsの認証情報を設定：
```env
JQUANTS_MAIL_ADDRESS=your_email@example.com
JQUANTS_PASSWORD=your_password
```

### 2. 過去データの一括取得

```bash
# 過去2年分のデータを取得（デフォルト）
python scripts/collect_historical_data.py

# 特定期間のデータを取得
python scripts/collect_historical_data.py --start-date 2024-01-01 --end-date 2024-12-31

# 特定銘柄のデータを取得
python scripts/collect_historical_data.py --codes "7203,6758,9984" --start-date 2024-01-01

# バッチサイズと遅延時間を調整
python scripts/collect_historical_data.py --batch-size 5 --delay 2.0
```

### 3. 日次データ更新

```bash
# 最新営業日のデータを更新
python scripts/daily_update.py

# 特定日のデータを更新
python scripts/daily_update.py --date 2024-12-31

# 強制更新（非営業日でも実行）
python scripts/daily_update.py --force
```

### 4. データ概要の確認

```bash
# 詳細レポートの生成
python scripts/data_overview.py --report

# サンプルデータの表示
python scripts/data_overview.py --sample

# 特定ファイルの確認
python scripts/data_overview.py --sample --file-pattern "*2024*" --n-samples 10
```

### 5. プログラム内での使用

```python
from src.data import StockDataFetcher

# データ取得インスタンス作成
fetcher = StockDataFetcher()

# 単一銘柄のデータ取得
data = fetcher.get_stock_prices('7203', '2024-01-01', '2024-12-31')

# 複数銘柄のデータ取得
codes = ['7203', '6758', '9984']
data = fetcher.get_multiple_stocks(codes, '2024-01-01', '2024-12-31')

# 企業情報の取得
company_info = fetcher.get_company_info('7203')
```

## データ仕様

### 取得されるデータ項目

- **基本価格**: 始値(Open)、高値(High)、安値(Low)、終値(Close)
- **出来高**: Volume、回転代金(TurnoverValue)
- **調整後価格**: 株式分割等を考慮した調整済み価格
- **制限値幅**: UpperLimit、LowerLimit
- **調整係数**: AdjustmentFactor

### データ形式

- **ファイル形式**: Parquet（高効率な圧縮・読み込み）
- **日付形式**: YYYY-MM-DD
- **銘柄コード**: 4桁数字（例：7203）
- **エンコーディング**: UTF-8

### データ品質

- ✅ 自動データ型変換
- ✅ 欠損値の適切な処理
- ✅ 異常値の検出・除外
- ✅ データ整合性チェック

## パフォーマンス

### API制限への対応
- バッチサイズ: 5-20銘柄/バッチ（推奨）
- バッチ間遅延: 1-2秒（API負荷軽減）
- リクエスト間遅延: 0.5秒
- エラー時の自動リトライ

### データ収集速度
- **小規模** (10銘柄×30日): ~30秒
- **中規模** (100銘柄×365日): ~10分
- **大規模** (200銘柄×2年): ~30分

### ストレージ効率
- Parquet形式により70-80%のサイズ削減
- 100銘柄×2年分 ≈ 10-20MB

## トラブルシューティング

### よくあるエラー

1. **認証エラー**
   ```
   ValueError: JQuants authentication credentials not provided
   ```
   → `.env`ファイルの認証情報を確認

2. **API制限エラー**
   ```
   HTTP 429: Too Many Requests
   ```
   → バッチサイズを小さく、遅延時間を長く設定

3. **データなしエラー**
   ```
   No data retrieved: code=XXXX
   ```
   → 銘柄コードや日付の妥当性を確認

### ログの確認

データ収集時のログは以下に記録されます：
- `data_collection.log`: 過去データ取得ログ
- `daily_update.log`: 日次更新ログ

## 次のステップ

データ収集完了後は以下の機能を実装予定：

1. **特徴量エンジニアリング**
   - テクニカル指標の計算
   - 市場環境特徴量の追加
   - ラベル作成

2. **機械学習パイプライン**
   - モデル訓練
   - 性能評価
   - 予測システム

3. **運用システム**
   - 自動予測
   - 結果の可視化
   - アラート機能

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 免責事項

このツールは教育・研究目的で作成されています。実際の投資判断に使用する場合は、十分な検証とリスク管理を行ってください。投資判断は自己責任で行ってください。