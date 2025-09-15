# 🚀 統合トレーディングシステム

J-Quants APIを使用した完全自動化株価予測システム

## 📋 システム概要

このシステムは**1コマンド実行**で以下の処理を自動実行します：

1. **📊 データ収集** - J-Quants APIから最新株価データ取得
2. **🔧 特徴量生成** - テクニカル指標・市場環境特徴量の計算
3. **🤖 機械学習** - 回帰・分類モデルの訓練
4. **🔮 予測実行** - 次日の株価予測とランキング生成

## ⚡ クイック実行

### 📝 事前準備
```bash
# 1. 環境変数設定（.envファイル）
JQUANTS_MAIL_ADDRESS=your_email@example.com
JQUANTS_PASSWORD=your_password

# 2. 必要パッケージのインストール
pip install pandas numpy scikit-learn xgboost lightgbm
```

### 🚀 実行方法

#### **方法1: 完全自動実行（推奨）**
```bash
# データ取得から予測まで全て実行
python quick_trade_existing.py
```

#### **方法2: 新データ取得込み**
```bash  
# 最新データを取得してから実行（時間がかかります）
python quick_trade.py
```

#### **方法3: 詳細オプション指定**
```bash
# カスタム設定で実行
python scripts/run_trading_pipeline.py --start-date 2024-01-01 --end-date 2024-12-31
```

## 📊 実行結果

### **出力ファイル**
```
data/
├── predictions/        # 予測結果（CSV形式）
├── models/            # 訓練済みモデル
├── reports/           # 実行レポート
└── processed/         # 生成された特徴量
```

### **予測結果例**
```
🏆 上位予測結果:
      Date  Code  Close  prediction  prob_up  confidence
2025-08-28 63610 3135.0           1     79.9%      79.9%
2025-08-28 13010 4730.0           1     76.8%      76.8%
2025-08-28 78320 5127.0           1     75.5%      75.5%
```

## 🛠️ コマンドラインオプション

### `run_trading_pipeline.py`の主要オプション

```bash
# データ収集をスキップ（既存データ使用）
python scripts/run_trading_pipeline.py --skip-data-collection

# 特定期間のデータ収集
python scripts/run_trading_pipeline.py --start-date 2024-01-01 --end-date 2024-12-31

# 回帰モデルのみ訓練
python scripts/run_trading_pipeline.py --no-classification

# 分類モデルのみ訓練  
python scripts/run_trading_pipeline.py --no-regression

# 詳細出力を抑制
python scripts/run_trading_pipeline.py --quiet
```

## 📈 システム構成

### **1. データ収集モジュール**
- `scripts/collect_historical_data.py` - 過去データ一括取得
- `scripts/daily_update.py` - 日次データ更新

### **2. 特徴量エンジニアリング**
- `scripts/simple_feature_generation.py` - 特徴量生成
- **生成される特徴量:**
  - 移動平均（5日、10日、20日）
  - RSI（14日）
  - ボラティリティ（20日）
  - 市場幅指標
  - 相対パフォーマンス

### **3. 機械学習モジュール**
- `scripts/simple_model_training.py` - モデル訓練
- **使用アルゴリズム:**
  - Random Forest（回帰・分類）
  - Linear Regression / Logistic Regression

### **4. 予測システム**
- `scripts/simple_prediction.py` - 予測実行
- **予測対象:**
  - 次日リターン率（回帰）
  - 上昇/下降方向（分類）

## ⚙️ 設定ファイル

`config/trading_config.yaml`で各種パラメータを調整可能：

```yaml
# データ収集設定
data_collection:
  default_days: 30
  batch_size: 5
  delay_between_batches: 1.0

# 特徴量設定
features:
  ma_periods: [5, 10, 20, 60]
  rsi_period: 14
  volatility_window: 20

# モデル設定
models:
  regression:
    algorithms: ["linear_regression", "random_forest"]
    test_size: 0.2
```

## 📋 実行例とパフォーマンス

### **典型的な実行時間**
- **高速版**（既存データ使用）: 約10秒
- **完全版**（データ取得込み）: 約5分

### **モデル性能例**
```
📊 回帰モデル (Next_Day_Return):
   Random Forest - MSE: 0.001388, MAE: 0.017199
   Linear Regression - MSE: 0.000994, MAE: 0.012099

📊 分類モデル (Up/Down Direction):
   Random Forest - Accuracy: 55.3%
   Logistic Regression - Accuracy: 45.9%
```

## 🔧 トラブルシューティング

### **よくあるエラー**

1. **認証エラー**
   ```
   ValueError: JQuants authentication credentials not provided
   ```
   → `.env`ファイルのメール・パスワードを確認

2. **データファイルが見つからない**
   ```
   FileNotFoundError: No data files found
   ```
   → まず`python scripts/collect_historical_data.py`でデータ収集

3. **モデルファイルが見つからない**
   ```
   FileNotFoundError: Model file not found
   ```
   → `--skip-data-collection`を外してフル実行

### **ログ確認**
```bash
# 実行ログの確認
tail -f trading_pipeline.log

# 詳細なデバッグ情報
python scripts/run_trading_pipeline.py --help
```

## 🚀 次のステップ

1. **毎日の運用**
   ```bash
   # crontab設定例（毎日18時実行）
   0 18 * * * cd /path/to/project && python quick_trade_existing.py
   ```

2. **予測精度向上**
   - より多くの特徴量追加
   - ハイパーパラメータ最適化
   - アンサンブル手法

3. **リスク管理**
   - ポジションサイジング
   - 損切り・利確ルール
   - ポートフォリオ分散

## 📞 サポート

- **ログファイル**: `trading_pipeline.log`
- **設定ファイル**: `config/trading_config.yaml`
- **データディレクトリ**: `data/`

---

**⚠️ 免責事項**: このツールは教育・研究目的です。実際の投資判断は自己責任で行ってください。