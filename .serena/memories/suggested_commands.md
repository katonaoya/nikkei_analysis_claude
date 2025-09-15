# 推奨コマンド一覧

## 基本実行コマンド

### 日常運用
```bash
# 高速実行（推奨） - 既存データを使用
python quick_trade_existing.py

# 完全実行 - 最新データ取得込み
python quick_trade.py
```

### 詳細設定での実行
```bash
# カスタム期間でのデータ収集と実行
python scripts/run_trading_pipeline.py --start-date 2024-01-01 --end-date 2024-12-31

# データ収集をスキップ
python scripts/run_trading_pipeline.py --skip-data-collection

# 回帰のみ実行
python scripts/run_trading_pipeline.py --no-classification

# 分類のみ実行
python scripts/run_trading_pipeline.py --no-regression

# 静かに実行（詳細出力抑制）
python scripts/run_trading_pipeline.py --quiet
```

## 個別実行コマンド

### データ収集
```bash
# 過去データ一括取得
python scripts/collect_historical_data.py

# 日次データ更新
python scripts/daily_update.py
```

### 特徴量生成
```bash
python scripts/simple_feature_generation.py
```

### モデル訓練
```bash
# 回帰モデル
python scripts/simple_model_training.py --target Next_Day_Return --model-type regression

# 分類モデル  
python scripts/simple_model_training.py --target Binary_Direction --model-type classification
```

### 予測実行
```bash
python scripts/simple_prediction.py --model-file [model_file.joblib]
```

## 開発・デバッグ用

### 依存関係
```bash
pip install -r requirements.txt
```

### ログ確認
```bash
tail -f trading_pipeline.log
tail -f data_collection.log
```

### 自動実行設定（cron例）
```bash
# 毎日18時に実行
0 18 * * * cd /path/to/project && python quick_trade_existing.py
```

## システム固有（macOS Darwin）
```bash
# ファイル検索
find . -name "*.py" | head -20

# プロセス確認  
ps aux | grep python

# ディスク使用量
du -sh data/
```