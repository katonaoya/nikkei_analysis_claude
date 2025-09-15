# 株価予測自動売買システム - プロジェクト概要

## プロジェクトの目的
J-Quants APIを使用した完全自動化株価予測システム。1コマンド実行でデータ収集から予測まで全プロセスを自動化。

## テクニカルスタック
- **言語**: Python 3.x
- **主要ライブラリ**:
  - データ処理: pandas, numpy
  - 機械学習: scikit-learn, lightgbm, catboost, xgboost
  - テクニカル分析: ta-lib
  - API接続: requests, httpx
  - 設定管理: pyyaml, python-dotenv
  - ログ: loguru
  - 可視化: matplotlib, seaborn, plotly

## プロジェクト構造
```
project_root/
├── src/                      # メインソースコード
├── scripts/                  # 実行スクリプト
├── config/                   # 設定ファイル
├── data/                     # データディレクトリ
├── tests/                    # テストファイル
├── utils/                    # ユーティリティ
├── results/                  # 実行結果
├── quick_trade_existing.py   # 高速実行（推奨）
├── quick_trade.py           # 完全実行
└── requirements.txt         # 依存関係
```

## 主要エントリーポイント
1. **高速実行（推奨）**: `python quick_trade_existing.py`
2. **完全実行**: `python quick_trade.py`
3. **詳細設定**: `python scripts/run_trading_pipeline.py`

## システムの特徴
- ワンコマンド実行
- データ収集 → 特徴量生成 → 機械学習 → 予測の完全自動化
- 回帰・分類両方のモデルサポート
- 日次実行対応
- 詳細なログ出力