# タスク完了時のチェックリスト

## コード開発完了時の必須確認事項

### 1. コード品質確認
```bash
# Python コードの構文チェック
python -m py_compile your_script.py

# 実行テスト
python your_script.py --help  # ヘルプが正常表示されるか確認
```

### 2. ログ確認
```bash
# エラーログの確認
tail -n 50 trading_pipeline.log
tail -n 50 data_collection.log

# 実行時エラーがないかチェック
grep -i "error\|exception\|failed" *.log
```

### 3. 依存関係確認
- requirements.txtに新しいライブラリを追加した場合は動作確認
- 環境変数（.env）の設定が必要な場合は確認

### 4. データ整合性確認
```bash
# 出力ファイルの存在確認
ls -la data/predictions/
ls -la data/models/
ls -la data/reports/

# データサイズの妥当性確認
du -sh data/processed/*.parquet
```

### 5. 設定ファイル確認
- `config/trading_config.yaml` の設定変更が意図通りか確認
- パフォーマンスに影響する設定値をチェック

### 6. 実行時間測定
```bash
# 実行時間を測定して性能劣化がないか確認
time python quick_trade_existing.py
```

### 7. 予測結果の妥当性確認
- 予測値の範囲が現実的か
- 極端な外れ値がないか
- 信頼度の分布が適切か

## 注意事項
- 本システムは教育・研究目的のため、実運用前は十分な検証が必要
- APIレート制限に注意（batch_size, delayの設定確認）
- 大量データ処理時はメモリ使用量の監視