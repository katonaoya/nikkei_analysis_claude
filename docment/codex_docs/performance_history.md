# 精度検証ログ

実装が一定水準に達した段階で精度検証を行い、その時点での最高指標・条件・実装状況を記録します。最新の記録を上に追記してください（上書き禁止）。

| 日付 (YYYY-MM-DD) | 精度指標 | データ条件 / 期間 | 実装バージョン / Gitコミット | 検証コマンド / 設定 | 備考・課題 |
| --- | --- | --- | --- | --- | --- |
| 2025-11-07 | **Rolling Precision (last 6 windows) 0.8259** / mean accuracy 0.7834 / precision range 0.7570〜0.8646 | 2025-04-23〜2025-10-28 の21営業日ステップWFOの最新6区間（550,199行から該当期間のみ抽出）。株価: `data/processed/nikkei225_complete_225stocks_20251106_214402.parquet` + 外部: `data/processed/enhanced_integrated_data.parquet` | `801c7d8` の `models/enhanced_v3/enhanced_results_v3_20251107_215318.joblib` を再計測 | `python - <<'PY' ...` で joblib `wfo_results` の末尾6区間を集計（精度・適合率平均とレンジを算出） | 直近データのみでの信頼指標。min 0.7570 の区間再分析とKabutan/高頻度データ統合が改善テーマ。 |
| 2025-11-07 | Final accuracy 0.8117 / WFO mean accuracy 0.8001 / WFO mean precision 0.8093 (max 0.9184, min 0.7237) | 株価: `data/processed/nikkei225_complete_225stocks_20251106_214402.parquet` + 外部: `data/processed/enhanced_integrated_data.parquet`、約10年=550,199行、21営業日ステップのウォークフォワード92区間 | `801c7d8` (`systems/enhanced_precision_system_v3.py` 現行) | `python systems/enhanced_precision_system_v3.py`（特徴量34→SelectKBestで25, LightGBM, 外部指標9列統合） | Top3精度要件は満たす水準だが Precision低下区間 (min 0.7237) への対策とKabutan/高頻度データ連携が今後の課題 |

## 記載ガイド
- **精度指標**: Precision, Recall, Top-N precision など主要メトリクスを明記。複数ある場合は `Precision@Top3 0.68` のように記述。
- **データ条件 / 期間**: 使用した銘柄範囲、期間、クロスバリデーション手法（例: 30営業日ローリングCV）。
- **実装バージョン / Gitコミット**: 適用したブランチ名やコミットハッシュ、関連する設定ファイル/モデルバージョン。
- **検証コマンド / 設定**: 実行したスクリプト、CLIオプション、重要な設定値を記載。
- **備考・課題**: 改善余地や次のアクション、異常値、制約など。

> 最高精度を更新した場合は、必ず新しい行を上に追加し、以前の結果を残したまま履歴を積み上げてください。
