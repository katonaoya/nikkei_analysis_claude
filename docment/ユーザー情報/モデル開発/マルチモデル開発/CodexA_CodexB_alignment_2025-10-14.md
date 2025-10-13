# Codex-A ↔ Codex-B Alignment メモ (2025-10-14)

## 1. 現状サマリ
- 120 営業日（2025-04-17〜2025-10-10）の候補データ (`production_data/multi_model_candidates.parquet`) を構築済み。
- 閾値サーチのベストは `threshold_up=0.18`, `threshold_down=0.52`, `threshold_risk=0.45`、重み `1.0 / 0.5 / 0.4`。
  - Precision 41.46%、平均リターン +0.59%、Coverage 55%（フォールバック無し）。
  - `analysis/multi_model_threshold_grid_fallback_refined.csv` / `analysis/multi_model_fallback_summary.csv` に詳細あり。
- フォールバック (max_fallback=1, min_passed_ratio=0.40〜0.45) で Coverage 100% は確保できるが Precision は 25〜31% 前後に低下。
- 日次メトリクスログ (`analysis/log_multi_model_metrics.py` → `production_data/multi_model_metrics.csv`) と下落モデルキャリブレーション (`analysis/downside_calibration_report.py`) の運用準備完了。
- レポート (`production_reports/2025-10/multi_model/2025-10-10_multi.md`) には直近14営業日の Precision / AvgReturn / Coverage を表示。

## 2. 協議事項（Codex-A → Codex-B）
1. **フォールバック戦略**
   - Precision ≥45% かつ Coverage ≥60% の両立が未達。`threshold_up` 引き上げ (0.19〜0.22) や `fallback_min_passed_ratio` を 0.45〜0.55 に拡張する案、セクター別フォールバック制限案など、優先順位を確認したいです。
   - デフォルトで `fallback.max_per_sector=1` を設定し、フォールバック採用数をセクター単位で抑制できるようにしています。必要に応じて値をご相談ください。

2. **モニタリングと運用**
   - `production_data/multi_model_metrics.csv` は最新月のみ保持し、古い月を `production_data/multi_model_metrics_archive/` に退避する仕組みを導入済み。バックアップ方法（例: 月次アーカイブの保存先）をご相談したいです。
   - `analysis/downside_calibration_report.py` は日次パイプラインで自動出力するようにしました。Brier 0.20 などの閾値をどう扱うか、再学習トリガーの定義をご相談したいです。

3. **レポート指標拡張**
   - 直近14営業日以外に掲載したい指標（例: フォールバック採用率、セクター別平均など）があればご連絡ください。`log_multi_model_metrics.py` の出力列を拡張します。

4. **今後のサイクル**
   - Codex-A 側で追加特徴量（業種モメンタム、指数差分、ギャップ方向ラベル等）を投入した再学習を予定しています。精度改善後に再度閾値サーチ・フォールバック設計を実施する段取りを確認したいです。

## 3. 推奨アクション
1. フォールバック案（例: `threshold_up=0.20`, `min_passed_ratio=0.5`; `threshold_up=0.21`, `min_passed_ratio=0.55` 等）について、Codex-B の優先順位を伺い、次回再学習後に検証する。
2. `analysis/log_multi_model_metrics.py` を日次バッチへ組み込み、月次スナップショット（`metrics_YYYYMM.csv` など）を作成する運用を整備。
3. 下落モデルの Brier / ECE を週次で記録し、基準値を超えた場合のリトレーニングフローを定義。

## 4. 参考ファイル
- `analysis/multi_model_threshold_grid_fallback_refined.csv` — 閾値×フォールバックの詳細結果。
- `analysis/multi_model_fallback_summary.csv` — シナリオ別 Precision / Coverage の要約。
- `analysis/downside_calibration_metrics_latest.csv` — 最新 Brier / ECE。
- `analysis/log_multi_model_metrics.py`, `production_data/multi_model_metrics.csv` — 日次メトリクスログ。
- `production_reports/2025-10/multi_model/2025-10-10_multi.md` — 新フォーマットレポート例。
- `CodexB_hand_over_2025-10-14.md`, `CodexA_CodexB_alignment_2025-10-14.md` — ハンドオーバー／協議メモ。

---
ご確認のうえ、フォールバック戦略・ログ運用・キャリブレーション監視についてフィードバックをお願いします。
