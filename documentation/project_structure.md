## プロジェクト構成ガイド（フォルダ構造と役割）

このドキュメントは、リポジトリ全体の「どこに何があるか」を初心者向けにわかりやすくまとめたものです。まずは全体像（ツリー）を見てから、各ディレクトリの役割を簡潔に説明します。

### 主要ディレクトリのツリー
（サイズが大きいので要点のみを掲載。省略記号 ... は大量ファイル/下層を省略したことを示します）

```
.
├─ daily_trading_automation.py           # 毎日の自動処理の入口（エントリポイント）
├─ systems/                              # 学習・推論の中心ロジック（モデルごとのパイプライン）
├─ src/                                  # 共通ユーティリティ/特徴量/前処理などの再利用コード
├─ utils/                                # 小規模な補助関数群（ログ/時間/I/O など）
├─ data_management/                      # データ収集・前処理・保存のスクリプト群
├─ analysis/                             # 解析・可視化・ハイパラ探索など（実験用スクリプト）
├─ reports/                              # 分析が出力する CSV/集計/スクリプト
├─ production_reports/                   # 本番用に出力される日次レポート（Markdown）
├─ models/                               # 学習済みモデルや関連成果物（.joblib 等）
├─ data/                                 # すべてのデータ置き場（raw/processed/predictions 等）
├─ production_data/                      # 本番利用に必要な軽量データ（ライブ使用向け）
├─ tests/                                # Pytest によるテストコード
├─ documentation/                        # ドキュメント（本ファイル含む）
├─ docment/                              # 運用手順・古い資料・ユーザー情報（過去資産）
├─ config/                               # モデル/レポート/閾値などの設定ファイル
├─ production_config.yaml                # 本番設定（バックアップ含む）
├─ logs/                                 # 各処理のログ
├─ results/                              # 実験結果の出力（CSV/JSON/TXTなど）
├─ optimization/                         # 最適化スクリプト（閾値/期間 等）
├─ profit_loss_optimization_results/     # 収益最適化の結果（CSV/JSON/PNG）
├─ signals/                              # シグナル生成・組み合わせ関連
├─ production/                           # 本番運用用スクリプト（配信/整形など）
├─ archive/                              # マイルストーン的な成果記録
├─ data_exports/                         # 外部共有・一時エクスポート
├─ investigation_results/                # 調査の単発結果
├─ tmp, tmp_full_run.log, tmp_optimizer.log  # 一時ファイル/ログ
├─ requirements.txt, pytest.ini          # 依存関係/テスト設定
├─ yahoo_market_data.py, stock_info_utils.py # 単体ユーティリティ
└─ プロンプト/                           # プロンプト関連のメモ/ドキュメント
```

### ディレクトリ別の役割と中身

- **daily_trading_automation.py（エントリポイント）**
  - 毎日の処理を一括実行します（データ取得 → 統合 → モデル更新 → レポート出力）。
  - 実行例: `python daily_trading_automation.py`（事前に `.env` の設定が必要です）。

- **systems/**（学習・推論の心臓部）
  - 例: `enhanced_close_return_system_v1.py`（終値→終値のパイプライン）、`enhanced_precision_system_v3.py`（複数シグナル版）。
  - 学習/推論/評価の流れをクラスや関数でまとめています。

- **src/**（共通ライブラリ）
  - 前処理、特徴量生成、評価関数など、複数のシステムから使う再利用コードをまとめています。

- **utils/**（小さな補助機能）
  - 日付処理、ログ設定、ファイル I/O 補助など、細かな便利機能を提供します。

- **data_management/**（データの収集と整備）
  - 外部 API からの取得、欠損補完、整形、保存までをスクリプトごとに担当します。
  - 生成物は基本的に `data/` の該当サブフォルダへ保存されます。

- **analysis/**（実験・検証・可視化）
  - 例: `close_optuna_search.py`（Optuna による探索）、`close_threshold_optimizer.py`（閾値最適化）。
  - 実験結果は `results/` や `reports/`、一部は `config/` に反映されます。

- **reports/**（分析スクリプト/成果物）
  - 例: `monitoring` 配下の CSV（しきい値スキャン等）。
  - スクリプトは簡易な出力や整形を行い、`production_reports/` と連携します。

- **production_reports/**（本番レポート）
  - 日付ごとの Markdown レポートを保存します（例: `production_reports/2025-10/2025-10-23.md`）。
  - 社外共有時は機微情報のマスキングに注意してください。

- **models/**（学習済みモデル/成果物）
  - 学習後の `.joblib`、評価結果 `.json/.csv` などが格納されます。
  - バージョン/日付を含む命名で追跡しやすくしています。

- **data/**（データルート）
  - 主な下位階層: `raw/`（生データ）, `processed/`（加工後）, `predictions/`（予測）, `feature/`（特徴量）, `evaluation/`（評価用）, ほか複数。
  - 大容量です。`.gitignore` 対象や外部配布不可のファイルに注意してください。

- **production_data/**（本番向け軽量データ）
  - 運用に最低限必要な JSON などを配置。自動化フローが参照します。

- **tests/**（テストコード）
  - `pytest` で実行。新しい振る舞いを追加したら、最低 1 つはテストを追加します。
  - 実行例: `pytest` または `pytest -k close_return`。

- **documentation/**（ドキュメント）
  - 運用ガイドやデータ収集の手順などの説明資料。本ファイルもここに置きます。

- **docment/**（旧ドキュメント/運用資料）
  - 過去の手順やユーザー情報などのアーカイブ。最新は `documentation/` を優先してください。

- **config/**（設定）
  - 例: `close_model_params.json`（Optuna/グリッド探索の反映）、`close_threshold.json`（推奨しきい値）、`trading_config.yaml` など。
  - 設定変更時は PR で根拠（実験結果）を添付するとトレーサブルです。

- **production_config.yaml（および backup）**
  - 本番実行のための包括設定。更新時は慎重にレビューし、テストを通してから反映します。

- **logs/**（ログ）
  - データ品質チェックや日次実行のログが入ります。障害調査の起点になります。

- **results/**（実験結果の置き場）
  - 可視化や分析の中間/最終アウトプット。再現性のため日付や条件をファイル名に含めます。

- **optimization/**（最適化スクリプト）
  - 期間最適化や閾値探索など、性能を底上げする処理をまとめています。

- **signals/**（シグナル生成）
  - 売買シグナルの作成・集約ロジック。`systems/` のモデルと組み合わせて使います。

- **production/**（本番ユーティリティ）
  - レポート整形・通知・配信など本番運用のためのスクリプト群。

- **その他の補助ディレクトリ**
  - `archive/`（節目の達成ログ等）、`data_exports/`（外部共有用出力）、`investigation_results/`（単発調査の結果）、`profit_loss_optimization_results/`（損益最適化の成果）。

### 使いどころの目安（困ったらここを見てください）

- どのファイルを実行すれば「いつもの日次」が走る？ → `daily_trading_automation.py`
- モデル学習や評価の本体は？ → `systems/`
- データの取得・整形の入口は？ → `data_management/`
- グラフや閾値調整、検証は？ → `analysis/` と `reports/`
- 本番向けに配る日報はどこ？ → `production_reports/`
- ハイパラやしきい値の設定は？ → `config/` と `production_config.yaml`
- テストはどこから？ → `tests/`（`pytest` 実行）

### 実行・運用時の注意

- 秘密情報は `.env` に保存します（例: J-Quants 認証）。`.env.template` を参考に、Git には絶対に本物の秘密情報をコミットしないでください。
- macOS の数値計算ライブラリ（Accelerate）前提の箇所があるため、NumPy/LightGBM のバージョン更新時は互換性を確認してください。
- `production_reports/` や `data_exports/` を外部共有する場合は、顧客情報や識別子を必ずマスキングしてください。

### 更新のしかた（このドキュメント）

- 新しいディレクトリを追加した/役割が変わった場合は、本ファイルの該当箇所に短く追記してください。
- 大規模変更の前後では、テスト（`pytest`）と主要スクリプトの実行手順を README/ドキュメントに反映することをおすすめします。

以上です。まずはツリーで全体をつかみ、必要に応じて各フォルダの説明を参照してください。


