# 🤖 AI株式取引システム 運用ガイド

## 📋 概要

このシステムは、AI予測に基づく株式投資の支援を行う運用システムです。
機械学習による銘柄選定、リスク管理、レポート生成を自動化し、楽天証券での手動取引をサポートします。

---

## 🚀 コマンド一覧

### **1. メイン運用コマンド（日常使用）**

```bash
# フル運用分析実行（推奨）
python production_trading_system.py
```

**実行内容:**
- 最新データ読み込み・AI予測
- 購入推奨レポート生成
- 保有銘柄管理レポート生成  
- パフォーマンス分析
- 詳細レポートファイル出力

**出力ファイル:**
- `production_reports/YYYYMMDD/trading_reports_YYYYMMDD_HHMMSS.md` - **見やすいMarkdownレポート（メイン）**
- `production_reports/YYYYMMDD/trading_reports_YYYYMMDD_HHMMSS.txt` - テキストレポート（コンソール用）
- `production_reports/YYYYMMDD/trading_reports_YYYYMMDD_HHMMSS.json` - 詳細データ（JSON形式）

---

### **2. パラメータ最適化コマンド（定期実行推奨）**

```bash
# 直近1年間で最適化（推奨）
python production_optimizer.py optimize recent_1year

# 直近6ヶ月で最適化
python production_optimizer.py optimize recent_6months

# 直近3ヶ月で最適化
python production_optimizer.py optimize recent_3months
```

**実行内容:**
- 指定期間でのパラメータ全組み合わせ検証
- 最優秀パラメータの自動検出
- 設定ファイル（`production_config.yaml`）への自動反映

**実行時間目安:**
- 直近1年間: 約15-20分（1,800パターン検証）
- 直近6ヶ月: 約10-15分
- 直近3ヶ月: 約5-10分

**出力ファイル:**
- `production_optimization_results_recent_1year_YYYYMMDD_HHMMSS.csv`

---

### **3. ヘルプ・テストコマンド**

```bash
# ヘルプ表示
python production_trading_system.py --help

# テストモード（サンプルデータでレポート生成）
python production_trading_system.py test

# レポート生成のみテスト
python production_reports.py
```

---

## ⚙️ 設定ファイル管理

### **設定ファイル: `production_config.yaml`**

```yaml
# 運用パラメータ（最適化で自動更新）
optimal_params:
  hold_days: 9        # 保有期間（日）
  profit_target: 0.14 # 利確閾値（14%）
  stop_loss: 0.06     # 損切閾値（6%）

# 基本設定（手動調整可能）
system:
  initial_capital: 1000000  # 初期資金（円）
  max_positions: 5          # 最大保有銘柄数
  confidence_threshold: 0.55 # AI信頼度閾値
```

**重要:** パラメータ最適化を実行すると、`optimal_params`が自動で最新の最優秀値に更新されます。

---

## 📅 推奨運用フロー

### **日常運用（平日夜 or 朝）**

```bash
# 1. メイン運用分析実行
python production_trading_system.py
```

1. 生成されたテキストレポートを確認
2. 購入推奨銘柄を楽天証券で指値注文
3. 売却推奨銘柄を楽天証券で指値注文
4. 利確・損切り価格を指値で設定

### **定期最適化（月1回程度）**

```bash
# 1. 最新データでパラメータ最適化
python production_optimizer.py optimize recent_1year

# 2. 最適化後に運用分析実行
python production_trading_system.py
```

---

## 📊 レポート内容説明

### **購入推奨レポート**

```
推奨銘柄一覧:
コード | 会社名     | 現在値  | 推奨株数 | 投資額     | 利確価格 | 損切価格 | 信頼度
  7203 | トヨタ自動車 | ¥2,800 |    100株 | ¥280,000 | ¥3,192 | ¥2,632 |  75.0%
```

- **推奨株数**: 分散投資を考慮した推奨購入株数
- **利確価格**: 14%上昇時の売却指値
- **損切価格**: 6%下落時の売却指値  
- **信頼度**: AI予測の確信度

### **保有銘柄管理レポート**

```
保有銘柄一覧:
コード | 会社名     | 株数   | 買値    | 現在値  | 評価損益  | 保有日数 | 売却判定
  7203 | トヨタ自動車 |   100株 | ¥2,750 | ¥2,800 | ¥+5,000 |     8日 | 保有継続
```

- **売却判定**: 利確・損切・期間満了の自動判定
- **評価損益**: 現在の含み損益（手数料込み）

---

## 🛠️ トラブルシューティング

### **よくあるエラーと解決方法**

#### **1. データファイルが見つからない**
```
❌ データファイルが見つかりません: data/processed/integrated_with_external.parquet
```

**解決方法:** データ収集スクリプトを先に実行
```bash
# 既存のデータ収集スクリプトを実行（具体的なファイル名は環境に応じて）
python scripts/data_collection_script.py
```

#### **2. 設定ファイルが読めない**
```
❌ 設定ファイル読み込みエラー
```

**解決方法:** 設定ファイルの文法確認
```bash
# YAML文法チェック
python -c "import yaml; yaml.safe_load(open('production_config.yaml'))"
```

#### **3. 最適化が長時間かかる**
```
📊 進捗: 100/1800 (5.6%) - 現在最高年率: 14.72%
```

**対処法:**
- そのまま待つ（バックグラウンド実行推奨）
- CPUコア数を減らす: `production_config.yaml` の `max_cpu_cores: 4` に変更

---

## 📁 ファイル・ディレクトリ構造

```
claude_code_develop/
├── production_config.yaml           # 設定ファイル
├── production_trading_system.py     # メイン運用システム
├── production_optimizer.py          # パラメータ最適化
├── production_reports.py           # レポート生成
├── production_data/                # 運用データ保存
│   ├── current_portfolio.json     # 現在のポートフォリオ
│   └── trade_history.json         # 取引履歴
└── production_reports/             # レポート出力
    └── YYYYMMDD/                  # 日付別フォルダ
        ├── trading_reports_*.md   # Markdownレポート（メイン）
        ├── trading_reports_*.txt  # テキストレポート
        └── trading_reports_*.json # JSONレポート
```

---

## 🔒 セキュリティと注意事項

### **重要な注意点**

1. **本システムは投資助言ではありません** - 最終的な投資判断は自己責任で行ってください
2. **過去の成績は将来を保証しません** - 市場環境変化により性能が変わる可能性があります
3. **小額でのテスト運用を推奨** - 最初は少額で動作確認を行ってください
4. **定期的なパラメータ最適化** - 月1回程度の再最適化を推奨します

### **データバックアップ**

重要ファイルの定期バックアップを推奨:
```bash
# 設定ファイル・運用データのバックアップ
cp production_config.yaml backup/
cp -r production_data/ backup/
cp -r production_reports/ backup/
```

---

## 💡 運用のコツ

### **効果的な使用方法**

1. **市場終了後の分析**: 15:30以降にメインコマンド実行
2. **翌朝の注文設定**: 生成されたレポートを基に楽天証券で指値注文
3. **週次での見直し**: 金曜日に週間パフォーマンスを確認
4. **月次での最適化**: 月初に最適化コマンド実行

### **パフォーマンス向上のポイント**

- **信頼度の高い銘柄に集中**: 75%以上の信頼度を優先
- **分散投資の維持**: 最大5銘柄での分散を推奨
- **感情的な判断を避ける**: システム推奨に従った機械的な売買

---

## 📞 サポート

このシステムに関する質問や改善提案があれば、開発者にお知らせください。

**現在の設定値（2025-08-31時点）:**
- 保有期間: 9日
- 利確閾値: 14%  
- 損切閾値: 6%
- 年率リターン: 123.21%（直近1年検証）

---

*🤖 AI株式取引システム - 効率的で科学的な投資支援ツール*