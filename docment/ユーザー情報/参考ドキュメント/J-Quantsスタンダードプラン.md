# J-Quantsスタンダードプラン

## 概要
ユーザーはJ-Quantsのスタンダードプランを利用しています。そのため、スタンダードプランで取得できるデータを使用することができます。

## 以下は結論です

J-Quantsの**スタンダードプラン**では、原則「**過去10年分**」のヒストリカルデータが取得できるAPIが利用できます（例外：決算発表予定日は直近のみ／取引カレンダーは翌年末まで＋過去10年）。呼び出しは**GET**（認証はBearer **idToken**）で、主に`code`（銘柄コード）や`date`/`from`/`to`を指定します。下表に**取得できるデータ一覧**と\*\*APIの呼び出し方法（エンドポイント・主なクエリ・例）\*\*を整理しました。 ([JPX GitBook][1])

---

# 認証（共通）

1. **リフレッシュトークン取得**
   `POST https://api.jquants.com/v1/token/auth_user`
   ボディ：`{"mailaddress":"<登録メール>","password":"<パスワード>"}` → `{"refreshToken":"..."}` が返る。 ([JPX GitBook][2])
2. **IDトークン取得**
   `POST https://api.jquants.com/v1/token/auth_refresh?refreshtoken=<refreshToken>`
   → `{"idToken":"..."}`（**有効24時間**）を受け取り、以降のAPIで`Authorization: Bearer <idToken>`を付与。 ([JPX GitBook][3])

> 日付は多くのAPIで`YYYY-MM-DD`と`YYYYMMDD`どちらも受理。大量取得時は`pagination_key`で続きが返ります（共通仕様）。各API仕様ページの例に準拠。

---

# データ一覧とAPI呼び出し（スタンダードプラン）

> 例の`$ID_TOKEN`は取得済みidTokenを想定。`curl`は最小限の例です。履歴の「10年」は**スタンダードのデータ提供期間**を意味します。

## 1) 上場銘柄一覧（上場基本情報）

* **期間**：10年前まで（時点指定で過去状態も取得可） ([JPX GitBook][1])
* **Endpoint**：`GET /v1/listed/info`（ベースURL：`https://api.jquants.com`） ([JPX GitBook][4])
* **主なクエリ**：`date`（任意、基準日）、`code`（任意、銘柄絞り込み）
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/listed/info?date=2024-12-30" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 2) 株価四本値（現物株・日次）

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/prices/daily_quotes` ([JPX GitBook][4])
* **主なクエリ**：`code`、`date` または `from`/`to`、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/prices/daily_quotes?code=7203&from=2018-01-01&to=2024-12-31" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 3) 財務情報（決算短信等の要約数値）

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/fins/statements` ([JPX GitBook][5])
* **主なクエリ**：`code`、`date`（開示日 or 会計期に紐づく日付）、`from`/`to`、`type`（開示書類種別；短信/有報など）、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/fins/statements?code=6758&from=2016-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 4) 決算発表予定日

* **期間**：**直近データのみ**（全プラン共通） ([JPX GitBook][1])
* **Endpoint**：`GET /v1/fins/announcement` ([JPX GitBook][6])
* **主なクエリ**：`date` または `from`/`to`、`code`、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/fins/announcement?from=2025-09-01&to=2025-10-31" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 5) 取引カレンダー（営業日情報）

* **期間**：**翌年末〜10年前まで** ([JPX GitBook][1])
* **Endpoint**：`GET /v1/markets/trading_calendar` ([JPX GitBook][7])
* **主なクエリ**：`from`/`to`、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/markets/trading_calendar?from=2016-01-01&to=2026-12-31" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 6) 投資部門別情報（主体別売買動向・東証公表）

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/markets/trades_spec` ([JPX GitBook][8])
* **主なクエリ**：`date` または `from`/`to`、（必要に応じ **市場** 指定）、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/markets/trades_spec?from=2018-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 7) TOPIX 四本値（日次）

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/indices/topix` ([JPX GitBook][9])
* **主なクエリ**：`from`/`to`、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/indices/topix?from=2016-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 8) 指数 四本値（主要株価指数等）

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/indices`（`index`で対象指数を指定） ([JPX GitBook][10])
* **主なクエリ**：`index`（配信対象指数コードのいずれか）、`from`/`to`、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/indices?index=TOPIX&from=2016-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 9) 日経225オプション 四本値（指数オプション）

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/option/index_option`（権利行使価格や限月などの指定）
  ※エンドポイントはAPI仕様セクションに掲載。詳細パラメータは該当ページ参照。 ([JPX GitBook][4])
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/option/index_option?from=2018-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 10) 信用取引 週末残高

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/markets/weekly_margin_interest` ([JPX GitBook][11])
* **主なクエリ**：`code`、`date` または `from`/`to`、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/markets/weekly_margin_interest?code=7203&from=2016-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 11) 業種別 空売り比率

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/markets/short_selling` ([JPX GitBook][12])
* **主なクエリ**：`date` または `from`/`to`、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/markets/short_selling?from=2016-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 12) 空売り残高報告（0.5%以上の残高公表）

* **期間**：10年前まで ([JPX GitBook][1])
* **Endpoint**：`GET /v1/markets/short_selling_positions` ([JPX GitBook][13])
* **主なクエリ**：**いずれか必須** → `code` **or** `disclosed_date` **or** `calculated_date`（`*_from`/`*_to`も可）、`pagination_key`
* **例**：

  ```bash
  # 指定日の全銘柄
  curl "https://api.jquants.com/v1/markets/short_selling_positions?disclosed_date=2024-08-01" \
    -H "Authorization: Bearer $ID_TOKEN"
  # 銘柄コード＋計算日
  curl "https://api.jquants.com/v1/markets/short_selling_positions?code=86970&calculated_date=2024-08-01" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

## 13) 日々公表 信用取引残高（制度信用 等）

* **期間**：10年前まで（2025/8/22追加） ([JPX GitBook][1])
* **Endpoint**：`GET /v1/markets/daily_margin_interest` ([JPX GitBook][14])
* **主なクエリ**：`code`、`date` または `from`/`to`、`reason_code`（公表理由の絞り込み等）、`pagination_key`
* **例**：

  ```bash
  curl "https://api.jquants.com/v1/markets/daily_margin_interest?code=7203&from=2016-01-01&to=2025-09-06" \
    -H "Authorization: Bearer $ID_TOKEN"
  ```

---

# 使えない（プレミアム限定など）主なAPI（補足）

スタンダードでは以下は**対象外**です（混同しやすいので注意）。

* 先物四本値 `/v1/derivatives/futures`、オプション四本値（個別商品） `/v1/derivatives/options`、売買内訳 `/v1/markets/breakdown`、前場四本値 `/v1/prices/prices_am`、配当金情報 `/v1/fins/dividend`、財務諸表（BS/PL明細） `/v1/fins/fs_details`。 ([JPX GitBook][1])

---

# 実務Tips（中学生にもわかる簡潔版）

* **まず鍵を取る**：メール＋パスワードで**refreshToken**→それを使って**idToken**（24時間有効）。鍵（idToken）をヘッダにつける。 ([JPX GitBook][2])
* **日付は幅で取る**：`from`と`to`を付けると、期間の連続データが返って便利。
* **多すぎる時**：`pagination_key`が返るので、その値を次の呼び出しで付けて続き取得。
* **銘柄コード**：原則4桁（優先株などで5桁になる場合あり）。空売り残高報告は**5桁**で返る点に注意。 ([JPX GitBook][13])

---

# 参考（公式ドキュメント）

* **プラン別で使えるAPIと期間の一覧**（最新）： ([JPX GitBook][1])
* **API仕様（総合インデックス）**：各エンドポイントの詳細・サンプルコードにリンクあり。 ([JPX GitBook][4])
* **個別仕様ページ（例）**：投資部門別情報、信用取引残高、空売り関連、指数、TOPIX、財務情報など。 ([JPX GitBook][8])

---

必要なら、**Python（requests/pandas）での一括取得・自動ページング処理テンプレート**もすぐ用意します。どのAPIを最優先で回したいか（例：株価→財務→空売り…）だけ教えてください。

---

# 実装済み拡張データ取得システム（2025年9月6日追加）

## 🔧 J-Quants API問題解決

### **銘柄コード形式の問題と解決**
```
❌ 問題: 400エラーの原因判明
- 使用コード: 4桁形式（7203, 9984等）
- J-Quants形式: 5桁形式（13320, 13330等）

✅ 解決: 上場銘柄一覧API活用
GET /v1/listed/info → 正確な銘柄コード取得
レスポンス例: {"Code": "86970", "CompanyName": "日本取引所グループ"}
```

### **実装システム: jquants_correct_fetcher.py**
```python
# 主要機能
1. 上場銘柄一覧取得（4,408社）
2. プライム市場大型株自動選択（473社）
3. 正確な5桁コードでの株価データ取得
4. 既存データとの統合・重複除去
5. 拡張データセット自動生成

# 取得仕様
- 期間: 5年間（2020年9月〜2025年9月）
- 対象: プライム市場大型株50銘柄
- 期待レコード数: 約61,250件
- API制限対応: 2秒間隔、10銘柄ごと10秒待機
```

## 📊 データ拡張完了と精度検証結果（2025年9月6日最終）

### **✅ 拡張データ取得完了**
```
実行完了: 2025年9月6日17:06
取得データ: 61,250件（50銘柄×5年間）
統合後データ: 86,975件（既存25,725件+拡張61,250件）
データファイル: enhanced_jquants_86975records_20250906_165041.parquet
処理時間: 約2時間（API制限対応）
```

### **❌ 拡張データ精度検証結果**
```
ベースライン精度: 57.58%（既存データ）
拡張データ精度: 51.52%（拡張データ後）
精度変化: -6.06ポイント（-10.5%劣化）

検証結果分析:
- 予想: 65-70%精度向上 → 実際: 51.52%精度劣化
- 原因: データ品質差、市場環境差（コロナ期間含む）
- 結論: 単純なデータ量増加では精度向上しない
```

### **🔍 問題分析と学習**
```
問題1: データ品質の不均一性
- 既存データ（2022-2025）: 安定成長期
- 拡張データ（2020-2025）: コロナショック期間含む

問題2: モデルパラメータの不適合
- 小規模データ用パラメータを大規模データに適用
- ハイパーパラメータ最適化未実施

問題3: 過学習の発生
- 86,975件での複雑なパターン学習
- 汎化性能の劣化
```

### **💡 今後の改善方針**
```
✅ 現在の推奨: 既存25,725件データでの運用継続（57.58%精度）
🔧 改善案1: 拡張データ用ハイパーパラメータ最適化
🔧 改善案2: データ品質向上とクリーニング
🔧 改善案3: 時期別モデルアンサンブル手法
```

[1]: https://jpx.gitbook.io/j-quants-ja/outline/data-spec "プランごとに利用可能なAPIとデータ期間 | J-Quants API"
[2]: https://jpx.gitbook.io/j-quants-ja/api-reference/refreshtoken "リフレッシュトークン取得(/token/auth_user) | J-Quants API"
[3]: https://jpx.gitbook.io/j-quants-ja/api-reference/idtoken "IDトークン取得(/token/auth_refresh) | J-Quants API"
[4]: https://jpx.gitbook.io/j-quants-ja/api-reference "API仕様 | J-Quants API"
[5]: https://jpx.gitbook.io/j-quants-ja/api-reference/statements "財務情報(/fins/statements) | J-Quants API"
[6]: https://jpx.gitbook.io/j-quants-ja/api-reference/announcement "決算発表予定日(/fins/announcement) | J-Quants API"
[7]: https://jpx.gitbook.io/j-quants-ja/api-reference/trading_calendar "取引カレンダー(/markets/trading_calendar) | J-Quants API"
[8]: https://jpx.gitbook.io/j-quants-ja/api-reference/trades_spec "投資部門別情報(/markets/trades_spec) | J-Quants API"
[9]: https://jpx.gitbook.io/j-quants-ja/api-reference/topix "TOPIX指数四本値(/indices/topix) | J-Quants API"
[10]: https://jpx.gitbook.io/j-quants-ja/api-reference/indices "指数四本値(/indices) | J-Quants API"
[11]: https://jpx.gitbook.io/j-quants-ja/api-reference/weekly_margin_interest "信用取引週末残高(/markets/weekly_margin_interest) | J-Quants API"
[12]: https://jpx.gitbook.io/j-quants-ja/api-reference/short_selling "業種別空売り比率(/markets/short_selling) | J-Quants API"
[13]: https://jpx.gitbook.io/j-quants-ja/api-reference/short_selling_positions "空売り残高報告(/markets/short_selling_positions) | J-Quants API"
[14]: https://jpx.gitbook.io/j-quants-ja/api-reference/daily_margin_interest "日々公表信用取引残高(/markets/daily_margin_interest) | J-Quants API"
