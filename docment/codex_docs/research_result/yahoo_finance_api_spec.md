# Yahoo Finance API仕様まとめ（2024-10時点）

Yahoo! Financeは公式に商用APIを公開していませんが、公開Web API（`query1/2.finance.yahoo.com`）を呼び出すことでマーケットデータを取得できます。当プロジェクトでは `yfinance` ライブラリを介してこれらのエンドポイントを利用し、外部指標（USD/JPY、VIX、指数、コモディティ等）を補完しています（例: `data_management/yahoo_finance_extended_fetcher.py:21`、`data_management/highfreq_market_feature_builder.py:34`）。本書では実務で使用する主要エンドポイント、パラメータ、データ範囲、レート制約、ライセンス上の注意点を整理します。

## 情報ソース
- 社内コード: `data_management/yahoo_market_data.py`（日次OHLC取得）、`data_management/yahoo_finance_extended_fetcher.py`（10年取得）、`data_management/highfreq_market_feature_builder.py`（5分足〜1時間足）など。
- `docment/データ利用ガイド_外部ソース_20251026.md`（Yahoo!関連のライセンス方針）。
- `yfinance` 0.2.18 のドキュメント / GitHub Wiki（非公式APIの仕様解説）。
- 一般公開されている Yahoo! Finance Webリクエスト（Chrome DevToolsで確認可能）。

## 1. 主要エンドポイント一覧

| エンドポイント | 用途 | 主なクエリ/パラメータ | 備考 |
| --- | --- | --- | --- |
| `GET https://query1.finance.yahoo.com/v8/finance/chart/{symbol}` | 時系列OHLCV（1分〜1か月足） | `interval`, `range` または `period1/period2`（UNIX秒）, `events=div%2Csplits`, `includePrePost`, `lang`, `region` | `yfinance.Ticker(symbol).history()` はこのJSONを内部使用。 |
| `GET https://query1.finance.yahoo.com/v7/finance/download/{symbol}` | CSV形式のヒストリカルデータ | `period1`, `period2`, `interval`, `events=history`, `includeAdjustedClose=true` | ブラウザDLリンクと同一。`yfinance.download()` は必要なクッキー/crumbを自動解決。 |
| `GET https://query1.finance.yahoo.com/v7/finance/quote` | 複数銘柄のリアルタイム気配 | `symbols=AAPL,MSFT`, `fields`（任意） | `Ticker.fast_info` や `yf.Tickers` が内部利用。遅延15分前後。 |
| `GET https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}` | ファンダメンタル/統計/財務指標 | `modules=price,summaryProfile,defaultKeyStatistics,...` | モジュールごとに複数JSONセクションをまとめて返却。 |
| `GET https://query1.finance.yahoo.com/v7/finance/options/{symbol}` | 株式オプションチェーン | `date`（UNIX秒の満期、未指定で最短満期） | 返値に`calls`/`puts` 配列を含む。 |
| `GET https://query1.finance.yahoo.com/v6/finance/recommendationsbysymbol/{symbol}` | アナリスト推奨履歴 | なし | 過去数カ月分の推奨レーティングが返る。 |
| `GET https://query1.finance.yahoo.com/v1/finance/trending/{region}` | 地域別トレンド銘柄 | `region=US/J P`等 | モバイルアプリで使用。 |
| `GET https://query1.finance.yahoo.com/v8/finance/spark` | 複数銘柄の短期チャートを一括取得 | `symbols=`, `interval=`, `range=` | UIのスパークライン表示向け。 |
| `GET https://query1.finance.yahoo.com/ws/insights/v1/finance/insights` | 需給/ニュースインサイト | `symbol=` | ベータ機能、変化頻度は低い。 |

> `query2.finance.yahoo.com` は `query1` のミラー。レスポンス構造は同じです。

## 2. 時系列データの取得仕様

### 2.1 サポート区間と粒度
Yahoo! Financeは足種ごとに取得可能期間が決まっており、`yfinance` も同じ制約をラップします。`data_management/highfreq_market_feature_builder.py:48` のデフォルトマップを基に整理すると以下の通りです。

| Interval | 取得可能期間（概ね） | 用途 |
| --- | --- | --- |
| `1m` | 過去7日 | 超短期検証（`fetch_symbol(..., interval="1m")`）。 |
| `2m` | 過去60日 | 公式には未公開、`chart`が返す場合あり。 |
| `5m` | 過去60日/`yfinance`デフォルト5日 | 本システムの高頻度指標（N225, USDJPY等）。 |
| `15m` | 過去60日 | Fallback間隔として自動選択。 |
| `30m`, `60m`, `90m`, `1h` | 最大2年（730日） | 中期の需給指標。 |
| `1d`, `5d` | 最大1970年まで（実際は1950年代以降） | `yahoo_finance_extended_fetcher` にて10年分取得。 |
| `1wk`, `1mo`, `3mo` | 過去最大50年以上 | 長期ベンチマーク。 |

### 2.2 代表的なパラメータ
- `period1`, `period2`: UNIX秒（UTC）。`period1=0` で最古まで。`yfinance.download` は日付文字列を内部で変換。
- `range`: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`。`range` 指定時は `period1/2` 無効。
- `interval`: 上記表。`chart` APIはサポート外の組み合わせを 400 で拒否。
- `includePrePost`: プレ/アフターマーケットを含めるか。
- `events`: `div`, `split`, `capitalGain`。`chart` の `events_data` ブロックに含まれる。
- `crumb`: `download` エンドポイントで必要なCSRFトークン。`yfinance` が自動挿入。

## 3. レート制限と安定運用Tips
- Yahoo! は公式のリミット値を公開していません。経験上 **2〜5リクエスト/秒** を超えると HTTP 429/999 が返ります。長時間バッチでは **1req/sec + 0.5秒ジッタ** を推奨。
- 大量シンボルは `yf.download(["^N225","^TOPX"], group_by="ticker")` でまとめて1リクエストにする。
- `yfinance` 0.2系では `session=requests_cache` によるキャッシュ機構あり。`CachedSession('yfinance.cache')` を使うと429を避けやすい。
- ネットワーク失敗時は `yfinance` が指数バックオフしないため、呼び出し側でリトライ/`time.sleep` を挿入（`data_management/yahoo_finance_extended_fetcher.py:116` が `time.sleep(1)` でレート制限を吸収）。
- Web側は頻繁にエンドポイントを変更するため、`yfinance` を最新に保つ（`requirements.txt:25` で 0.2.18 を固定）。

## 4. データカテゴリ別仕様

### 4.1 ヒストリカルOHLCV
- エンドポイント: `v8/finance/chart`, `v7/finance/download`.
- データ列: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`. `Adj Close` は配当・分割調整済み。
- 欠損: 祝日は空。`auto_adjust=True` を指定すると `Close` を調整済みで返す（`yahoo_finance_extended_fetcher.py:63`）。
- 通貨/指数: 国ごとにティッカーが異なる（例: 日経平均 `^N225`, TOPIX `^TOPX`, US 10Y `^TNX`, FX `USDJPY=X`）。`YahooMarketData.market_symbols` に一覧あり（`data_management/yahoo_market_data.py:18`）。

### 4.2 オプション/先物
- `v7/finance/options/{symbol}` は株式オプション専用。返却JSONに `expirationDates`、`strikes`、`calls`、`puts`。
- 先物は通常のシンボル（例: `ES=F`, `GC=F`, `CL=F`）として `chart` から取得。
- 建玉/IVなどは`quoteSummary`の `optionChain` モジュールを参照。

### 4.3 ファンダメンタル・統計
- `quoteSummary` の `modules` に `summaryProfile`, `financialData`, `defaultKeyStatistics`, `calendarEvents`, `recommendationTrend` などを列挙。
- 1回のリクエストで複数モジュールを要求可能。未サポートモジュールは `null`。
- 更新頻度は日次〜四半期。証券会社の公式フィードに比べ遅延やデータ欠損があるため、重要数値は公的ソースで検証が必要。

### 4.4 マーケットデータ／ムーバー
- `market/v2/get-quotes`, `market/v2/get-summary`, `market/get-movers` などのRESTはアプリが使用。リージョン/カテゴリ（`MOST_ACTIVE`, `GAINERS` 等）を指定するとランキングが返る。
- レスポンスには `marketState`, `regularMarketPrice`, `postMarketChangePercent` などリアルタイム近いフィールドが含まれるが、15分ディレイ扱い。

## 5. yfinanceによる実装パターン

### 5.1 日次データ（10年分）
```python
from data_management.yahoo_finance_extended_fetcher import YahooFinanceExtendedFetcher
fetcher = YahooFinanceExtendedFetcher(start_date="2015-01-01", end_date="2025-01-01")
usd_jpy = fetcher.fetch_symbol_data("USDJPY=X", "usdjpy")  # chart/download を透過的に利用
```

### 5.2 マルチシンボル一括
```python
import yfinance as yf
df = yf.download(["^N225","^TOPX","USDJPY=X"], start="2022-01-01", end="2025-01-01", group_by="ticker", auto_adjust=True)
```

### 5.3 5分足特徴量
```python
from data_management.highfreq_market_feature_builder import HighFreqMarketFeatureBuilder
builder = HighFreqMarketFeatureBuilder()
frames = builder.fetch_all(period="5d", interval="5m")  # intervalごとに可用期間を自動調整
features = builder.build_features(frames)
```

## 6. ライセンスと利用上の注意
- Yahoo! JAPAN / Yahoo! Inc. の利用規約では、無断転載・大量ダウンロード・商用再配布を禁止。商用プロダクションに組み込む場合は公式データプロバイダ（証券会社API、QUICK、Bloomberg等）を推奨（`docment/データ利用ガイド_外部ソース_20251026.md:30`）。
- `yfinance` はコミュニティプロジェクトであり、Yahoo! によるSLA・サポートは存在しない。仕様変更で突然停止するリスクがあるため、`data_management/highfreq_market_feature_builder.py:137` のようにローカルフォールバックや代替データを常備する。
- 取得したCSV/Parquetは社内限定共有とし、対外資料で引用する際は出典を明記する。

## 7. 監査・運用メモ
1. **ログ保全**: 外部データ取得コマンドの実行時間・件数を `logs/external_data_access.log` に残し、アクセス過多によるBANを避ける。
2. **キャッシュ**: `requests-cache` で1日キャッシュを挟むと再取得を減らせる。`yfinance` 0.2.18 では `yf.utils.get_yf_session()` を差し替え可能。
3. **バリデーション**: `data_management/yahoo_market_data.py:69` のように`Date`を統一し、欠損や空レス時は WARN を残す。
4. **プラン比較**: J-Quants等の公式ソースで日次データを賄える場合はそちらを優先し、Yahoo!は指数/FXなど補完用途に限定する。

---
これらの仕様を踏まえ、Yahoo!由来データを参照するスクリプトには「非公式APIでありエラー頻度が高い」「大量呼び出しは禁止されている」旨をコメントで明記し、長期的には正式なマーケットデータ契約に移行してください。
