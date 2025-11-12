# codex_loop 運用手順

Codex CLI を非対話バッチで回す際のドライラン→本番実行→定期スケジューリングまでをまとめました。`scripts/codex_loop.sh` と `codex_loop.py` を前提にしています。

## 1. 事前チェック
- `python3` と `codex` コマンドが PATH 上にあることを確認（`which python3`, `which codex`）。
- プロンプト一覧ファイルを `docment/codex_docs/*.txt` 等に用意（1 行 1 プロンプト。`#` と空行は無視）。
- 必要なら環境変数を設定  
  `PROMPTS_FILE`, `OUTPUT_DIR`, `LOG_DIR`, `CODEX_CMD`, `EXTRA_ARGS`, `PYTHON_BIN`, `RUN_LABEL`, `COMMAND_TIMEOUT`, `PROMPT_INTERVAL`, `FAIL_ON_ERROR`, `HISTORY_LOG`
- 期待する書き込み先を作成 or `scripts/codex_loop.sh` に任せる（`results/codex_cli_runs`, `logs/codex_loop`）。
- manifest (`manifest.json`) は各実行のメタデータを残すので、成果物パスや失敗状況を後追いしやすくなりました。

## 2. ドライラン（コマンド検証）
設定と PATH を確認するだけなら `-n` で十分です。

```bash
scripts/codex_loop.sh \
  -p docment/codex_docs/codex_loop_prompts.sample.txt \
  -e "--approvals full --max-output 1" \
  -i 2 \
  -t 60 \
  -n
```

履歴を別ディレクトリにまとめたい場合は `-H logs/codex_loop/run_history.jsonl` などを付与するか、`HISTORY_LOG` 環境変数で既定値を上書きできます（Dry-run でもコマンド文字列に反映されます）。

出力される `🧪 Dry-run command` を確認し、ログファイル/出力ディレクトリが意図通りかを見ます。失敗した場合は `PROMPTS_FILE` や `codex_loop.py` の存在をチェックしてください。Dry-run 中はファイル書き込みが起こらないため、再実行したいコマンドはシェル履歴や `script` コマンドで控えておくとトラブル調査が容易になります。

## 3. 本番実行
Dry-run をパスしたら `-n` を外して実行。`RUN_LABEL` を付与するとログ＆成果物が識別しやすくなります。

```bash
RUN_LABEL="nightly" EXTRA_ARGS="--approvals full --max-output 2" FAIL_ON_ERROR=1 \
HISTORY_LOG="logs/codex_loop/run_history.jsonl" \
scripts/codex_loop.sh \
  -p docment/codex_docs/codex_loop_prompts.sample.txt \
  -o results/codex_cli_runs \
  -l logs/codex_loop \
  -t 300 \
  -i 2 \
  -f
```

`RUN_LABEL` はログだけでなく `results/codex_cli_runs/<timestamp>_nightly/` のように成果物ディレクトリ名にも反映され、各 run の `manifest.json` へプロンプトごとのステータスが保存されます。`logs/codex_loop/<timestamp>_nightly.log` には codex CLI の標準出力が残ります。

さらに `--history-log`（デフォルト: `results/codex_cli_runs/run_history.jsonl`）へ各 run の要約が追記されるため、cron/launchd での実行結果を一本の JSON Lines で監査できます。失敗行だけを `rg '"had_failure": true' run_history.jsonl` で抽出し、該当 manifest を辿る運用がおすすめです。

## 4. 成果物とログの確認ポイント
1. `logs/codex_loop/<timestamp>.log`  
   - `✅ codex_loop.py が正常終了`：成功  
   - `❌` 行：CLI 失敗。`stderr` を追跡。
2. `results/codex_cli_runs/<timestamp>/texts/*.txt`  
   - 各プロンプトの生出力。`stderr` 付きファイルがあればコマンド失敗。
3. `results/codex_cli_runs/<timestamp>/manifest.json`  
   - プロンプト/コマンド/終了コード/実行秒数を一覧化し、`summary` には `ok/error/timeout` 件数と `had_failure`、全体の処理時間がまとまります。`status:error/timeout` の行があれば即把握できる。
4. `results/codex_cli_runs/run_history.jsonl`（または `-H` で指定したパス）  
   - 各 run の manifest/summary/タイムスタンプを JSON Lines で追記。cron/launchd 実績を一覧し、失敗 run を素早く抽出可能。
5. `codex_loop.py` が 0 exit か（`echo $?`）。

## 5. スケジューリング案
### 5.1 cron
単純な日次実行で十分なら cron で `/usr/local/bin` など PATH を明示。

```cron
# codex_loop nightly at 23:30 JST
30 23 * * * cd /Users/naoya/Desktop/AI関係/自動売買ツール/claude_code_develop && \
  RUN_LABEL="nightly" \
  COMMAND_TIMEOUT=300 \
  EXTRA_ARGS="--approvals full" \
  HISTORY_LOG="logs/codex_loop/run_history.jsonl" \
  /bin/zsh scripts/codex_loop.sh \
    -p docment/codex_docs/codex_loop_prompts.sample.txt \
    -o results/codex_cli_runs \
    -l logs/codex_loop >> logs/codex_loop/cron_stdout.log 2>&1
```

運用メモ:
- `cd` でリポジトリに入ってからスクリプトを呼ぶ。
- 出力をファイルへリダイレクトしておくと cron 失敗時の調査が容易。
- ループがハングしやすい場合は `COMMAND_TIMEOUT` を延長 or 短縮し、manifest で `status:timeout` を確認。
- テンプレートを編集して適用: `scripts/templates/cron/codex_loop.cron.sample` → `crontab scripts/templates/cron/codex_loop.cron.sample`

### 5.2 launchd (macOS 推奨)
launchd ならスリープ復帰時実行やログ分離が容易です。`~/Library/LaunchAgents/com.local.codex_loop.plist` の例:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.local.codex_loop</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>23</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>WorkingDirectory</key>
  <string>/Users/naoya/Desktop/AI関係/自動売買ツール/claude_code_develop</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>RUN_LABEL</key><string>nightly</string>
    <key>EXTRA_ARGS</key><string>--approvals full</string>
    <key>COMMAND_TIMEOUT</key><string>300</string>
    <key>HISTORY_LOG</key><string>logs/codex_loop/run_history.jsonl</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>scripts/codex_loop.sh</string>
    <string>-p</string>
    <string>docment/codex_docs/codex_loop_prompts.sample.txt</string>
    <string>-o</string>
    <string>results/codex_cli_runs</string>
    <string>-l</string>
    <string>logs/codex_loop</string>
  </array>
  <key>StandardOutPath</key>
  <string>/Users/naoya/Desktop/AI関係/自動売買ツール/claude_code_develop/logs/codex_loop/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/naoya/Desktop/AI関係/自動売買ツール/claude_code_develop/logs/codex_loop/launchd_stderr.log</string>
</dict>
</plist>
```

適用手順:
1. `launchctl load ~/Library/LaunchAgents/com.local.codex_loop.plist`
2. `launchctl list | grep codex_loop` で状態確認
3. 更新後は `launchctl unload ... && launchctl load ...`
- テンプレート: `scripts/templates/launchd/com.local.codex_loop.plist.sample` を `~/Library/LaunchAgents/` にコピーしてパスを書き換える。

## 6. トラブルシューティング
- `codex_loop.py が見つからない`: リポジトリパスを確認 (`ROOT_DIR`)。
- `codex: command not found`: `which codex` で CLI をインストール or `CODEX_CMD` でフルパス指定。
- 出力フォルダの権限エラー: `OUTPUT_DIR`/`LOG_DIR` を `mkdir -p` して権限を整える。
- manifest が生成されない: `codex_loop.py` 実行前にプロンプトが空でないかを確認し、`results/.../<timestamp>/manifest.json` の親ディレクトリ権限を見直す。
- レートリミット懸念: `-i` or `PROMPT_INTERVAL` でプロンプト間に待機を挿入し、manifest の `interval_sec` で効果を確認。
- タスク重複を避ける: `RUN_LABEL` でジョブ名を付け、ログ検索しやすくする。

## 7. 履歴レポート (`codex_history_report.py`)
run history を JSONL へ蓄積したあと、`codex_history_report.py` で最近のジョブ状況を即座に確認できます。

```bash
# 直近3日間の失敗 run のみ表示
python codex_history_report.py --last-days 3 --failures-only

# nightly ラベルの最新5件を確認
python codex_history_report.py --label nightly --limit 5

# 履歴ファイルを指定
python codex_history_report.py --history logs/codex_loop/run_history.jsonl
```

出力例:

```
Showing 2 of 4 matching runs (newest first)
--------------------------------------------------------------------------------
2025-11-08 23:30:02 | label=nightly | prompts=4 (ok=4 error=0 timeout=0) | OK | /.../manifest.json
...
Totals: runs=4 failures=1 prompts=16 ok=15 error=1 timeout=0
```

主なフィルタ:
- `--label`: `RUN_LABEL` ごとの成功/失敗傾向を把握。
- `--since` または `--last-days`: メンテナンス期間だけを抽出。
- `--failures-only`: 失敗 run の manifest を優先的に追跡。
- `--prompts-file`: 監視対象のプロンプトリスト（絶対パス）で絞り込み。`nightly` と `weekly` が同じ `run_label` を共有している場合でも、誤検知なく対象ジョブだけを抽出できる。

cron/launchd と組み合わせれば、`rg '"FAIL"' run_history.jsonl` からの手作業検索を減らし、エラー収集を高速化できます。

### 7.1 監視用の終了コード
`codex_history_report.py` を監視スクリプトとして呼び出したい場合は、以下のフラグで終了コードを制御できます。

- `--require-matches`: フィルタ結果が 0 件なら exit code 2。例: `python codex_history_report.py --last-days 1 --label nightly --require-matches` で「24時間以内に nightly run が存在するか」を監視。
- `--fail-when-failure`: 1 件でも `summary.had_failure=true` があれば exit code 3。例: `python codex_history_report.py --last-days 1 --label nightly --fail-when-failure` を cron で実行し、失敗検知時のみアラートを飛ばす。

両フラグを組み合わせれば「昨日以降の nightly run が存在し、全て成功しているか」を 1 コマンドでチェックできます。exit code を監視サービスに渡すだけで簡易なアラートが構成可能です。

### 7.2 カバレッジ監視（run／prompt 数の下限チェック）
定期ジョブが「予定回数を回っているか」「全プロンプトを消化できたか」も `codex_history_report.py` で確認できます。時間単位で直近のみを監視したい場合は `--last-hours N`（`--last-days`/`--since` と排他）を使うと「直近6時間で1 run 以上」などの SLA をそのまま反映できます。

- `--min-runs N` : フィルタ条件に合致する run が N 件未満なら exit code 2。例: `--last-days 1 --label nightly --min-runs 2` で「1 日に 2 回以上まわっているか」を監視。
- `--min-prompts N` : フィルタ結果全体の `summary.total_prompts` 合計が N 件未満なら exit code 2。run が 1 件だけでも、全プロンプト数でカバレッジを検証したいときに使用。
- `--min-prompts-per-run N` : 各 run ごとの `total_prompts` が N 未満なら exit code 2。途中でループが中断してプロンプトが欠けていないかを即判別できる。

例: 「昨日以降に nightly run が 2 回以上あり、かつ各 run のプロンプト数が 3 件以上であること」をチェックする場合

```bash
python codex_history_report.py \
  --last-days 1 \
  --label nightly \
  --min-runs 2 \
  --min-prompts-per-run 3 \
  --require-matches
```

`--fail-when-failure` と併用すると、run が所定回数走ったうえで失敗がないかも同時に監視できます（失敗検知時は exit code 3 が優先）。カバレッジ不足は exit code 2、実 run での失敗は exit code 3 という棲み分けになるため、監視サービス側で原因を切り分けやすくなります。

## 8. run_history watchdog スクリプトとテンプレート
`codex_history_report.py` を直接呼ぶ代わりに、`scripts/codex_history_watchdog.sh` を使うと監視コマンドの記述とログ整形を共通化できます。実行前に `chmod +x scripts/codex_history_watchdog.sh` 済みであることを確認してください。

### 8.1 CLI の流れ
```bash
scripts/codex_history_watchdog.sh \
  -H results/codex_cli_runs/run_history.jsonl \
  -L nightly \
  -a 6 \
  -m 2 \
  -p 6 \
  -u 3 \
  -F \
  -R
```

- `📊` ログでフィルタ条件が明示され、`Totals: ...` の下に `✅ coverage OK` or `❌ watchdog exit ...` が残る。
- `-n` で Dry-run できるため、cron/launchd に組み込む前にコマンド確認が容易。
- 主要オプションは環境変数 (`RUN_LABEL`, `MIN_RUNS`, `MIN_PROMPTS` など) でも上書き可能。`-k` で `--limit` を指定。
- `-d` (days) と `-a` (hours) はどちらか一方を指定し、run history の監視ウィンドウを柔軟に切り替えられる。
- `-B include|exclude|only` で backfill 済みエントリの扱いを切り替え (`--backfilled`)。`codex_history_backfill.py` で復元した run を監視対象から除外したい場合は `-B exclude` を指定。
- `-P PATH` or `PROMPTS_FILE=/path/...`: 特定の `prompts_file` を持つ run のみを対象にできます。`RUN_LABEL` を共有する複数のループがある環境でも、誤カウントを防ぎやすくなります。
- exit code は `0=正常`, `2=カバレッジ不足 or run 不在`, `3=had_failure 検知`。監視先で原因別にアラートを分岐できる。

### 8.2 cron/launchd テンプレート
- `scripts/templates/cron/codex_history_watchdog.cron.sample`  
  codex_loop 本番ジョブの数分後に watchdog を回すサンプル。`-a` で直近数時間のみを対象にしつつ `-m/-p/-u/-F/-R` の閾値を指定し、backfill で復元した run を除外するため `-B exclude` も付与済みです。出力は `watchdog_stdout.log` へ吐きます。
- `scripts/templates/launchd/com.local.codex_history_watchdog.plist.sample`  
  launchd で 23:35 に監視する例。`ProgramArguments` 配列でオプションを順序通りに列挙し、`StandardOutPath/StandardErrorPath` を codex_loop ログ配下へ向ければ macOS のログローテーション一元化が可能。こちらも `-B exclude` で backfilled run を監視対象外にしています。

cron/launchd で監視を動かす際は、codex_loop 側の run history (`--history-log`) パスと watchdog の `-H` 引数が一致しているかを必ず確認してください。run history が溜まっていない期間は `--require-matches` を外すか、メンテナンス中に cron/launchd を一時停止することで false positive を防げます。

## 9. codex_loop + watchdog を単一ジョブで回す
定期実行を 1 本のジョブへまとめたい場合は `scripts/codex_loop_watchdog_runner.sh` を利用します。`--` を境に手前へ `scripts/codex_loop.sh` の引数、後段へ `scripts/codex_history_watchdog.sh` の引数をそのまま列挙する仕組みです。

```bash
scripts/codex_loop_watchdog_runner.sh \
  -p docment/codex_docs/codex_loop_prompts.sample.txt \
  -o results/codex_cli_runs \
  -l logs/codex_loop \
  -H logs/codex_loop/run_history.jsonl \
  -f \
  -- \
  -L nightly \
  -a 6 \
  -m 1 \
  -p 3 \
  -u 3 \
  -F \
  -R
```

- `CODEX_LOOP_BIN` / `WATCHDOG_BIN` 環境変数でそれぞれのスクリプトパスを差し替え可能。`WATCHDOG_SKIP_ON_FAILURE=1` をセットすると codex_loop が非0終了だった場合に watchdog をスキップできます（デフォルトは続行）。
- 引数をそのまま委譲するだけなので、既存の `codex_loop.sh` / `codex_history_watchdog.sh` オプションや `-n` Dry-run もそのまま活用できます。
- ログは各スクリプト側で記録されるほか、ランナー自身が `🧭` ログでコマンド開始/終了ステータスを追加します。cron/launchd で 1 エントリにまとめたい場合はこのランナーを呼び出すのが最短。

### 9.1 cron/launchd サンプル
- `scripts/templates/cron/codex_loop_with_watchdog.cron.sample`  
  23:30 に codex_loop を回し、そのまま同一エントリ内で watchdog を続けるテンプレート。watchdog 側には `-B exclude` を入れており、backfill を実施した日でも誤検知を避けつつ Exit code (codex_loop 優先→watchdog) でジョブ成否を把握できます。
- `scripts/templates/launchd/com.local.codex_loop_with_watchdog.plist.sample`  
  `ProgramArguments` にランナーを 1 本登録しておけば、macOS launchd でも同様に一括管理が可能。`EnvironmentVariables` へ `RUN_LABEL` や `WATCHDOG_SKIP_ON_FAILURE` を並べておくと設定の見通しが良くなります。watchdog の引数にも `-B exclude` を含め、backfilled run 監視を切り分ける方針を明文化しました。

## 10. 既存 manifest から run_history を復元する
`--history-log` を付け忘れて過去の run を記録できていない場合でも、`codex_loop.py` が出力した `manifest.json` さえ残っていれば履歴を再構築できます。新設した `codex_history_backfill.py` を使うと、`results/codex_cli_runs/*/manifest.json` を走査して `run_history.jsonl` を生成/更新します。

### 10.1 使いどころ
- cron/launchd への移行前に、手動実行で溜まった manifest から run history を一本化したい。
- 監査ログ (`run_history.jsonl`) を削除してしまったので再生成したい。
- 複数マシンの結果をまとめたい（`--results-dir` を切り替え or rsync したうえで backfill）。

### 10.2 基本コマンド
```bash
python codex_history_backfill.py \
  --results-dir results/codex_cli_runs \
  --history-log logs/codex_loop/run_history.jsonl
```

- 既存の `run_history.jsonl` があれば新規 manifest ぶんのみ追記（manifest の絶対パスで重複を判定）。
- `--dry-run` で「どの run ディレクトリが対象になるか」を確認可能。
- 追記先ディレクトリが無ければ自動作成されます（`logs/codex_loop/` など）。

### 10.3 全面再構築 (`--rebuild`)
履歴ファイルが壊れている／別のディレクトリへ移したい場合は `--rebuild` を追加し、検出した manifest すべてから JSONL を書き直します。

```bash
python codex_history_backfill.py \
  --results-dir results/codex_cli_runs \
  --history-log logs/codex_loop/run_history.jsonl \
  --rebuild
```

- 書き込みは一時ファイル経由で行うため、途中で中断しても旧ファイルは保持されます。
- `backfilled: true` フラグを付与しており、`codex_history_report.py` からも通常の run と同じように集計できます。
- 再構築後は `scripts/codex_history_watchdog.sh -H <same path>` で監視を開始すれば OK。

### 10.4 backfilled エントリと監視の切り分け
backfill で復元した run は `backfilled: true` が付与されます。日次の自動ジョブだけを監視したい場合は `scripts/codex_history_watchdog.sh -B exclude ...`（もしくは `codex_history_report.py --backfilled exclude ...`）を併用し、監査ログには残しつつ監視アラートの対象外にできます。逆に、backfill の進捗を確認したいときは `-B only` を指定して復元済み run のみを抽出してください。

---
DRY-RUN → 本番 → スケジュールの順で運用すれば BL-002 の完了条件である「codex CLI 非対話ループの自動起動手順」が明文化されます。疑問が出たら `docment/codex_docs/task_progress/current.md` に記載して次ループへ引き継いでください。
