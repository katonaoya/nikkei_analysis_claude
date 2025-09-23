#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保有期間ごとの最適利確・損切りパラメータ探索スクリプト

- production_reports に出力された日次レポートを解析して、推奨上位銘柄のトレードデータを収集
- 日次株価（Nikkei225 データ）に基づいて、保有期間を 1〜20 日とした場合の利確・損切り 1〜20% 組み合わせを全探索
- 各保有期間ごとに複利リターン（全トレード連続運用時の総リターン）が最大となるパラメータを抽出

実行例:
    python analysis/holding_period_best_params.py
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.market_calendar import JapanMarketCalendar

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

MAX_HOLDING_DAYS = 20
PROFIT_RANGE = [i / 100 for i in range(1, 21)]  # 1% 〜 20%
STOP_RANGE = [i / 100 for i in range(1, 21)]    # 1% 〜 20%
MAX_WORKERS = min(mp.cpu_count(), 8)


def normalize_code(code: str) -> str:
    code = code.strip()
    if len(code) == 4:
        return f"{code}0"
    return code


def find_latest_price_file() -> Optional[Path]:
    """data/processed 配下から最新の日経225 parquet を探す"""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        return None

    candidates = sorted(
        processed_dir.glob("nikkei225_complete_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if candidates:
        return candidates[0]
    return None


def load_price_data() -> Dict[str, pd.DataFrame]:
    """最新の株価データを読み込み、銘柄ごとの DataFrame 辞書を返却"""
    price_file = find_latest_price_file()
    if price_file is None:
        raise FileNotFoundError("data/processed 内に nikkei225_complete_*.parquet が見つかりません")

    logger.info("📊 株価データ読み込み中: %s", price_file.name)
    df = pd.read_parquet(price_file, columns=["Code", "Date", "High", "Low", "Close"])
    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)

    grouped: Dict[str, pd.DataFrame] = {}
    for code, group in df.groupby("Code"):
        grouped[code] = group.set_index("Date")

    latest_date = df["Date"].max()
    earliest_date = df["Date"].min()
    logger.info("✅ 株価データ期間: %s 〜 %s (銘柄数: %d)", earliest_date.date(), latest_date.date(), len(grouped))
    return grouped


def extract_trade_blocks(content: str) -> List[Dict[str, Optional[float]]]:
    """レポート本文から推奨銘柄ブロックを抽出"""
    trades: List[Dict[str, Optional[float]]] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match_primary = re.match(r"(\d+)位:\s*(.+?)\s*\((\d{4,5})\)", line)
        match_alt = re.match(r"###\s*(\d+)\.\s*(.+)", line)

        if match_primary:
            rank = int(match_primary.group(1))
            company = match_primary.group(2).strip()
            code = normalize_code(match_primary.group(3))

            price_value: Optional[float] = None
            probability_value: Optional[float] = None
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                price_match = re.search(r"現在価格:\s*¥([\d,]+)", lines[j])
                prob_match = re.search(r"予測確率:\s*([\d.]+)%", lines[j])
                if price_match:
                    price_value = float(price_match.group(1).replace(",", ""))
                if prob_match:
                    probability_value = float(prob_match.group(1))
                j += 1

            if price_value is not None:
                trades.append({
                    "rank": rank,
                    "company_name": company,
                    "code": code,
                    "entry_price": price_value,
                    "probability": probability_value
                })
            i = j
            continue

        if match_alt:
            rank = int(match_alt.group(1))
            company = match_alt.group(2).strip()
            code_value: Optional[str] = None
            price_value = None
            probability_value = None
            j = i + 1
            while j < len(lines) and lines[j].startswith("-"):
                code_match = re.search(r"銘柄コード[^:]*:\s*(\d+)", lines[j])
                price_match = re.search(r"現在価格[^:]*:\s*([\d,.]+)円", lines[j])
                prob_match = re.search(r"予測上昇確率[^:]*:\s*([\d.]+)%", lines[j])
                if code_match:
                    code_value = normalize_code(code_match.group(1))
                if price_match:
                    price_value = float(price_match.group(1).replace(",", ""))
                if prob_match:
                    probability_value = float(prob_match.group(1))
                j += 1

            if code_value and price_value is not None:
                trades.append({
                    "rank": rank,
                    "company_name": company,
                    "code": code_value,
                    "entry_price": price_value,
                    "probability": probability_value
                })
            i = j
            continue

        i += 1
    return trades


def parse_report_file(path: Path) -> Optional[Dict]:
    """レポートファイルからトレード候補を抽出"""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("⚠️ レポート読み込み失敗 (%s): %s", path.name, exc)
        return None

    trade_date_match = re.search(r"推奨取引日:\s*(\d{4}-\d{2}-\d{2})", content)
    if trade_date_match:
        trade_date = datetime.strptime(trade_date_match.group(1), "%Y-%m-%d")
    else:
        base_match = re.search(r"(基準日付|対象日):\s*(\d{4}-\d{2}-\d{2})", content)
        if not base_match:
            logger.debug("基準日付/推奨取引日が見つからないためスキップ: %s", path)
            return None
        base_date = datetime.strptime(base_match.group(2), "%Y-%m-%d")
        trade_date_dt = JapanMarketCalendar.get_next_market_day(base_date)
        trade_date = datetime(trade_date_dt.year, trade_date_dt.month, trade_date_dt.day)

    trades = extract_trade_blocks(content)
    if not trades:
        logger.debug("銘柄ブロックが見つからないためスキップ: %s", path)
        return None

    trades_sorted = sorted(trades, key=lambda x: x.get("rank", 999))
    top_entries = trades_sorted[:5]

    for trade in top_entries:
        trade["entry_date"] = trade_date

    return {
        "path": path,
        "trade_date": trade_date,
        "entries": top_entries
    }


def collect_trades() -> List[Dict]:
    """production_reports から有効なトレード候補を収集"""
    report_dir = Path("production_reports")
    if not report_dir.exists():
        raise FileNotFoundError("production_reports ディレクトリが見つかりません")

    trade_entries: List[Dict] = []
    files = sorted(report_dir.rglob("*.md"))
    logger.info("📄 レポート候補: %d件", len(files))
    for path in files:
        report = parse_report_file(path)
        if not report:
            continue
        for entry in report["entries"]:
            trade_entries.append({
                "code": entry["code"],
                "company_name": entry["company_name"],
                "entry_price": entry["entry_price"],
                "entry_date": entry["entry_date"],
                "probability": entry.get("probability")
            })
    logger.info("✅ 抽出トレード数: %d", len(trade_entries))
    return trade_entries


@dataclass
class PreparedTrade:
    code: str
    company_name: str
    entry_price: float
    probability: Optional[float]
    series: np.ndarray  # shape: (MAX_HOLDING_DAYS, 3) -> columns [High, Low, Close]


def prepare_trades(raw_trades: Iterable[Dict], price_data: Dict[str, pd.DataFrame]) -> List[PreparedTrade]:
    """株価データを参照し、保有20日分の価格系列が取得できるトレードのみ残す"""
    prepared: List[PreparedTrade] = []
    dropped = 0
    max_buffer_days = MAX_HOLDING_DAYS * 3  # 休日を考慮したバッファ

    latest_available_date = max(df.index.max() for df in price_data.values())

    for entry in raw_trades:
        code = entry["code"]
        entry_date = entry["entry_date"]
        if entry_date > latest_available_date:
            dropped += 1
            continue

        price_df = price_data.get(code)
        if price_df is None:
            dropped += 1
            continue

        start_date = entry_date
        end_date = entry_date + timedelta(days=max_buffer_days)
        window = price_df.loc[(price_df.index >= start_date) & (price_df.index <= end_date)]

        if len(window) < MAX_HOLDING_DAYS:
            dropped += 1
            continue

        sliced = window.iloc[:MAX_HOLDING_DAYS][["High", "Low", "Close"]].to_numpy(dtype=float)
        prepared.append(PreparedTrade(
            code=code,
            company_name=entry["company_name"],
            entry_price=float(entry["entry_price"]),
            probability=entry.get("probability"),
            series=sliced
        ))

    logger.info("✅ シミュレーション対象トレード: %d件 (除外: %d件)", len(prepared), dropped)
    return prepared


# --- 並列実行用のグローバル変数 ---
GLOBAL_TRADES: List[PreparedTrade] = []


def init_worker(trades: List[PreparedTrade]):
    global GLOBAL_TRADES
    GLOBAL_TRADES = trades


def simulate_combo(params: tuple) -> Dict:
    hold_days, profit_target, stop_loss = params
    trades = GLOBAL_TRADES

    returns: List[float] = []
    profit_hits = 0
    stop_hits = 0
    expiry_hits = 0

    for trade in trades:
        entry_price = trade.entry_price
        profit_price = entry_price * (1 + profit_target)
        stop_price = entry_price * (1 - stop_loss)

        exit_return = 0.0
        for day in range(hold_days):
            high, low, close = trade.series[day]

            if high >= profit_price:
                exit_return = profit_target
                profit_hits += 1
                break
            if low <= stop_price:
                exit_return = -stop_loss
                stop_hits += 1
                break
            if day == hold_days - 1:
                exit_return = (close - entry_price) / entry_price
                expiry_hits += 1
        returns.append(exit_return)

    returns_arr = np.array(returns, dtype=float)
    compound_return = float(np.prod(1 + returns_arr) - 1)
    total_return = float(returns_arr.sum())
    avg_return = float(returns_arr.mean()) if len(returns_arr) > 0 else 0.0
    win_rate = float((returns_arr > 0).mean()) if len(returns_arr) > 0 else 0.0

    trade_count = len(returns_arr)
    profit_rate = profit_hits / trade_count if trade_count else 0.0
    stop_rate = stop_hits / trade_count if trade_count else 0.0
    expiry_rate = expiry_hits / trade_count if trade_count else 0.0

    return {
        "holding_days": hold_days,
        "profit_target": profit_target,
        "stop_loss": stop_loss,
        "compound_return": compound_return,
        "total_return": total_return,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "profit_trigger_rate": profit_rate,
        "stop_trigger_rate": stop_rate,
        "expiry_trigger_rate": expiry_rate,
        "trade_count": trade_count
    }


def evaluate_holding_day(hold_days: int, trades: List[PreparedTrade]) -> Dict:
    combos = [
        (hold_days, profit, stop)
        for profit in PROFIT_RANGE
        for stop in STOP_RANGE
        if profit > stop
    ]

    if not combos:
        raise ValueError("パラメータ組み合わせが生成できませんでした")

    logger.info("⚙️ 保有%2dd日: %dパターン検証", hold_days, len(combos))

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(trades,)) as executor:
        for result in executor.map(simulate_combo, combos, chunksize=10):
            results.append(result)

    best = max(results, key=lambda r: r["compound_return"])
    logger.info(
        "🏆 保有%2dd日 最良: 利確%.0f%% / 損切%.0f%% | 複利%.2f%% | 勝率%.1f%%",
        hold_days,
        best["profit_target"] * 100,
        best["stop_loss"] * 100,
        best["compound_return"] * 100,
        best["win_rate"] * 100,
    )
    return best


def main():
    price_data = load_price_data()
    raw_trades = collect_trades()
    prepared_trades = prepare_trades(raw_trades, price_data)

    if not prepared_trades:
        logger.error("シミュレーション対象のトレードがありません")
        return

    logger.info("🚀 保有期間別の最適パラメータ探索を開始")

    results: List[Dict] = []
    for hold_days in range(1, MAX_HOLDING_DAYS + 1):
        best = evaluate_holding_day(hold_days, prepared_trades)
        results.append(best)

    df = pd.DataFrame(results)
    output_dir = Path("profit_loss_optimization_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"holding_period_best_params_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    print("\n📊 保有期間別 最適パラメータ一覧")
    for row in results:
        print(
            "保有{:2d}日 | 利確{:>5.1f}% | 損切{:>5.1f}% | 複利{:>6.2f}% | 勝率{:>5.1f}% | 平均リターン{:>6.3f}".format(
                int(row["holding_days"]),
                row["profit_target"] * 100,
                row["stop_loss"] * 100,
                row["compound_return"] * 100,
                row["win_rate"] * 100,
                row["avg_return"],
            )
        )

    print(f"\n✅ 結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()
