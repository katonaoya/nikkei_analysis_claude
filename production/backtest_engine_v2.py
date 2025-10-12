#!/usr/bin/env python3
"""Portfolio-level backtest engine with realistic fills and metrics."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.market_calendar import JapanMarketCalendar
from utils.logger import StructuredLogger


@dataclass
class TradeSignal:
    recommendation_date: pd.Timestamp
    code: str
    company: str
    entry_price: float
    probability: float


@dataclass
class Position:
    code: str
    company: str
    entry_date: pd.Timestamp
    entry_price_exec: float
    base_entry_price: float
    quantity: int
    target_price: float
    stop_price: float
    max_hold_days: int
    open_fees: float
    status: str = "open"
    notes: str = ""

    def mark_to_market(self, price: float) -> float:
        return price * self.quantity


@dataclass
class TradeResult:
    code: str
    company: str
    recommendation_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    gross_profit: float
    net_profit: float
    return_pct: float
    holding_days: int
    status: str
    notes: str
    fees: float


class BacktestEngine:
    def __init__(
        self,
        price_df: pd.DataFrame,
        signals: List[TradeSignal],
        initial_capital: float = 1_000_000.0,
        position_size: float = 200_000.0,
        max_positions: int = 5,
        profit_target: float = 0.07,
        stop_loss: float = 0.05,
        max_holding_days: int = 10,
        slippage: float = 0.001,
        transaction_cost: float = 0.001,
        entry_delay_days: int = 1,
        benchmark_code: Optional[str] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.logger = logger or StructuredLogger("backtest_engine_v2")
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days
        self.slippage = slippage
        self.transaction_cost = transaction_cost
        self.benchmark_code = benchmark_code
        if entry_delay_days < 0:
            raise ValueError("entry_delay_days must be >= 0")
        self.entry_delay_days = entry_delay_days

        self.price_lookup = self._prepare_price_lookup(price_df)
        self.signals = sorted(signals, key=lambda s: s.recommendation_date)

    @staticmethod
    def _prepare_price_lookup(price_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df = price_df.copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['Code'] = df['Code'].astype(str)
        required_cols = {'Open', 'High', 'Low', 'Close'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Price data missing columns: {missing_cols}")

        lookup: Dict[str, pd.DataFrame] = {}
        for code, group in df.groupby('Code'):
            lookup[code] = group.sort_values('Date').set_index('Date')
        if not lookup:
            raise ValueError("Price data is empty")
        return lookup

    def _get_price_row(self, code: str, date: pd.Timestamp) -> Optional[pd.Series]:
        table = self.price_lookup.get(code)
        if table is None:
            return None
        try:
            return table.loc[date]
        except KeyError:
            return None

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        if is_buy:
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)

    def _transaction_fee(self, amount: float) -> float:
        return amount * self.transaction_cost

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        positions: List[Position] = []
        trades: List[TradeResult] = []
        cash = self.initial_capital
        equity_records: List[Dict[str, float]] = []

        signal_iter = iter(self.signals)
        pending_signals: List[TradeSignal] = []

        # Pre-compute trading days in price data range
        trading_dates = sorted({date for df in self.price_lookup.values() for date in df.index})
        if not trading_dates:
            raise ValueError("No trading dates available in price data")

        current_signal = next(signal_iter, None)
        for current_date in trading_dates:
            # Add signals whose recommendation date has passed and entry day is today
            while current_signal and self._next_entry_day(current_signal.recommendation_date) <= current_date:
                pending_signals.append(current_signal)
                current_signal = next(signal_iter, None)

            # Exit logic for open positions
            to_remove: List[Position] = []
            for pos in positions:
                price_row = self._get_price_row(pos.code, current_date)
                if price_row is None:
                    continue
                exit_reason = None
                exit_price_exec = None

                holding_days = (current_date - pos.entry_date).days
                high = price_row['High']
                low = price_row['Low']
                close_price = price_row['Close']

                if current_date == pos.entry_date:
                    # Evaluate intraday target/stop on entry day after open fill
                    high_hit = high >= pos.target_price
                    low_hit = low <= pos.stop_price
                else:
                    high_hit = high >= pos.target_price
                    low_hit = low <= pos.stop_price

                if high_hit and low_hit:
                    # Ambiguous scenario: assume worst case (stop hit first)
                    exit_reason = 'hit_stop_and_target_same_day'
                    exit_base_price = pos.stop_price
                    exit_price_exec = self._apply_slippage(pos.stop_price, is_buy=False)
                    pos.notes = 'Stop prioritized over target on same day'
                elif low_hit:
                    exit_reason = 'hit_stop'
                    exit_base_price = pos.stop_price
                    exit_price_exec = self._apply_slippage(pos.stop_price, is_buy=False)
                elif high_hit:
                    exit_reason = 'hit_target'
                    exit_base_price = pos.target_price
                    exit_price_exec = self._apply_slippage(pos.target_price, is_buy=False)
                elif holding_days >= pos.max_hold_days:
                    exit_reason = 'max_holding'
                    exit_price_exec = self._apply_slippage(close_price, is_buy=False)
                else:
                    continue

                exit_fees = self._transaction_fee(exit_price_exec * pos.quantity)
                gross = (exit_price_exec - pos.entry_price_exec) * pos.quantity
                net = gross - pos.open_fees - exit_fees
                return_pct = (exit_price_exec - pos.entry_price_exec) / pos.entry_price_exec if pos.entry_price_exec else 0.0

                trades.append(
                    TradeResult(
                        code=pos.code,
                        company=pos.company,
                        recommendation_date=self._find_signal_date(pos.code, pos.entry_date),
                        entry_date=pos.entry_date,
                        exit_date=current_date,
                        entry_price=pos.entry_price_exec,
                        exit_price=exit_price_exec,
                        quantity=pos.quantity,
                        gross_profit=gross,
                        net_profit=net,
                        return_pct=return_pct,
                        holding_days=holding_days if holding_days > 0 else 0,
                        status=exit_reason or 'unknown',
                        notes=pos.notes,
                        fees=pos.open_fees + exit_fees,
                    )
                )

                cash += exit_price_exec * pos.quantity - exit_fees
                to_remove.append(pos)

            for pos in to_remove:
                positions.remove(pos)

            # Attempt to open new positions
            new_pending: List[TradeSignal] = []
            for signal in pending_signals:
                entry_day = self._next_entry_day(signal.recommendation_date)
                if entry_day != current_date:
                    new_pending.append(signal)
                    continue

                if len(positions) >= self.max_positions:
                    new_pending.append(signal)
                    continue

                price_row = self._get_price_row(signal.code, current_date)
                if price_row is None:
                    continue

                base_open = price_row['Open']
                entry_exec = self._apply_slippage(base_open, is_buy=True)
                quantity = int(self.position_size // entry_exec)
                if quantity <= 0:
                    continue
                required_cash = entry_exec * quantity + self._transaction_fee(entry_exec * quantity)
                if cash < required_cash:
                    new_pending.append(signal)
                    continue

                target_price = base_open * (1 + self.profit_target)
                stop_price = base_open * (1 - self.stop_loss)

                cash -= required_cash
                positions.append(
                    Position(
                        code=signal.code,
                        company=signal.company,
                        entry_date=current_date,
                        entry_price_exec=entry_exec,
                        base_entry_price=base_open,
                        quantity=quantity,
                        target_price=target_price,
                        stop_price=stop_price,
                        max_hold_days=self.max_holding_days,
                        open_fees=self._transaction_fee(entry_exec * quantity),
                    )
                )
            pending_signals = new_pending

            # record daily equity
            exposure = 0.0
            for pos in positions:
                price_row = self._get_price_row(pos.code, current_date)
                mark_price = price_row['Close'] if price_row is not None else pos.entry_price_exec
                exposure += mark_price * pos.quantity
            equity = cash + exposure
            equity_records.append({
                'Date': current_date,
                'Equity': equity,
                'Cash': cash,
                'Exposure': exposure,
                'OpenPositions': len(positions),
            })

        equity_df = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
        metrics = self._calculate_metrics(equity_df, trades_df)
        return equity_df, trades_df, metrics

    def _find_signal_date(self, code: str, entry_date: pd.Timestamp) -> pd.Timestamp:
        for signal in self.signals:
            entry_day = self._next_entry_day(signal.recommendation_date)
            if signal.code == code and entry_day == entry_date:
                return signal.recommendation_date
        return entry_date

    def _next_entry_day(self, recommendation_date: pd.Timestamp) -> pd.Timestamp:
        if isinstance(recommendation_date, pd.Timestamp):
            base_date = recommendation_date.to_pydatetime()
        else:
            base_date = recommendation_date

        target_date = pd.Timestamp(base_date).tz_localize(None)
        if self.entry_delay_days == 0:
            if not JapanMarketCalendar.is_market_open(target_date):
                next_day = JapanMarketCalendar.get_next_market_day(target_date)
                target_date = pd.Timestamp(next_day).tz_localize(None)
            return target_date

        current_date = target_date
        for _ in range(self.entry_delay_days):
            current_date = JapanMarketCalendar.get_next_market_day(current_date)
        return pd.Timestamp(current_date).tz_localize(None)

    def _calculate_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, float]:
        equity_df = equity_df.sort_values('Date')
        equity_df['Return'] = equity_df['Equity'].pct_change().fillna(0)
        total_return = (equity_df['Equity'].iloc[-1] / equity_df['Equity'].iloc[0]) - 1
        trading_days = len(equity_df)
        cagr = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        sharpe = 0.0
        if equity_df['Return'].std() > 0:
            sharpe = (equity_df['Return'].mean() / equity_df['Return'].std()) * np.sqrt(252)

        running_max = equity_df['Equity'].cummax()
        drawdown = equity_df['Equity'] / running_max - 1
        max_drawdown = drawdown.min()

        metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': float(equity_df['Equity'].iloc[-1]),
            'total_return': float(total_return),
            'cagr': float(cagr),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'num_trades': int(len(trades_df)),
            'win_rate': float((trades_df['net_profit'] > 0).mean()) if not trades_df.empty else 0.0,
        }

        if not trades_df.empty:
            metrics['avg_trade_return'] = float(trades_df['return_pct'].mean())
            metrics['median_trade_return'] = float(trades_df['return_pct'].median())
            metrics['avg_holding_days'] = float(trades_df['holding_days'].mean())
        else:
            metrics['avg_trade_return'] = 0.0
            metrics['median_trade_return'] = 0.0
            metrics['avg_holding_days'] = 0.0

        if self.benchmark_code and self.benchmark_code in self.price_lookup:
            bench_df = self.price_lookup[self.benchmark_code]
            bench_series = bench_df['Close'].reindex(equity_df['Date']).fillna(method='ffill')
            bench_returns = bench_series.pct_change().fillna(0)
            bench_total = (bench_series.iloc[-1] / bench_series.iloc[0]) - 1 if bench_series.iloc[0] else 0
            metrics['benchmark_total_return'] = float(bench_total)
            if bench_returns.std() > 0:
                metrics['benchmark_sharpe'] = float((bench_returns.mean() / bench_returns.std()) * np.sqrt(252))
        return metrics


def parse_reports(report_dir: Path) -> List[TradeSignal]:
    signals: List[TradeSignal] = []
    for path in sorted(report_dir.glob('*.md')):
        content = path.read_text(encoding='utf-8')
        date_match = Path(path).stem
        try:
            report_date = pd.Timestamp(date_match)
        except ValueError:
            continue

        seen_keys = set()

        def add_signal(code: str, company: str, price: float, probability: float) -> None:
            key = (code, round(price, 4), round(probability, 4))
            if key in seen_keys:
                return
            seen_keys.add(key)
            signals.append(
                TradeSignal(
                    recommendation_date=report_date,
                    code=code,
                    company=company.strip(),
                    entry_price=price,
                    probability=probability,
                )
            )

        pattern = r"(\d+)位:\s*(.+?)\s*\((\d{4,5})\)[\s\S]*?現在価格:\s*¥([\d,]+)[\s\S]*?🎯 予測確率:\s*([\d.]+)%"
        for _, company, code, price_str, prob_str in re.findall(pattern, content):
            code = code.strip()
            if len(code) == 4:
                code = f"{code}0"
            price = float(price_str.replace(',', ''))
            probability = float(prob_str)
            add_signal(code, company, price, probability)

        # Pattern for markdown sections like "### 1. Company" with bullet details
        section_pattern = (
            r"###\s*\d+\.\s*(?P<company>.+?)\n"
            r"- \*\*銘柄コード\*\*:\s*(?P<code>\d{4,5})\n"
            r"- \*\*現在価格\*\*:\s*(?P<price>[\d,.]+)円\n"
            r"- \*\*予測上昇確率\*\*:\s*(?P<prob>[\d.]+)%"
        )
        for match in re.finditer(section_pattern, content):
            data = match.groupdict()
            code = data['code']
            if len(code) == 4:
                code = f"{code}0"
            price = float(data['price'].replace(',', ''))
            probability = float(data['prob'])
            add_signal(code, data['company'], price, probability)

        # Pattern for markdown tables (| rank | code | ... | probability% |)
        table_pattern = (
            r"\|\s*(?P<rank>\d+)\s*\|\s*(?P<company>[^|]+?)\s*\|\s*(?P<code>\d{4,5})\s*\|"
            r"\s*(?P<price>[\d,.]+)円\s*\|\s*(?P<prob>[\d.]+)%"
        )
        for match in re.finditer(table_pattern, content):
            rank = int(match.group('rank'))
            if rank > 10:
                continue
            code = match.group('code').strip()
            if len(code) == 4:
                code = f"{code}0"
            price = float(match.group('price').replace(',', ''))
            probability = float(match.group('prob'))
            add_signal(code, match.group('company'), price, probability)
    return signals


def compute_monthly_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty or 'Equity' not in equity_df or 'Date' not in equity_df:
        return pd.DataFrame(columns=['Year', 'Month', 'Return'])

    monthly_df = equity_df.copy()
    monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
    monthly_df = monthly_df.sort_values('Date').set_index('Date')
    monthly_equity = monthly_df['Equity'].resample('M').last()
    monthly_returns = monthly_equity.pct_change().dropna()
    if monthly_returns.empty:
        return pd.DataFrame(columns=['Year', 'Month', 'Return'])

    result = monthly_returns.to_frame(name='Return').reset_index()
    result['Year'] = result['Date'].dt.year
    result['Month'] = result['Date'].dt.month
    return result[['Year', 'Month', 'Return']]


def plot_equity_curve(equity_df: pd.DataFrame, output_path: Path) -> None:
    if equity_df.empty:
        return

    plot_df = equity_df.copy()
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    plot_df = plot_df.sort_values('Date')
    if plot_df.empty:
        return

    running_max = plot_df['Equity'].cummax()
    drawdown = (plot_df['Equity'] / running_max) - 1
    drawdown = drawdown.fillna(0)

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(plot_df['Date'], plot_df['Equity'], color='tab:blue', label='Equity')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.fill_between(plot_df['Date'], drawdown, 0, color='tab:red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown', color='tab:red')
    ax2.set_ylim(-1.0, 0.0)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: Dict[str, float],
    *,
    logger: Optional[StructuredLogger] = None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    equity_path = output_dir / f'daily_equity_{timestamp}.csv'
    trades_path = output_dir / f'trades_{timestamp}.csv'
    summary_path = output_dir / f'summary_{timestamp}.json'

    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    paths: Dict[str, Path] = {
        'equity': equity_path,
        'trades': trades_path,
        'summary': summary_path,
    }

    monthly_returns = compute_monthly_returns(equity_df)
    if not monthly_returns.empty:
        monthly_path = output_dir / f'monthly_returns_{timestamp}.csv'
        monthly_returns.to_csv(monthly_path, index=False)
        paths['monthly_returns'] = monthly_path

    plot_path = output_dir / f'equity_curve_{timestamp}.png'
    try:
        plot_equity_curve(equity_df, plot_path)
        paths['equity_curve_plot'] = plot_path
    except Exception as exc:  # pragma: no cover - plotting should not break run
        if logger:
            logger.warning('⚠️ Failed to create equity curve plot', error=str(exc))

    if logger:
        logger.info(
            '📦 Backtest artifacts saved',
            **{f'path_{key}': str(value) for key, value in paths.items()}
        )

    return paths


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Backtest Engine V2 with realistic fills')
    parser.add_argument('--report-dir', required=True, type=Path, help='Directory with markdown recommendation reports')
    parser.add_argument('--price-file', required=True, type=Path, help='Parquet file with price data')
    parser.add_argument('--output-dir', default=Path('results/backtest_v2'), type=Path, help='Output directory')
    parser.add_argument('--initial-capital', type=float, default=1_000_000.0)
    parser.add_argument('--position-size', type=float, default=200_000.0)
    parser.add_argument('--max-positions', type=int, default=5)
    parser.add_argument('--profit-target', type=float, default=0.07)
    parser.add_argument('--stop-loss', type=float, default=0.05)
    parser.add_argument('--max-holding-days', type=int, default=10)
    parser.add_argument('--slippage', type=float, default=0.001)
    parser.add_argument('--transaction-cost', type=float, default=0.001)
    parser.add_argument('--entry-delay-days', type=int, default=1, help='Number of market days to wait before entering positions')
    parser.add_argument('--benchmark-code', type=str, default=None)
    return parser


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()

    logger = StructuredLogger('backtest_engine_v2')
    if not args.report_dir.exists():
        logger.error(f"Report directory not found: {args.report_dir}")
        return 2
    if not args.price_file.exists():
        logger.error(f"Price file not found: {args.price_file}")
        return 2

    price_df = pd.read_parquet(args.price_file)
    signals = parse_reports(args.report_dir)
    if not signals:
        logger.warning('No trade signals found in reports')
        return 1

    engine = BacktestEngine(
        price_df=price_df,
        signals=signals,
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        max_positions=args.max_positions,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss,
        max_holding_days=args.max_holding_days,
        slippage=args.slippage,
        transaction_cost=args.transaction_cost,
        entry_delay_days=args.entry_delay_days,
        benchmark_code=args.benchmark_code,
        logger=logger,
    )

    try:
        equity_df, trades_df, metrics = engine.run()
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")
        return 1

    save_outputs(args.output_dir, equity_df, trades_df, metrics, logger=logger)
    logger.info('Backtest completed', **metrics)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
