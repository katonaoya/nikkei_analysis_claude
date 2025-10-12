#!/usr/bin/env python3
"""Scenario sweep utility for the portfolio backtest engine."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from production.backtest_engine_v2 import (
    BacktestEngine,
    parse_reports,
    save_outputs,
)
from utils.logger import StructuredLogger


def _parse_grid(raw: Optional[str], cast) -> Optional[List]:
    if not raw:
        return None
    values: List = []
    for token in raw.split(','):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            values.append(cast(stripped))
        except ValueError as exc:  # pragma: no cover - argparse guards usage
            raise argparse.ArgumentTypeError(f"Invalid grid value: {token}") from exc
    return values or None


def _format_float(value: float) -> str:
    return format(value, 'g').replace('.', 'p').replace('-', 'm')


def _scenario_slug(params: Dict[str, float]) -> str:
    return (
        f"pt{_format_float(params['profit_target'])}_"
        f"sl{_format_float(params['stop_loss'])}_"
        f"slip{_format_float(params['slippage'])}_"
        f"tc{_format_float(params['transaction_cost'])}_"
        f"delay{params['entry_delay']}_"
        f"ps{_format_float(params['position_size'])}_"
        f"mp{params['max_positions']}"
    )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run parameter sweeps for BacktestEngine scenarios.'
    )
    parser.add_argument('--report-dir', type=Path, required=True, help='Directory containing markdown recommendation reports')
    parser.add_argument('--price-file', type=Path, required=True, help='Parquet price file aligned with the recommendation universe')
    parser.add_argument('--output-dir', type=Path, default=Path('results/backtest_v2'), help='Base output directory for scenario artifacts')

    parser.add_argument('--initial-capital', type=float, default=1_000_000.0)
    parser.add_argument('--position-size', type=float, default=200_000.0)
    parser.add_argument('--position-size-grid', type=str, help='Comma separated list of position sizes to evaluate (e.g. 150000,200000)')
    parser.add_argument('--max-positions', type=int, default=5)
    parser.add_argument('--max-positions-grid', type=str, help='Comma separated list of max position counts')
    parser.add_argument('--profit-target', type=float, default=0.07)
    parser.add_argument('--profit-target-grid', type=str, help='Comma separated profit targets (fractional, e.g. 0.05,0.07)')
    parser.add_argument('--stop-loss', type=float, default=0.05)
    parser.add_argument('--stop-loss-grid', type=str, help='Comma separated stop losses (fractional)')
    parser.add_argument('--slippage', type=float, default=0.001)
    parser.add_argument('--slippage-grid', type=str, help='Comma separated slippage rates (fractional)')
    parser.add_argument('--transaction-cost', type=float, default=0.001)
    parser.add_argument('--transaction-cost-grid', type=str, help='Comma separated transaction cost rates (fractional)')
    parser.add_argument('--entry-delay', type=int, default=1, help='Base entry delay in market days (0 = same day)')
    parser.add_argument('--entry-delay-grid', type=str, help='Comma separated list of entry delays (integers)')
    parser.add_argument('--max-holding-days', type=int, default=10)
    parser.add_argument('--benchmark-code', type=str, default=None)
    parser.add_argument('--max-scenarios', type=int, default=None, help='Optional cap on the number of scenario combinations (evaluated in lexical order)')
    parser.add_argument('--save-best-details', action='store_true', help='Persist full equity/trade outputs for the best-performing scenario')
    parser.add_argument('--export-json', action='store_true', help='Export scenario metrics alongside CSV')
    return parser


def _build_scenarios(args: argparse.Namespace) -> List[Dict[str, float]]:
    position_sizes = _parse_grid(args.position_size_grid, float) or [args.position_size]
    max_positions = _parse_grid(args.max_positions_grid, int) or [args.max_positions]
    profit_targets = _parse_grid(args.profit_target_grid, float) or [args.profit_target]
    stop_losses = _parse_grid(args.stop_loss_grid, float) or [args.stop_loss]
    slippages = _parse_grid(args.slippage_grid, float) or [args.slippage]
    transaction_costs = _parse_grid(args.transaction_cost_grid, float) or [args.transaction_cost]
    entry_delays = _parse_grid(args.entry_delay_grid, int) or [args.entry_delay]

    scenario_iter: Iterable = itertools.product(
        position_sizes,
        max_positions,
        profit_targets,
        stop_losses,
        slippages,
        transaction_costs,
        entry_delays,
    )

    scenarios: List[Dict[str, float]] = []
    for combo in scenario_iter:
        scenario = {
            'position_size': float(combo[0]),
            'max_positions': int(combo[1]),
            'profit_target': float(combo[2]),
            'stop_loss': float(combo[3]),
            'slippage': float(combo[4]),
            'transaction_cost': float(combo[5]),
            'entry_delay': int(combo[6]),
        }
        scenarios.append(scenario)
        if args.max_scenarios and len(scenarios) >= args.max_scenarios:
            break

    return scenarios


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()

    logger = StructuredLogger('backtest_scenario_runner')

    if not args.report_dir.exists():
        logger.error('❌ Report directory not found', path=str(args.report_dir))
        return 2
    if not args.price_file.exists():
        logger.error('❌ Price file not found', path=str(args.price_file))
        return 2

    signals = parse_reports(args.report_dir)
    if not signals:
        logger.warning('⚠️ No signals parsed from reports', report_dir=str(args.report_dir))
        return 1

    price_df = pd.read_parquet(args.price_file)

    scenarios = _build_scenarios(args)
    if not scenarios:
        logger.error('❌ No scenarios generated from provided parameters')
        return 2

    scenario_dir = args.output_dir / 'scenarios'
    scenario_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, float]] = []
    best_result: Optional[Dict] = None
    total = len(scenarios)

    for idx, params in enumerate(scenarios, start=1):
        scenario_id = _scenario_slug(params)
        logger.info(
            '🚀 Running scenario',
            scenario=scenario_id,
            index=idx,
            total=total,
            **params,
        )

        engine = BacktestEngine(
            price_df=price_df,
            signals=signals,
            initial_capital=args.initial_capital,
            position_size=params['position_size'],
            max_positions=params['max_positions'],
            profit_target=params['profit_target'],
            stop_loss=params['stop_loss'],
            max_holding_days=args.max_holding_days,
            slippage=params['slippage'],
            transaction_cost=params['transaction_cost'],
            entry_delay_days=params['entry_delay'],
            benchmark_code=args.benchmark_code,
            logger=logger,
        )

        try:
            equity_df, trades_df, metrics = engine.run()
        except Exception as exc:  # pragma: no cover - defensive guard for unexpected issues
            logger.error('💥 Scenario execution failed', scenario=scenario_id, error=str(exc))
            continue

        scenario_metrics = metrics.copy()
        scenario_metrics.update(
            {
                'scenario_id': scenario_id,
                'position_size': params['position_size'],
                'max_positions': params['max_positions'],
                'profit_target': params['profit_target'],
                'stop_loss': params['stop_loss'],
                'slippage': params['slippage'],
                'transaction_cost': params['transaction_cost'],
                'entry_delay_days': params['entry_delay'],
            }
        )
        metrics_rows.append(scenario_metrics)

        if not best_result or scenario_metrics['final_equity'] > best_result['metrics']['final_equity']:
            best_result = {
                'scenario_id': scenario_id,
                'metrics': scenario_metrics,
                'equity_df': equity_df,
                'trades_df': trades_df,
            }

    if not metrics_rows:
        logger.error('❌ All scenarios failed to execute successfully')
        return 1

    metrics_df = pd.DataFrame(metrics_rows).sort_values('final_equity', ascending=False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = scenario_dir / f'scenario_metrics_{timestamp}.csv'
    metrics_df.to_csv(metrics_path, index=False)

    artifacts = {'metrics_csv': metrics_path}

    if args.export_json:
        json_path = scenario_dir / f'scenario_metrics_{timestamp}.json'
        metrics_df.to_json(json_path, orient='records', indent=2)
        artifacts['metrics_json'] = json_path

    if best_result:
        best_summary_path = scenario_dir / f'best_scenario_{timestamp}.json'
        with best_summary_path.open('w', encoding='utf-8') as f:
            json.dump(best_result['metrics'], f, ensure_ascii=False, indent=2)
        artifacts['best_summary'] = best_summary_path

        if args.save_best_details:
            best_dir = scenario_dir / best_result['scenario_id']
            paths = save_outputs(
                best_dir,
                best_result['equity_df'],
                best_result['trades_df'],
                best_result['metrics'],
                logger=logger,
            )
            artifacts.update({f'best_{key}': value for key, value in paths.items()})

    logger.info(
        '🏁 Scenario sweep completed',
        scenarios_executed=len(metrics_rows),
        best_scenario=best_result['scenario_id'] if best_result else None,
        **{f'path_{key}': str(val) for key, val in artifacts.items()},
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

