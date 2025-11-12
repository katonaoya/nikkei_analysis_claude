import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[2]))

from production.backtest_scenario_runner import _build_scenarios


def test_build_scenarios_respects_max_cap():
    args = SimpleNamespace(
        position_size=200_000.0,
        position_size_grid='150000, 200000',
        max_positions=5,
        max_positions_grid='3,5',
        profit_target=0.07,
        profit_target_grid='0.05,0.07',
        stop_loss=0.05,
        stop_loss_grid='0.04',
        slippage=0.001,
        slippage_grid='0.0005,0.0010',
        transaction_cost=0.001,
        transaction_cost_grid=None,
        entry_delay=1,
        entry_delay_grid='1, 2',
        max_scenarios=4,
    )

    scenarios = _build_scenarios(args)

    assert len(scenarios) == 4
    assert scenarios[0]['position_size'] == 150000.0
    assert scenarios[0]['max_positions'] == 3
    assert scenarios[0]['entry_delay'] == 1
    assert all('profit_target' in scenario for scenario in scenarios)
