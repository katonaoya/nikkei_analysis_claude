# Repository Guidelines

## Project Structure & Module Organization
- `daily_trading_automation.py` is the daily entrypoint, chaining data fetch, integration, model refresh, and report generation.
- Model code sits in `systems/`; `enhanced_close_return_system_v1.py` handles the close-to-close pipeline, while `enhanced_precision_system_v3.py` powers the multi-signal variant. Utilities live under `src/` and `utils/`.
- Data collection and enrichment scripts reside in `data_management/`, writing artifacts to `data/` (raw/processed) and mirroring live-ready outputs into `production_data/`.
- Analytics notebooks and scripts are under `analysis/`; production-facing reports land in `production_reports/`, with exploratory results in `reports/`, `results/`, and `validation_results/`. Operations SOPs remain in `docment/` and `documentation/`.
- Tests live in `tests/` and mirror module structure; test assets and fixtures belong inside `tests/data/`.

## Build, Test, and Development Commands
- `python daily_trading_automation.py` — runs the full daily workflow (requires `.env` with J-Quants credentials and current market calendar).
- `python systems/enhanced_close_return_system_v1.py --target-return 0.01 --imbalance-boost 1.2` — retrains the close-return model, executes walk-forward validation, and persists artefacts in `models/enhanced_close_v1/`.
- `python analysis/close_threshold_optimizer.py --transaction-cost 0.002 --target-returns 0.008,0.010 --export-csv reports/monitoring/threshold_scan.csv` — recomputes最適閾値を複数ターゲットで比較し、設定ファイルへ反映。
- `python analysis/close_optuna_search.py --trials 50 --imbalance-strategy focal --metric precision_topn --top-n 5` — Optunaで上位Precisionを最大化する設定を探索し、結果を `config/close_model_params.json` に反映。
- `python analysis/close_param_search.py --metric precision_topn --top-n 5` — グリッドサーチでハイパーパラメータ/不均衡戦略を比較し、TOP5精度を直接評価。
- `pytest` or `pytest -k close_return` — run the full or targeted test suite; always execute before pushing.

## Coding Style & Naming Conventions
- Target Python 3.9; prefer explicit type hints on new public functions.
- Adopt Black defaults (4-space indent, ≤120 characters). Run `black .` before large submissions; couple with `isort .` if import ordering drifts.
- Modules and functions use `snake_case`; classes use `PascalCase`; constants remain `UPPER_SNAKE`. Feature flags and CLI options mirror existing naming (`target_return`, `imbalance_boost`).
- Loggers follow the emoji-prefixed convention (e.g., `logger.info("📊 ...")`) to keep dashboard alerts informative.

## Testing Guidelines
- Pytest is the standard; isolate external I/O via fixtures and sample parquet/CSV files in `tests/data/`.
- Name test modules `test_<module>.py`; align helper fixtures with the feature under test.
- Add at least one unit test per new behaviour and a smoke backtest when model logic or configs change. Document any new cached dataset paths in the PR description.
- Run `pytest` locally before submitting; CI expects zero skipped tests unless justified in-code.

## Commit & Pull Request Guidelines
- Use imperative, scoped commit subjects (`feat: add imbalance boost helper`) capped at 72 characters; include dataset versions, commands executed, and key metrics in the body.
- PRs should link Jira/GitHub issues, describe validation steps, attach relevant artefacts (e.g., `production_reports/2025-10/2025-10-05.md` excerpts), and flag configuration changes.
- Request review only after passing tests and formatting; note outstanding risks or follow-up tasks explicitly.

## Security & Configuration Tips
- Secrets stay in `.env`; provide sanitized samples via `.env.template`. Never commit credentials.
- Heavy pipelines rely on macOS Accelerate; verify NumPy/LightGBM compatibility after dependency bumps.
- Scrub `data_exports/` and `production_reports/` before sharing externally—redact client identifiers and PII.
