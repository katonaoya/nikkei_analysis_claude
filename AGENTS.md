# Repository Guidelines

## Project Structure & Module Organization
- `daily_trading_automation.py` orchestrates the four daily jobs (data fetch, integration, model run, report).
- Core model logic lives in `systems/` (notably `enhanced_precision_system_v3.py`); shared utilities are under `src/` and `utils/`.
- Data pipelines and loaders reside in `data_management/`; raw and processed artifacts are written to `data/` and mirrored into `production_data/` for live use.
- Generated insights, logs, and validation assets sit in `reports/`, `results/`, `logs/`, and `validation_results/`.
- Tests live in `tests/` (pytest) and mirror the module layout; reference docs and SOPs are under `docment/` and `documentation/`.

## Build, Test, and Development Commands
- `python daily_trading_automation.py` — full end-to-end daily run; relies on `.env` J-Quants credentials and current market calendar.
- `python systems/enhanced_precision_system_v3.py` — executes model training, walk-forward evaluation, and saves outputs to `models/enhanced_v3/`.
- `pytest` — run unit and integration tests in `tests/`; add `-k name` to target specific modules.
- `python production/production_backtest_analyzer.py` — recompute live-trading KPIs before publishing reports.

## Coding Style & Naming Conventions
- Target Python 3.9 via pyenv; prefer type hints for new code paths.
- Follow Black/PEP 8 defaults (4-space indentation, 120-char soft wrap). Run `black .` before large submissions.
- Module names stay snake_case; classes use PascalCase; configuration constants are UPPER_SNAKE.
- Log messages use emoji prefixes already established (e.g., `logger.info("📊 ...")`) to keep automation dashboards consistent.

## Testing Guidelines
- Use pytest fixtures for file I/O stubs; never hit production APIs during tests.
- Name test files `test_*.py` and mirror module names (`test_data_modules.py` covers `data_management`).
- Regression checks for model accuracy should load cached datasets from `data/processed/`; document any data snapshots you add.
- Aim to keep new features covered by at least one unit test and, when relevant, a smoke backtest.

## Commit & Pull Request Guidelines
- Write commits in imperative mood with scoped prefixes when appropriate (e.g., `feat:`, `fix:`, `docs:`); keep subjects ≤72 chars.
- Include context in the body: dataset versions touched, commands run, and relevant metrics (accuracy, precision, runtime).
- For PRs, link Jira/GitHub issues, describe validation steps, attach sample logs (trimmed) or report paths (`production_reports/YYYY-MM/DD.md`), and flag any config changes.

## Security & Configuration Notes
- Store API tokens in `.env`; never commit secrets. Use `.env.template` as the baseline when onboarding.
- Heavy scripts rely on macOS Accelerate; confirm NumPy/LightGBM compatibility after environment upgrades.
- Clean sensitive exports in `data_exports/` before sharing; redact client-specific company names unless contracts allow disclosure.
