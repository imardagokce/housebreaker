# housebreaker

An algorithmic sports betting engine that analyses bookmaker odds and uses machine learning, Poisson distributions, and the Kelly criterion to optimise stake sizing.

## Project Overview

housebreaker is being developed as a modular football analysis and betting engine. It will collect fixtures, team statistics, expected goals (xG), and bookmaker odds; prepare data for analysis; and support modelling, strategy evaluation, and backtesting.

## Architecture

- **Fetchers (`src/fetchers/`)** — asynchronous API-Football and The Odds API clients. They validate responses with Pydantic and honour the configured per-minute request limits.
- **Engine (`src/engine/`)** — Polars-based team strength and exponentially weighted form calculations, plus SciPy Poisson scoreline and market probabilities.
- **Machine learning (`src/ml/`)** — chronological Polars feature engineering, Elo/Glicko ratings, independent XGBoost/LightGBM home/away goal regressors, and Poisson expected-goal distributions.
- **Strategy (`src/strategy/`)** — planned probability, value, and stake-sizing strategies.
- **Backtesting (`src/backtest/`)** — planned historical strategy evaluation.
- **Data and models (`data/`, `models/`)** — raw and processed data, plus saved model artefacts.
- **Configuration (`config/settings.yaml`)** — API request limits and league selections.
- **Terminal interface (`src/ui/`)** — a Textual dashboard with league/team controls, tabbed data views, activity and status panels, and reusable display widgets.

## Tech Stack

- **Python 3.12+**
- **Polars** for columnar data processing
- **Pydantic V2** for validating API responses and application data
- **HTTPX** for asynchronous HTTP requests
- **Textual** for the terminal user interface
- **XGBoost** for machine-learning models
- **LightGBM** for alternative gradient-boosted models
- **scikit-learn** for the estimator API used by both model backends
- **PyYAML** for application configuration

Additional dependencies currently listed in `requirements.txt` include LightGBM, SciPy, and `python-dotenv`.

## Installation and Local Setup

1. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

2. Install the project dependencies:

   ```shell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your API credentials:

   ```text
   FOOTBALL_API_KEY=your_football_api_key
   THE_ODDS_API_KEY=your_the_odds_api_key
   ```

   The fetcher clients read these environment variables directly. Load `.env` in your local shell or application with `python-dotenv` before initialising the clients.

4. Review `config/settings.yaml`. The default request limits are conservative, and `league_selections` is empty until leagues are chosen.

5. Launch the terminal dashboard:

   ```shell
   python main.py
   ```

   The dashboard provides league and team selectors, fixtures and odds tables, value-signal and model-performance panels, an activity log, and status indicators. Live data-provider integration is not yet connected to the dashboard.

### Fetcher Usage

```python
import asyncio

from dotenv import load_dotenv

from src.fetchers.football_api import FootballAPIClient
from src.fetchers.odds_api import OddsAPIClient


async def main() -> None:
    load_dotenv()
    async with FootballAPIClient() as football:
        form = await football.fetch_recent_form(league_id=39, season=2026, team_id=42)
        print(form)

    async with OddsAPIClient() as odds:
        events = await odds.fetch_odds("soccer_epl")
        print(f"Retrieved odds for {len(events)} events")


asyncio.run(main())
```

The football client also provides `fetch_team_statistics`, `fetch_fixtures`, and `fetch_fixture_xg`. Both clients use the rate limits in `config/settings.yaml`, validate API responses, and raise fetcher-specific exceptions when requests or responses fail.

The odds client provides `fetch_event_odds` for odds on a specific upcoming or in-play event.

### Machine Learning Usage

Build historical features in date order and train independent home/away goal models:

```python
from src.ml import FeatureBuilder, GoalRegressor, ModelSettings

features = FeatureBuilder().build(historical_matches)
model = GoalRegressor(
    ModelSettings(backend="xgboost", n_estimators=200)
).fit(features)
metrics = model.evaluate(held_out_features)
model.save("models/goals.json")
```

`historical_matches` should be a Polars DataFrame with `fixture_date`, `home_team`, `away_team`, `home_goals`, `away_goals`, `home_xg`, and `away_xg` columns; `home_shots` and `away_shots` are optional. For upcoming matches, pass completed results and fixture rows separately to `FeatureBuilder.build(history, fixtures)`. Fixture features use results strictly before each fixture date. The regressor supports `xgboost` and `lightgbm` backends, MAE/RMSE evaluation, `.pkl` artefacts for either backend, and native `.json` artefacts for XGBoost. Only load pickle artefacts from trusted sources.

For Poisson goal-count distributions from processed fixture features:

```python
from src.ml import predict_goal_distributions

distributions = predict_goal_distributions(model, upcoming_features, max_goals=10)
```

## Project Status & Changelog

### Current Status

- **Implemented:** project scaffold, YAML-backed API rate limits, asynchronous API-Football and The Odds API fetchers, Pydantic V2 response models, a display-only Textual dashboard with reusable fixture, odds, and stake-recommendation widgets, statistical match/form calculations, Polars ML feature engineering, and XGBoost/LightGBM goal regressors with Poisson inference.
- **Planned:** betting strategies, backtesting, and connecting dashboard refreshes to live data providers.

### Changelog

- **Initial scaffold:** added the configuration, data, model, and source package structure.
- **API fetchers:** added rate-limited asynchronous clients for team statistics, recent form, fixtures, fixture xG, and upcoming or event-specific bookmaker odds.
- **Terminal dashboard:** added a Textual interface with league and team controls, tabbed data panels, activity/status views, fixture cards, odds tables, and fractional Kelly stake recommendations.
- **Statistical engine:** added venue-specific attack/defence strengths, xG estimates, scoreline and 1X2/goal-line/BTTS probabilities, plus exponentially weighted team form and momentum metrics.
- **Machine-learning module:** added leakage-aware rolling, Elo/Glicko, rest-day, and H2H features; independent home/away goal models with evaluation and persistence; and Poisson goal-count inference.
