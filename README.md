# housebreaker

An algorithmic sports betting engine that analyses bookmaker odds and uses machine learning, Poisson distributions, and the Kelly criterion to optimise stake sizing.

## Project Overview

housebreaker is being developed as a modular football analysis and betting engine. It will collect fixtures, team statistics, expected goals (xG), and bookmaker odds; prepare data for analysis; and support modelling, strategy evaluation, and backtesting.

## Architecture

- **Fetchers (`src/fetchers/`)** — asynchronous API-Football and The Odds API clients. They validate responses with Pydantic and honour the configured per-minute request limits.
- **Engine (`src/engine/`)** — planned coordination and match-analysis logic.
- **Machine learning (`src/ml/`)** — planned model training and prediction workflows.
- **Strategy (`src/strategy/`)** — planned probability, value, and stake-sizing strategies.
- **Backtesting (`src/backtest/`)** — planned historical strategy evaluation.
- **Data and models (`data/`, `models/`)** — raw and processed data, plus saved model artefacts.
- **Configuration (`config/settings.yaml`)** — API request limits and league selections.
- **Terminal interface (`src/`)** — Textual is part of the planned user interface stack.

## Tech Stack

- **Python 3.12+**
- **Polars** for columnar data processing
- **Pydantic V2** for validating API responses and application data
- **HTTPX** for asynchronous HTTP requests
- **Textual** for the planned terminal user interface
- **XGBoost** for machine-learning models
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

5. Run the current entry point:

   ```shell
   python main.py
   ```

   The entry point is currently a stub; API clients can be used directly from application code.

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

## Project Status & Changelog

### Current Status

- **Implemented:** project scaffold, YAML-backed API rate limits, asynchronous API-Football and The Odds API fetchers, and Pydantic V2 response models.
- **Planned:** analysis engine, machine-learning workflows, betting strategies, backtesting, and Textual interface.

### Changelog

- **Initial scaffold:** added the configuration, data, model, and source package structure.
- **API fetchers:** added rate-limited asynchronous clients for team statistics, recent form, fixtures, fixture xG, and upcoming or event-specific bookmaker odds.
