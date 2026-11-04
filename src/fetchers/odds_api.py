import logging
import os
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.fetchers._common import (
    AsyncRateLimiter,
    FetcherConfigurationError,
    get_json,
    load_rate_limit,
    validate_list,
    validate_model,
)

logger = logging.getLogger(__name__)


class OddsModel(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class OddsOutcome(OddsModel):
    name: str
    price: float = Field(gt=0)
    point: float | None = None


class OddsMarket(OddsModel):
    key: str
    outcomes: list[OddsOutcome] = Field(default_factory=list)


class OddsBookmaker(OddsModel):
    key: str
    title: str
    markets: list[OddsMarket] = Field(default_factory=list)


class OddsEvent(OddsModel):
    id: str
    sport_key: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: list[OddsBookmaker] = Field(default_factory=list)


class OddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"
    SETTINGS_NAME = "the_odds_api"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        settings_path: str | Path | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        if not resolved_api_key or not resolved_api_key.strip():
            raise FetcherConfigurationError(
                "Set THE_ODDS_API_KEY or pass api_key when creating OddsAPIClient."
            )

        requests_per_minute = load_rate_limit(self.SETTINGS_NAME, settings_path)
        self._api_key = resolved_api_key.strip()
        self._rate_limiter = AsyncRateLimiter(requests_per_minute)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0))

    async def __aenter__(self) -> "OddsAPIClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_odds(
        self,
        sport_key: str,
        *,
        regions: str = "uk",
        markets: str = "h2h",
    ) -> list[OddsEvent]:
        if not sport_key.strip():
            raise ValueError("sport_key must not be empty.")
        if not regions.strip():
            raise ValueError("regions must not be empty.")
        if not markets.strip():
            raise ValueError("markets must not be empty.")

        endpoint = f"/sports/{sport_key}/odds"
        payload = await get_json(
            self._client,
            self._rate_limiter,
            f"{self.BASE_URL}{endpoint}",
            {
                "apiKey": self._api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        events = validate_list(OddsEvent, payload, endpoint)

        logger.info(
            "Fetched %d odds events for sport %s in regions %s.",
            len(events),
            sport_key,
            regions,
        )
        return events

    async def fetch_event_odds(
        self,
        sport_key: str,
        event_id: str,
        *,
        regions: str = "uk",
        markets: str = "h2h",
    ) -> OddsEvent:
        if not sport_key.strip():
            raise ValueError("sport_key must not be empty.")
        if not event_id.strip():
            raise ValueError("event_id must not be empty.")
        if not regions.strip():
            raise ValueError("regions must not be empty.")
        if not markets.strip():
            raise ValueError("markets must not be empty.")

        endpoint = f"/sports/{sport_key}/events/{event_id}/odds"
        payload = await get_json(
            self._client,
            self._rate_limiter,
            f"{self.BASE_URL}{endpoint}",
            {
                "apiKey": self._api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        event = validate_model(OddsEvent, payload, endpoint)
        logger.info("Fetched odds for event %s in sport %s.", event_id, sport_key)
        return event
