import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.fetchers._common import (
    AsyncRateLimiter,
    FetcherConfigurationError,
    FetcherResponseError,
    get_api_response,
    get_json,
    load_rate_limit,
    validate_list,
    validate_model,
)

logger = logging.getLogger(__name__)


class APIModel(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class TeamReference(APIModel):
    id: int | None = None
    name: str | None = None


class LeagueReference(APIModel):
    id: int | None = None
    name: str | None = None
    season: int | None = None


class TeamStatistics(APIModel):
    team: TeamReference
    league: LeagueReference
    form: str | None = None
    fixtures: dict[str, Any] = Field(default_factory=dict)
    goals: dict[str, Any] = Field(default_factory=dict)


class FixtureDetails(APIModel):
    id: int
    date: datetime | None = None
    timestamp: int | None = None
    status: dict[str, Any] | None = None


class FixtureTeams(APIModel):
    home: TeamReference
    away: TeamReference


class FixtureRecord(APIModel):
    fixture: FixtureDetails
    teams: FixtureTeams
    league: LeagueReference | None = None
    goals: dict[str, int | None] = Field(default_factory=dict)


class FixtureStatistic(APIModel):
    type: str
    value: str | int | float | None = None


class FixtureTeamStatistics(APIModel):
    team: TeamReference
    statistics: list[FixtureStatistic]


class FixtureExpectedGoals(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    team: TeamReference
    expected_goals: float = Field(ge=0)


class FootballAPIClient:
    BASE_URL = "https://v3.football.api-sports.io"
    SETTINGS_NAME = "football_api"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        settings_path: str | Path | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("FOOTBALL_API_KEY")
        if not resolved_api_key or not resolved_api_key.strip():
            raise FetcherConfigurationError(
                "Set FOOTBALL_API_KEY or pass api_key when creating FootballAPIClient."
            )

        self._api_key = resolved_api_key.strip()
        requests_per_minute = load_rate_limit(self.SETTINGS_NAME, settings_path)
        self._rate_limiter = AsyncRateLimiter(requests_per_minute)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(20.0),
        )

    async def __aenter__(self) -> "FootballAPIClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_response(
        self, endpoint: str, params: dict[str, str | int]
    ) -> list[dict[str, Any]]:
        payload = await get_json(
            self._client,
            self._rate_limiter,
            f"{self.BASE_URL}{endpoint}",
            params,
            headers={"x-apisports-key": self._api_key},
        )
        return get_api_response(payload, endpoint)

    async def fetch_team_statistics(
        self, league_id: int, season: int, team_id: int
    ) -> TeamStatistics:
        endpoint = "/teams/statistics"
        response = await self._get_response(
            endpoint,
            {"league": league_id, "season": season, "team": team_id},
        )
        if not response:
            raise FetcherResponseError(f"The API returned no team statistics from {endpoint}.")
        statistics = validate_model(TeamStatistics, response[0], endpoint)
        logger.info("Fetched statistics for team %s in league %s.", team_id, league_id)
        return statistics

    async def fetch_recent_form(
        self, league_id: int, season: int, team_id: int
    ) -> str:
        statistics = await self.fetch_team_statistics(league_id, season, team_id)
        if statistics.form is None:
            logger.error("The API response did not include recent form for team %s.", team_id)
            raise FetcherResponseError(
                f"The API response did not include recent form for team {team_id}."
            )
        logger.info("Fetched recent form for team %s.", team_id)
        return statistics.form

    async def fetch_fixtures(
        self,
        league_id: int,
        season: int,
        *,
        team_id: int | None = None,
        fixture_date: date | str | None = None,
    ) -> list[FixtureRecord]:
        endpoint = "/fixtures"
        params: dict[str, str | int] = {"league": league_id, "season": season}
        if team_id is not None:
            params["team"] = team_id
        if fixture_date is not None:
            params["date"] = (
                fixture_date.isoformat()
                if isinstance(fixture_date, date)
                else fixture_date
            )

        response = await self._get_response(endpoint, params)
        fixtures = validate_list(FixtureRecord, response, endpoint)

        logger.info("Fetched %d fixtures for league %s.", len(fixtures), league_id)
        return fixtures

    async def fetch_fixture_xg(self, fixture_id: int) -> list[FixtureExpectedGoals]:
        endpoint = "/fixtures/statistics"
        response = await self._get_response(endpoint, {"fixture": fixture_id})
        statistics = validate_list(FixtureTeamStatistics, response, endpoint)
        expected_goals: list[FixtureExpectedGoals] = []

        for team_statistics in statistics:
            for statistic in team_statistics.statistics:
                normalised_type = "_".join(statistic.type.casefold().split())
                if normalised_type not in {"expected_goals", "xg"}:
                    continue
                if statistic.value is None:
                    continue
                try:
                    value = float(statistic.value)
                except (TypeError, ValueError) as exc:
                    logger.error(
                        "The expected-goals value for fixture %s was not numeric.",
                        fixture_id,
                    )
                    raise FetcherResponseError(
                        f"The expected-goals value for fixture {fixture_id} was not numeric."
                    ) from exc
                expected_goals.append(
                    validate_model(
                        FixtureExpectedGoals,
                        {"team": team_statistics.team, "expected_goals": value},
                        endpoint,
                    )
                )

        if not expected_goals:
            logger.warning("No expected-goals statistics were available for fixture %s.", fixture_id)
            raise FetcherResponseError(
                f"No expected-goals statistics were available for fixture {fixture_id}."
            )
        logger.info("Fetched expected-goals statistics for fixture %s.", fixture_id)
        return expected_goals
