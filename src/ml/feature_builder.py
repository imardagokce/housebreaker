"""Chronological football fixture feature engineering with Polars."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from math import isfinite, log, pi, sqrt
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_FORM_ALPHA: Final[float] = 0.35
DEFAULT_ELO_RATING: Final[float] = 1500.0
DEFAULT_GLICKO_RATING: Final[float] = 1500.0
DEFAULT_GLICKO_RD: Final[float] = 350.0
DEFAULT_GLICKO_C: Final[float] = 34.6

_REQUIRED_MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "fixture_date",
    "home_team",
    "away_team",
)
_LABEL_COLUMNS: Final[tuple[str, ...]] = ("home_goals", "away_goals")
_HISTORY_TARGET_COLUMNS: Final[tuple[str, ...]] = _LABEL_COLUMNS + ("home_xg", "away_xg")
_OPTIONAL_MATCH_COLUMNS: Final[tuple[str, ...]] = ("home_shots", "away_shots")
_METRIC_NAMES: Final[tuple[str, ...]] = (
    "xg_for",
    "xg_against",
    "goals_for",
    "goals_against",
    "shot_conversion",
)
_VENUES: Final[tuple[str, ...]] = ("overall", "home", "away")


class FeatureBuilderSettings(BaseModel):
    """Validated ratings and form settings for the feature builder."""

    model_config = ConfigDict(frozen=True)

    form_alpha: float = Field(default=DEFAULT_FORM_ALPHA, gt=0, le=1, allow_inf_nan=False)
    elo_initial_rating: float = Field(default=DEFAULT_ELO_RATING, gt=0, allow_inf_nan=False)
    elo_k_factor: float = Field(default=20.0, gt=0, allow_inf_nan=False)
    glicko_initial_rating: float = Field(
        default=DEFAULT_GLICKO_RATING, gt=0, allow_inf_nan=False
    )
    glicko_initial_rd: float = Field(
        default=DEFAULT_GLICKO_RD, gt=0, le=350, allow_inf_nan=False
    )
    glicko_c: float = Field(default=DEFAULT_GLICKO_C, gt=0, allow_inf_nan=False)
    h2h_window: int = Field(default=5, ge=1)


class ProcessedFixtureFeatures(BaseModel):
    """Validated numeric feature row and match labels for model training."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    fixture_date: date | datetime
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    home_goals: float | None = Field(default=None, ge=0)
    away_goals: float | None = Field(default=None, ge=0)
    home_xg: float | None = Field(default=None, ge=0)
    away_xg: float | None = Field(default=None, ge=0)
    home_elo: float = Field(gt=0)
    away_elo: float = Field(gt=0)
    home_glicko: float = Field(gt=0)
    away_glicko: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_extra_numeric_features(self) -> "ProcessedFixtureFeatures":
        for name, value in (self.__pydantic_extra__ or {}).items():
            if isinstance(value, float) and not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        return self


class _TeamMatch:
    def __init__(
        self,
        match_date: date | datetime,
        *,
        venue: str,
        xg_for: float,
        xg_against: float,
        goals_for: float,
        goals_against: float,
        shots: float | None,
    ) -> None:
        self.match_date: date | datetime = match_date
        self.venue: str = venue
        self.metrics: dict[str, float | None] = {
            "xg_for": xg_for,
            "xg_against": xg_against,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "shot_conversion": goals_for / shots if shots is not None and shots > 0 else None,
        }


class _Rating:
    def __init__(self, value: float, rd: float = DEFAULT_GLICKO_RD) -> None:
        self.value: float = value
        self.rd: float = rd
        self.last_date: date | datetime | None = None


def _validate_matches(
    matches: pl.DataFrame,
    name: str,
    *,
    require_results: bool,
) -> pl.DataFrame:
    if not isinstance(matches, pl.DataFrame):
        raise TypeError(f"{name} must be a Polars DataFrame.")
    if matches.is_empty():
        return matches

    expected_columns: set[str] = set(_REQUIRED_MATCH_COLUMNS)
    if require_results:
        expected_columns.update(_HISTORY_TARGET_COLUMNS)
    missing = sorted(expected_columns - set(matches.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}.")
    if matches.schema["fixture_date"] not in (pl.Date, pl.Datetime):
        raise TypeError(f"{name}.fixture_date must be a Polars Date or Datetime.")
    for column in _HISTORY_TARGET_COLUMNS:
        if column in matches.columns and not matches.schema[column].is_numeric():
            raise TypeError(f"{name}.{column} must be numeric.")
    for column in _OPTIONAL_MATCH_COLUMNS:
        if column in matches.columns and not matches.schema[column].is_numeric():
            raise TypeError(f"{name}.{column} must be numeric when provided.")

    expressions: list[pl.Expr] = [
        pl.col("fixture_date"),
        pl.col("home_team").cast(pl.String, strict=True),
        pl.col("away_team").cast(pl.String, strict=True),
    ]
    expressions.extend(
        pl.col(column).cast(pl.Float64, strict=True)
        if column in matches.columns
        else pl.lit(None, dtype=pl.Float64).alias(column)
        for column in _HISTORY_TARGET_COLUMNS
    )
    expressions.extend(
        pl.col(column).cast(pl.Float64, strict=True)
        if column in matches.columns
        else pl.lit(None, dtype=pl.Float64).alias(column)
        for column in _OPTIONAL_MATCH_COLUMNS
    )
    data = matches.select(*expressions)
    required_nulls: list[str] = list(_REQUIRED_MATCH_COLUMNS)
    if require_results:
        required_nulls.extend(_HISTORY_TARGET_COLUMNS)
    if any(data.select(required_nulls).null_count().row(0)):
        raise ValueError(f"{name} contains null values in required columns.")

    for column in _HISTORY_TARGET_COLUMNS + _OPTIONAL_MATCH_COLUMNS:
        if data.filter(
            pl.col(column).is_not_null()
            & (
                pl.col(column).is_nan()
                | pl.col(column).is_infinite()
                | (pl.col(column) < 0)
            )
        ).height:
            raise ValueError(f"{name}.{column} must contain finite, non-negative values.")
    for column in ("home_team", "away_team"):
        if data.filter(pl.col(column).str.strip_chars() == "").height:
            raise ValueError(f"{name}.{column} must not contain empty names.")
    if data.filter(pl.col("home_team") == pl.col("away_team")).height:
        raise ValueError(f"{name} contains a team playing itself.")
    return data.sort(["fixture_date", "home_team", "away_team"], maintain_order=True)


def _days_between(later: date | datetime, earlier: date | datetime) -> float:
    if isinstance(later, datetime):
        if not isinstance(earlier, datetime):
            raise TypeError("Match dates must use a consistent date or datetime type.")
        return (later - earlier).total_seconds() / 86400
    if not isinstance(earlier, datetime):
        return float((later - earlier).days)
    raise TypeError("Match dates must use a consistent date or datetime type.")


def _ewma(history: list[_TeamMatch], metric: str, alpha: float) -> float:
    observations = [
        match.metrics[metric]
        for match in history
        if match.metrics[metric] is not None
    ]
    if not observations:
        return 0.0
    return float(
        pl.Series(metric, observations, dtype=pl.Float64)
        .ewm_mean(alpha=alpha, adjust=False, min_samples=1)
        .tail(1)
        .item()
    )


def _glicko_g(rd: float) -> float:
    return 1.0 / sqrt(1.0 + 3.0 * rd**2 / (pi**2 * 400.0**2))


def _glicko_expected(rating: float, opponent_rating: float, opponent_rd: float) -> float:
    g = _glicko_g(opponent_rd)
    return 1.0 / (1.0 + 10.0 ** (-g * (rating - opponent_rating) / 400.0))


def _glicko_period_rd(rating: _Rating, match_date: date | datetime, settings: FeatureBuilderSettings) -> float:
    if rating.last_date is None:
        return rating.rd
    periods = max(0.0, _days_between(match_date, rating.last_date) / 30.0)
    return min(settings.glicko_initial_rd, sqrt(rating.rd**2 + periods * settings.glicko_c**2))


def _update_glicko(
    rating: float,
    rd: float,
    opponent_rating: float,
    opponent_rd: float,
    result: float,
) -> tuple[float, float]:
    g = _glicko_g(opponent_rd)
    expected = _glicko_expected(rating, opponent_rating, opponent_rd)
    variance = 1.0 / (g**2 * expected * (1.0 - expected))
    new_rd = 1.0 / sqrt(1.0 / rd**2 + 1.0 / variance)
    new_rating = rating + 0.0057565 * new_rd**2 * g * (result - expected)
    return new_rating, new_rd


class FeatureBuilder:
    """Build leakage-aware rolling, rating, rest, and H2H fixture features."""

    def __init__(self, settings: FeatureBuilderSettings | None = None) -> None:
        if settings is not None and not isinstance(settings, FeatureBuilderSettings):
            raise TypeError("settings must be a FeatureBuilderSettings instance.")
        self.settings = settings if settings is not None else FeatureBuilderSettings()

    def build(
        self,
        matches: pl.DataFrame,
        fixtures: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Create features for fixtures using only matches before each date.

        ``matches`` provides completed historical results. If ``fixtures`` is
        omitted, features are built for the historical matches themselves.
        For a future prediction, pass processed historical matches separately
        from the upcoming fixture rows. The expected feature inputs are
        ``fixture_date``, ``home_team``, ``away_team``, ``home_goals``,
        ``away_goals``, ``home_xg``, and ``away_xg``; shot columns are optional.
        """
        history = _validate_matches(matches, "matches", require_results=True)
        if history.is_empty():
            raise ValueError("matches must contain at least one completed match.")
        fixtures_data = _validate_matches(
            fixtures if fixtures is not None else matches,
            "fixtures",
            require_results=False,
        )
        if fixtures_data.is_empty():
            raise ValueError("fixtures must contain at least one fixture.")
        if history.schema["fixture_date"] != fixtures_data.schema["fixture_date"]:
            raise TypeError("matches and fixtures must use the same fixture_date type.")

        historical_rows = history.iter_rows(named=True)
        history_rows = list(historical_rows)
        history_index = 0
        team_matches: dict[str, list[_TeamMatch]] = defaultdict(list)
        elo_ratings: dict[str, float] = defaultdict(
            lambda: self.settings.elo_initial_rating
        )
        glicko_ratings: dict[str, _Rating] = {}
        head_to_head: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        feature_rows: list[dict[str, Any]] = []
        for fixture in fixtures_data.iter_rows(named=True):
            fixture_date = fixture["fixture_date"]
            while (
                history_index < len(history_rows)
                and history_rows[history_index]["fixture_date"] < fixture_date
            ):
                self._apply_result(
                    history_rows[history_index],
                    team_matches,
                    elo_ratings,
                    glicko_ratings,
                    head_to_head,
                )
                history_index += 1
            feature_row = self._features_for_fixture(
                fixture,
                team_matches,
                elo_ratings,
                glicko_ratings,
                head_to_head,
            )
            feature_row["home_goals"] = fixture["home_goals"]
            feature_row["away_goals"] = fixture["away_goals"]
            feature_row["home_xg"] = fixture["home_xg"]
            feature_row["away_xg"] = fixture["away_xg"]
            feature_rows.append(feature_row)

        if not feature_rows:
            raise ValueError("No fixture features could be generated.")
        try:
            validated_rows = [
                ProcessedFixtureFeatures.model_validate(row).model_dump()
                for row in feature_rows
            ]
        except ValidationError as exc:
            raise ValueError(f"Generated fixture features failed validation: {exc}") from exc
        return pl.DataFrame(validated_rows).sort(
            ["fixture_date", "home_team", "away_team"],
            maintain_order=True,
        )

    def _features_for_fixture(
        self,
        fixture: dict[str, Any],
        team_matches: dict[str, list[_TeamMatch]],
        elo_ratings: dict[str, float],
        glicko_ratings: dict[str, _Rating],
        head_to_head: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> dict[str, Any]:
        match_date = fixture["fixture_date"]
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        home_history = team_matches[home_team]
        away_history = team_matches[away_team]
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]
        home_glicko = glicko_ratings.get(
            home_team,
            _Rating(self.settings.glicko_initial_rating, self.settings.glicko_initial_rd),
        )
        away_glicko = glicko_ratings.get(
            away_team,
            _Rating(self.settings.glicko_initial_rating, self.settings.glicko_initial_rd),
        )
        home_pair = head_to_head[(home_team, away_team)]
        away_pair = head_to_head[(away_team, home_team)]
        h2h = sorted(home_pair + away_pair, key=lambda match: match["fixture_date"])[
            -self.settings.h2h_window :
        ]
        h2h_home_wins = sum(
            1
            for match in h2h
            if (
                match["home_team"] == home_team
                and match["home_goals"] > match["away_goals"]
            )
            or (
                match["away_team"] == home_team
                and match["away_goals"] > match["home_goals"]
            )
        )
        h2h_draws = sum(match["home_goals"] == match["away_goals"] for match in h2h)

        features: dict[str, Any] = {
            "fixture_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
            "home_glicko": home_glicko.value,
            "away_glicko": away_glicko.value,
            "glicko_difference": home_glicko.value - away_glicko.value,
            "home_glicko_rd": _glicko_period_rd(home_glicko, match_date, self.settings),
            "away_glicko_rd": _glicko_period_rd(away_glicko, match_date, self.settings),
            "home_rest_days": self._rest_days(home_history, match_date),
            "away_rest_days": self._rest_days(away_history, match_date),
            "h2h_matches": float(len(h2h)),
            "h2h_home_win_rate": h2h_home_wins / len(h2h) if h2h else 0.0,
            "h2h_draw_rate": h2h_draws / len(h2h) if h2h else 0.0,
            "h2h_away_win_rate": (
                (len(h2h) - h2h_home_wins - h2h_draws) / len(h2h) if h2h else 0.0
            ),
            "h2h_home_goals_ewm": self._h2h_goals(h2h, home_team, self.settings.form_alpha),
            "h2h_away_goals_ewm": self._h2h_goals(h2h, away_team, self.settings.form_alpha),
        }
        for team_side, team_history in (
            ("home", home_history),
            ("away", away_history),
        ):
            for venue in _VENUES:
                venue_history = (
                    team_history
                    if venue == "overall"
                    else [match for match in team_history if match.venue == venue]
                )
                for metric in _METRIC_NAMES:
                    metric_value = _ewma(
                        venue_history,
                        metric,
                        self.settings.form_alpha,
                    )
                    features[f"{team_side}_{venue}_{metric}_ewm"] = metric_value
            features[f"{team_side}_xg_differential"] = (
                features[f"{team_side}_overall_xg_for_ewm"]
                - features[f"{team_side}_overall_xg_against_ewm"]
            )
        features["home_away_xg_differential"] = (
            features["home_home_xg_for_ewm"]
            - features["home_home_xg_against_ewm"]
            - features["away_away_xg_for_ewm"]
            + features["away_away_xg_against_ewm"]
        )
        return features

    @staticmethod
    def _rest_days(history: list[_TeamMatch], match_date: date | datetime) -> float:
        if not history:
            return 0.0
        return max(0.0, _days_between(match_date, history[-1].match_date))

    @staticmethod
    def _h2h_goals(
        h2h: list[dict[str, Any]],
        team: str,
        alpha: float,
    ) -> float:
        goals = [
            match["home_goals"]
            if match["home_team"] == team
            else match["away_goals"]
            for match in h2h
            if match["home_team"] == team or match["away_team"] == team
        ]
        if not goals:
            return 0.0
        return float(
            pl.Series("goals", goals, dtype=pl.Float64)
            .ewm_mean(alpha=alpha, adjust=False, min_samples=1)
            .tail(1)
            .item()
        )

    def _apply_result(
        self,
        match: dict[str, Any],
        team_matches: dict[str, list[_TeamMatch]],
        elo_ratings: dict[str, float],
        glicko_ratings: dict[str, _Rating],
        head_to_head: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> None:
        home_team, away_team = match["home_team"], match["away_team"]
        match_date = match["fixture_date"]
        home_rating = glicko_ratings.setdefault(
            home_team,
            _Rating(self.settings.glicko_initial_rating, self.settings.glicko_initial_rd),
        )
        away_rating = glicko_ratings.setdefault(
            away_team,
            _Rating(self.settings.glicko_initial_rating, self.settings.glicko_initial_rd),
        )
        home_rd = _glicko_period_rd(home_rating, match_date, self.settings)
        away_rd = _glicko_period_rd(away_rating, match_date, self.settings)

        home_goals, away_goals = match["home_goals"], match["away_goals"]
        if home_goals > away_goals:
            result = 1.0
        elif home_goals == away_goals:
            result = 0.5
        else:
            result = 0.0
        expected = 1.0 / (
            1.0 + 10.0 ** ((elo_ratings[away_team] - elo_ratings[home_team]) / 400.0)
        )
        goal_difference = abs(home_goals - away_goals)
        margin_multiplier = log(goal_difference + 1.0) * (
            2.2 / (abs(elo_ratings[home_team] - elo_ratings[away_team]) * 0.001 + 2.2)
        )
        rating_change = (
            self.settings.elo_k_factor * margin_multiplier * (result - expected)
        )
        elo_ratings[home_team] += rating_change
        elo_ratings[away_team] -= rating_change

        new_home_rating, new_home_rd = _update_glicko(
            home_rating.value,
            home_rd,
            away_rating.value,
            away_rd,
            result,
        )
        new_away_rating, new_away_rd = _update_glicko(
            away_rating.value,
            away_rd,
            home_rating.value,
            home_rd,
            1.0 - result,
        )
        home_rating.value, home_rating.rd, home_rating.last_date = (
            new_home_rating,
            new_home_rd,
            match_date,
        )
        away_rating.value, away_rating.rd, away_rating.last_date = (
            new_away_rating,
            new_away_rd,
            match_date,
        )

        for team, venue, xg_for, xg_against, goals_for, goals_against, shots in (
            (
                home_team,
                "home",
                match["home_xg"],
                match["away_xg"],
                home_goals,
                away_goals,
                match["home_shots"],
            ),
            (
                away_team,
                "away",
                match["away_xg"],
                match["home_xg"],
                away_goals,
                home_goals,
                match["away_shots"],
            ),
        ):
            team_matches[team].append(
                _TeamMatch(
                    match_date,
                    venue=venue,
                    xg_for=xg_for,
                    xg_against=xg_against,
                    goals_for=goals_for,
                    goals_against=goals_against,
                    shots=shots,
                )
            )
        head_to_head[(home_team, away_team)].append(match)


def build_fixture_features(
    matches: pl.DataFrame,
    fixtures: pl.DataFrame | None = None,
    *,
    settings: FeatureBuilderSettings | None = None,
) -> pl.DataFrame:
    """Build chronological fixture features with a configured builder."""
    return FeatureBuilder(settings).build(matches, fixtures)
