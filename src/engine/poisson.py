"""Poisson-based football scoring and outcome probability calculations."""

from dataclasses import dataclass
from math import isfinite
from typing import Final

import polars as pl
from scipy.stats import poisson, skellam

DEFAULT_MAX_GOALS: Final[int] = 10
_MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)


@dataclass(frozen=True)
class TeamStrength:
    """Venue-specific relative attack and defence strengths for a team."""

    home_attack: float
    home_defence: float
    away_attack: float
    away_defence: float


@dataclass(frozen=True)
class LeagueStrengths:
    """League scoring averages and derived strengths for each team."""

    home_goal_average: float
    away_goal_average: float
    teams: dict[str, TeamStrength]


@dataclass(frozen=True)
class ExpectedGoals:
    """Expected home and away goals for a fixture."""

    home: float
    away: float


@dataclass(frozen=True)
class MarketProbabilities:
    """Match outcome, goal-line, and both-teams-to-score probabilities."""

    home_win: float
    draw: float
    away_win: float
    over_1_5: float
    under_1_5: float
    over_2_5: float
    under_2_5: float
    over_3_5: float
    under_3_5: float
    btts_yes: float
    btts_no: float


@dataclass(frozen=True)
class ScoreProbabilityMatrix:
    """Joint scoreline probabilities and exact derived market probabilities."""

    scores: pl.DataFrame
    markets: MarketProbabilities
    probability_mass: float


def _validate_match_data(matches: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(matches, pl.DataFrame):
        raise TypeError("matches must be a Polars DataFrame.")
    if matches.is_empty():
        raise ValueError("matches must contain at least one completed match.")

    missing_columns = sorted(set(_MATCH_COLUMNS) - set(matches.columns))
    if missing_columns:
        raise ValueError(
            "matches is missing required columns: " + ", ".join(missing_columns)
        )

    for column in ("home_goals", "away_goals"):
        if not matches.schema[column].is_numeric():
            raise TypeError(f"{column} must contain numeric goal counts.")

    selected = matches.select(
        pl.col("home_team").cast(pl.String, strict=True),
        pl.col("away_team").cast(pl.String, strict=True),
        pl.col("home_goals").cast(pl.Float64, strict=True),
        pl.col("away_goals").cast(pl.Float64, strict=True),
    )
    if any(selected.null_count().row(0)):
        raise ValueError("matches must not contain null required values.")
    for column in ("home_team", "away_team"):
        if selected.filter(pl.col(column).str.strip_chars() == "").height:
            raise ValueError(f"{column} must not contain empty team names.")
    for column in ("home_goals", "away_goals"):
        invalid = selected.filter(
            pl.col(column).is_null()
            | pl.col(column).is_nan()
            | pl.col(column).is_infinite()
            | (pl.col(column) < 0)
        )
        if invalid.height:
            raise ValueError(f"{column} must contain finite, non-negative values.")
    if selected.filter(pl.col("home_team") == pl.col("away_team")).height:
        raise ValueError("A team cannot play itself.")
    return selected


def calculate_team_strengths(matches: pl.DataFrame) -> LeagueStrengths:
    """Calculate venue-specific team attack and defence strengths.

    The input must contain one completed match per row with ``home_team``,
    ``away_team``, ``home_goals``, and ``away_goals`` columns. Strengths are
    ratios to the corresponding league home- or away-goal average. A missing
    venue-specific team average defaults to neutral strength (1.0).
    """
    data = _validate_match_data(matches)
    home_average_value = data.get_column("home_goals").mean()
    away_average_value = data.get_column("away_goals").mean()
    if home_average_value is None or away_average_value is None:
        raise ValueError("Unable to calculate league scoring averages.")
    home_average = float(home_average_value)
    away_average = float(away_average_value)
    if home_average <= 0 or away_average <= 0:
        raise ValueError(
            "League home and away goal averages must both be greater than zero."
        )

    home_stats = data.group_by("home_team").agg(
        pl.col("home_goals").mean().alias("home_scored"),
        pl.col("away_goals").mean().alias("home_conceded"),
    ).rename({"home_team": "team"})
    away_stats = data.group_by("away_team").agg(
        pl.col("away_goals").mean().alias("away_scored"),
        pl.col("home_goals").mean().alias("away_conceded"),
    ).rename({"away_team": "team"})
    team_stats = home_stats.join(away_stats, on="team", how="full", coalesce=True)

    team_stats = team_stats.with_columns(
        (pl.col("home_scored") / home_average).fill_null(1.0).alias("home_attack"),
        (pl.col("home_conceded") / away_average)
        .fill_null(1.0)
        .alias("home_defence"),
        (pl.col("away_scored") / away_average).fill_null(1.0).alias("away_attack"),
        (pl.col("away_conceded") / home_average)
        .fill_null(1.0)
        .alias("away_defence"),
    )
    teams = {
        row["team"]: TeamStrength(
            home_attack=row["home_attack"],
            home_defence=row["home_defence"],
            away_attack=row["away_attack"],
            away_defence=row["away_defence"],
        )
        for row in team_stats.iter_rows(named=True)
    }
    return LeagueStrengths(
        home_goal_average=home_average,
        away_goal_average=away_average,
        teams=teams,
    )


def calculate_expected_goals(
    home_team: str,
    away_team: str,
    strengths: LeagueStrengths,
) -> ExpectedGoals:
    """Estimate fixture xG from league scoring averages and team strengths."""
    if not home_team or not home_team.strip():
        raise ValueError("home_team must not be empty.")
    if not away_team or not away_team.strip():
        raise ValueError("away_team must not be empty.")
    if home_team == away_team:
        raise ValueError("A team cannot play itself.")
    try:
        home_strength = strengths.teams[home_team]
        away_strength = strengths.teams[away_team]
    except KeyError as exc:
        raise ValueError(f"No strength data is available for team {exc.args[0]!r}.") from exc

    home_xg = (
        strengths.home_goal_average
        * home_strength.home_attack
        * away_strength.away_defence
    )
    away_xg = (
        strengths.away_goal_average
        * away_strength.away_attack
        * home_strength.home_defence
    )
    if not isfinite(home_xg) or not isfinite(away_xg) or home_xg < 0 or away_xg < 0:
        raise ValueError("Calculated expected goals must be finite and non-negative.")
    return ExpectedGoals(home=home_xg, away=away_xg)


def score_probability_matrix(
    home_xg: float,
    away_xg: float,
    *,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> ScoreProbabilityMatrix:
    """Create a scoreline matrix and 1X2, totals, and BTTS probabilities.

    The returned Polars DataFrame has one row per scoreline from 0 to
    ``max_goals`` for each team. Goal-line probabilities and 1X2 probabilities
    are calculated from the full independent Poisson distributions; the
    ``probability_mass`` field reports the portion represented by the finite
    scoreline matrix.
    """
    if not isfinite(home_xg) or not isfinite(away_xg) or home_xg < 0 or away_xg < 0:
        raise ValueError("Expected goals must be finite and non-negative.")
    if isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 0:
        raise ValueError("max_goals must be a non-negative integer.")

    goal_values = list(range(max_goals + 1))
    home_probabilities = poisson.pmf(goal_values, home_xg)
    away_probabilities = poisson.pmf(goal_values, away_xg)
    score_rows = [
        (home_score, away_score, float(home_probability * away_probability))
        for home_score, home_probability in zip(goal_values, home_probabilities)
        for away_score, away_probability in zip(goal_values, away_probabilities)
    ]
    scores = pl.DataFrame(
        score_rows,
        schema={
            "home_goals": pl.Int64,
            "away_goals": pl.Int64,
            "probability": pl.Float64,
        },
        orient="row",
    )
    total_xg = home_xg + away_xg
    if home_xg == 0 and away_xg == 0:
        home_win, draw, away_win = 0.0, 1.0, 0.0
    elif home_xg == 0:
        draw = float(poisson.pmf(0, away_xg))
        home_win, away_win = 0.0, 1.0 - draw
    elif away_xg == 0:
        draw = float(poisson.pmf(0, home_xg))
        home_win, away_win = 1.0 - draw, 0.0
    else:
        home_win = float(skellam.sf(0, home_xg, away_xg))
        draw = float(skellam.pmf(0, home_xg, away_xg))
        away_win = float(skellam.cdf(-1, home_xg, away_xg))

    markets = MarketProbabilities(
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        over_1_5=float(poisson.sf(1, total_xg)),
        under_1_5=float(poisson.cdf(1, total_xg)),
        over_2_5=float(poisson.sf(2, total_xg)),
        under_2_5=float(poisson.cdf(2, total_xg)),
        over_3_5=float(poisson.sf(3, total_xg)),
        under_3_5=float(poisson.cdf(3, total_xg)),
        btts_yes=float((1 - poisson.pmf(0, home_xg)) * (1 - poisson.pmf(0, away_xg))),
        btts_no=float(
            1 - (1 - poisson.pmf(0, home_xg)) * (1 - poisson.pmf(0, away_xg))
        ),
    )
    return ScoreProbabilityMatrix(
        scores=scores,
        markets=markets,
        probability_mass=float(scores.get_column("probability").sum()),
    )
