"""Expected-goal distribution inference for processed fixture features."""

from __future__ import annotations

from datetime import date, datetime
from math import isclose, isfinite
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from scipy.stats import poisson

from src.ml.model import GoalRegressor


class GoalDistribution(BaseModel):
    """Validated Poisson score-count distribution for one fixture side."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    expected_goals: float = Field(ge=0)
    probabilities: list[float] = Field(min_length=1)
    tail_probability: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_probability_mass(self) -> "GoalDistribution":
        probability_mass = sum(self.probabilities) + self.tail_probability
        if not isclose(probability_mass, 1.0, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Goal probabilities and tail probability must sum to one.")
        return self


class FixtureGoalPrediction(BaseModel):
    """Validated home and away expected goals and score-count distributions."""

    model_config = ConfigDict(frozen=True, extra="allow", allow_inf_nan=False)

    fixture_date: date | datetime | None = None
    home_team: str | None = None
    away_team: str | None = None
    home: GoalDistribution
    away: GoalDistribution


def _distribution(expected_goals: float, max_goals: int) -> GoalDistribution:
    if not isfinite(expected_goals) or expected_goals < 0:
        raise ValueError("Expected goals must be finite and non-negative.")
    probabilities = [
        float(probability)
        for probability in poisson.pmf(range(max_goals + 1), expected_goals)
    ]
    tail_probability = float(poisson.sf(max_goals, expected_goals))
    try:
        return GoalDistribution(
            expected_goals=expected_goals,
            probabilities=probabilities,
            tail_probability=tail_probability,
        )
    except ValidationError as exc:
        raise ValueError(f"Unable to validate expected-goal distribution: {exc}") from exc


def predict_goal_distributions(
    model: GoalRegressor,
    features: pl.DataFrame,
    *,
    max_goals: int = 10,
) -> pl.DataFrame:
    """Predict home/away expected goals and Poisson probabilities per fixture.

    The result includes ``home_xg`` and ``away_xg``, a list of probabilities
    for each score count from zero through ``max_goals``, and each distribution's
    probability tail above that count. Optional fixture date and team columns
    are carried through from the supplied processed features.
    """
    if not isinstance(model, GoalRegressor):
        raise TypeError("model must be a fitted GoalRegressor.")
    if isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 0:
        raise ValueError("max_goals must be a non-negative integer.")
    predictions = model.predict(features)
    output_rows: list[dict[str, Any]] = []
    for row in predictions.iter_rows(named=True):
        home_distribution = _distribution(float(row["home_xg"]), max_goals)
        away_distribution = _distribution(float(row["away_xg"]), max_goals)
        prediction_row: dict[str, Any] = {
            "home_xg": home_distribution.expected_goals,
            "away_xg": away_distribution.expected_goals,
            "home_goal_probabilities": home_distribution.probabilities,
            "away_goal_probabilities": away_distribution.probabilities,
            "home_goal_tail_probability": home_distribution.tail_probability,
            "away_goal_tail_probability": away_distribution.tail_probability,
        }
        for column in ("fixture_date", "home_team", "away_team"):
            if column in row:
                prediction_row[column] = row[column]
        try:
            FixtureGoalPrediction.model_validate(
                {
                    **{
                        key: prediction_row[key]
                        for key in ("fixture_date", "home_team", "away_team")
                        if key in prediction_row
                    },
                    "home": home_distribution,
                    "away": away_distribution,
                }
            )
        except ValidationError as exc:
            raise ValueError(f"Predicted fixture distribution failed validation: {exc}") from exc
        output_rows.append(prediction_row)
    if not output_rows:
        raise ValueError("No fixture predictions were produced.")
    return pl.DataFrame(output_rows)
