"""Polars-based recent form and momentum calculations."""

from math import isfinite
from typing import Final

import polars as pl

DEFAULT_FORM_ALPHA: Final[float] = 0.35
_FORM_COLUMNS: Final[tuple[str, ...]] = (
    "team",
    "match_date",
    "xg_for",
    "xg_against",
    "goals_for",
    "goals_against",
    "shots",
)
_EWMA_COLUMNS: Final[dict[str, str]] = {
    "xg_for": "xg_for_ewm",
    "xg_against": "xg_against_ewm",
    "goals_for": "goals_for_ewm",
    "goals_against": "goals_against_ewm",
    "shot_conversion": "shot_conversion_ewm",
}


def _validate_form_data(matches: pl.DataFrame, alpha: float) -> pl.DataFrame:
    if not isinstance(matches, pl.DataFrame):
        raise TypeError("matches must be a Polars DataFrame.")
    if matches.is_empty():
        raise ValueError("matches must contain at least one observation.")
    if not isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in the interval (0, 1].")

    missing_columns = sorted(set(_FORM_COLUMNS) - set(matches.columns))
    if missing_columns:
        raise ValueError(
            "matches is missing required columns: " + ", ".join(missing_columns)
        )
    for column in ("xg_for", "xg_against", "goals_for", "goals_against", "shots"):
        if not matches.schema[column].is_numeric():
            raise TypeError(f"{column} must contain numeric values.")
    if matches.schema["match_date"] not in (pl.Date, pl.Datetime):
        raise TypeError("match_date must use a Polars Date or Datetime type.")

    data = matches.select(
        pl.col("team").cast(pl.String, strict=True),
        pl.col("match_date"),
        *[
            pl.col(column).cast(pl.Float64, strict=True)
            for column in ("xg_for", "xg_against", "goals_for", "goals_against", "shots")
        ],
    )
    required = ("team", "match_date", "xg_for", "xg_against", "goals_for", "goals_against", "shots")
    if any(data.null_count().row(0)):
        raise ValueError("matches must not contain null required values.")
    if data.filter(pl.col("team").str.strip_chars() == "").height:
        raise ValueError("team must not contain empty names.")
    for column in ("xg_for", "xg_against", "goals_for", "goals_against", "shots"):
        if data.filter(
            pl.col(column).is_nan()
            | pl.col(column).is_infinite()
            | (pl.col(column) < 0)
        ).height:
            raise ValueError(f"{column} must contain finite, non-negative values.")
    return data.sort(["team", "match_date"], maintain_order=True)


def calculate_form(
    matches: pl.DataFrame,
    *,
    alpha: float = DEFAULT_FORM_ALPHA,
) -> pl.DataFrame:
    """Add exponentially weighted form, conversion, and momentum columns.

    The input contains one row per team match and must include ``team``,
    ``match_date``, ``xg_for``, ``xg_against``, ``goals_for``, ``goals_against``,
    and ``shots``. Rows are sorted by team and date before calculating EWMAs.
    ``alpha`` controls recency, with larger values placing more weight on the
    latest observations.
    """
    data = _validate_form_data(matches, alpha)
    data = data.with_columns(
        pl.when(pl.col("shots") > 0)
        .then(pl.col("goals_for") / pl.col("shots"))
        .otherwise(None)
        .alias("shot_conversion"),
        pl.when(pl.col("goals_for") > pl.col("goals_against"))
        .then(3.0)
        .when(pl.col("goals_for") == pl.col("goals_against"))
        .then(1.0)
        .otherwise(0.0)
        .alias("form_points"),
    )
    ewm_expressions = [
        pl.col(source)
        .ewm_mean(alpha=alpha, adjust=False, min_samples=1, ignore_nulls=True)
        .over("team", order_by="match_date")
        .alias(target)
        for source, target in _EWMA_COLUMNS.items()
    ]
    data = data.with_columns(*ewm_expressions)
    data = data.with_columns(
        pl.col("form_points")
        .ewm_mean(alpha=alpha, adjust=False, min_samples=1)
        .over("team", order_by="match_date")
        .alias("form_points_ewm")
    )
    return data.with_columns(
        (
            pl.col("form_points_ewm")
            - pl.col("form_points_ewm").shift(1).over("team", order_by="match_date")
        )
        .fill_null(0.0)
        .alias("momentum")
    )


def latest_form(
    matches: pl.DataFrame,
    *,
    alpha: float = DEFAULT_FORM_ALPHA,
) -> pl.DataFrame:
    """Return each team's latest exponentially weighted form summary."""
    form = calculate_form(matches, alpha=alpha)
    return form.group_by("team", maintain_order=True).tail(1).sort("team")
