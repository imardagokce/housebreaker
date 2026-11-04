import math

from rich.text import Text
from textual.widgets import Static


def calculate_kelly_stake(
    *,
    probability: float,
    decimal_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Return a fractional Kelly stake in the bankroll's currency units."""
    values = (probability, decimal_odds, bankroll, kelly_fraction)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Recommendation inputs must be finite numbers.")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one.")
    if decimal_odds <= 1:
        raise ValueError("decimal_odds must be greater than one.")
    if bankroll < 0:
        raise ValueError("bankroll must not be negative.")
    if not 0 <= kelly_fraction <= 1:
        raise ValueError("kelly_fraction must be between zero and one.")

    full_kelly = max(
        0.0,
        (probability * decimal_odds - 1) / (decimal_odds - 1),
    )
    return bankroll * full_kelly * kelly_fraction


class StakeRecommendation(Static):
    """Present a fractional Kelly stake when model inputs are available."""

    def __init__(
        self,
        *,
        expand: bool = False,
        shrink: bool = False,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            "No stake recommendation is available until a value signal is selected.",
            expand=expand,
            shrink=shrink,
            markup=False,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )

    def set_recommendation(
        self,
        *,
        probability: float,
        decimal_odds: float,
        bankroll: float,
        kelly_fraction: float = 0.25,
    ) -> None:
        stake = calculate_kelly_stake(
            probability=probability,
            decimal_odds=decimal_odds,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
        )
        expected_return = probability * decimal_odds - 1
        recommendation = Text()
        recommendation.append("Fractional Kelly recommendation\n", style="bold")
        recommendation.append(f"Model probability: {probability:.1%}\n")
        recommendation.append(f"Decimal odds: {decimal_odds:.2f}\n")
        recommendation.append(f"Expected return: {expected_return:+.1%}\n")
        recommendation.append(
            f"Suggested stake: {stake:.2f} bankroll units",
            style="green" if stake > 0 else "yellow",
        )
        self.update(recommendation)
