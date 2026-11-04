from datetime import datetime

from textual.widgets import DataTable

from src.fetchers.odds_api import OddsEvent


class OddsTable(DataTable[str]):
    """Show bookmaker prices for a collection of odds events."""

    def __init__(
        self,
        *,
        zebra_stripes: bool = False,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            zebra_stripes=zebra_stripes,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self._columns_added = False

    def on_mount(self) -> None:
        self._ensure_columns()

    def set_events(self, events: list[OddsEvent]) -> None:
        self._ensure_columns()
        self.clear(columns=False)
        row_count = 0
        for event in events:
            kick_off = self._format_datetime(event.commence_time)
            for bookmaker in event.bookmakers:
                for market in bookmaker.markets:
                    for outcome in market.outcomes:
                        self.add_row(
                            f"{event.home_team} v {event.away_team}",
                            kick_off,
                            bookmaker.title,
                            market.key,
                            outcome.name,
                            f"{outcome.price:.2f}",
                        )
                        row_count += 1
        if row_count == 0:
            self.add_row("No odds available", "—", "—", "—", "—", "—")

    def _ensure_columns(self) -> None:
        if self._columns_added:
            return
        self.add_columns("Fixture", "Kick-off", "Bookmaker", "Market", "Selection", "Odds")
        self._columns_added = True

    @staticmethod
    def _format_datetime(value: datetime) -> str:
        return value.strftime("%d %b %H:%M")
