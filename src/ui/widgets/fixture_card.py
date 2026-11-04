from rich.text import Text
from textual.widgets import Static

from src.fetchers.football_api import FixtureRecord


class FixtureCard(Static):
    """Display the key details for one football fixture."""

    def __init__(
        self,
        fixture: FixtureRecord,
        *,
        expand: bool = False,
        shrink: bool = False,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        self.fixture = fixture
        super().__init__(
            self._render_fixture(fixture),
            expand=expand,
            shrink=shrink,
            markup=False,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )

    @staticmethod
    def _render_fixture(fixture: FixtureRecord) -> Text:
        home_team = fixture.teams.home.name or "Home team"
        away_team = fixture.teams.away.name or "Away team"
        fixture_date = fixture.fixture.date
        kick_off = (
            fixture_date.strftime("%d %b %Y, %H:%M")
            if fixture_date is not None
            else "Kick-off time unavailable"
        )
        league = fixture.league.name if fixture.league and fixture.league.name else "League"
        status = (
            str(fixture.fixture.status.get("long") or fixture.fixture.status.get("short"))
            if fixture.fixture.status
            else "Scheduled"
        )

        content = Text()
        content.append(f"{home_team}  v  {away_team}\n", style="bold")
        content.append(f"{kick_off}  ·  {league}\n", style="dim")
        content.append(status, style="cyan")
        return content
