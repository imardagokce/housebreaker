import logging
from pathlib import Path
from typing import Any

import yaml
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabPane,
    TabbedContent,
)

from src.ui.widgets import OddsTable, StakeRecommendation

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def load_league_options(
    settings_path: Path = SETTINGS_PATH,
) -> tuple[list[tuple[str, str]], str | None]:
    try:
        with settings_path.open(encoding="utf-8") as settings_file:
            settings: Any = yaml.safe_load(settings_file)
    except (OSError, yaml.YAMLError) as exc:
        logger.exception("Unable to load league selections from %s.", settings_path)
        return [], f"Unable to load league selections from {settings_path}: {exc}"

    if not isinstance(settings, dict):
        return [], f"Settings in {settings_path} must contain a YAML mapping."
    leagues = settings.get("league_selections", [])
    if not isinstance(leagues, list):
        return [], "The league_selections setting must be a list."

    options: list[tuple[str, str]] = []
    for league in leagues:
        if isinstance(league, str) and league.strip():
            options.append((league.strip(), league.strip()))
        elif isinstance(league, dict):
            league_id = league.get("id")
            name = league.get("name")
            if league_id is None or not isinstance(name, str) or not name.strip():
                return [], "Each configured league needs an id and a non-empty name."
            options.append((name.strip(), str(league_id)))
        else:
            return [], "Each configured league must be a name or an id/name mapping."
    return options, None


class HousebreakerApp(App[None]):
    """A keyboard-friendly terminal dashboard for housebreaker."""

    TITLE = "housebreaker"
    SUB_TITLE = "Football analysis dashboard"

    CSS = """
    Screen {
        background: #10151d;
        color: #e6edf3;
    }

    #status-indicators {
        height: 3;
        padding: 0 1;
        background: #18212c;
        border-bottom: solid #2d3b4a;
        align: right middle;
    }

    .status-indicator {
        width: auto;
        margin: 0 1;
        padding: 0 1;
        color: #9fb3c8;
    }

    #dashboard {
        height: 1fr;
        min-height: 12;
        padding: 1;
    }

    #sidebar {
        width: 30;
        min-width: 24;
        padding: 1;
        border: round #2d3b4a;
        background: #151c26;
    }

    #sidebar-title, .section-title {
        height: 2;
        color: #78dba9;
        text-style: bold;
    }

    #sidebar Label {
        margin-top: 1;
        color: #aab8c5;
    }

    #sidebar Select, #sidebar Input {
        width: 1fr;
    }

    #refresh-button {
        width: 1fr;
        margin-top: 2;
    }

    #workspace {
        width: 1fr;
        height: 1fr;
        margin-left: 1;
    }

    #connection-summary {
        height: 3;
        margin-bottom: 1;
        padding: 1;
        color: #b5c3d1;
        background: #18212c;
        border: round #2d3b4a;
    }

    TabbedContent {
        height: 1fr;
        border: round #2d3b4a;
        background: #151c26;
    }

    TabPane {
        padding: 1;
    }

    #fixture-odds-layout {
        height: 1fr;
    }

    .data-panel {
        width: 1fr;
        height: 1fr;
        padding: 1;
        border: round #2d3b4a;
    }

    #fixture-list {
        height: 1fr;
    }

    #odds-table {
        height: 1fr;
    }

    .empty-state {
        padding: 1;
        color: #9fb3c8;
    }

    #value-panel, #performance-panel {
        height: 1fr;
        padding: 1;
    }

    #stake-recommendation {
        height: auto;
        margin-top: 1;
        padding: 1;
        border: round #2d3b4a;
    }

    #bottom-panel {
        height: 10;
        min-height: 7;
        padding: 0 1;
    }

    #activity-panel {
        width: 2fr;
        padding: 0 1;
        border: round #2d3b4a;
    }

    #activity-log {
        height: 1fr;
    }

    #system-panel {
        width: 1fr;
        margin-left: 1;
        padding: 1;
        border: round #2d3b4a;
    }

    #system-status {
        height: auto;
        color: #78dba9;
        text-style: bold;
    }

    Footer {
        background: #18212c;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._league_options, self._settings_error = load_league_options()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="status-indicators"):
            yield Static("● INTERFACE READY", classes="status-indicator")
            yield Static("● DATA FEEDS NOT CONNECTED", classes="status-indicator")

        with Horizontal(id="dashboard"):
            with Vertical(id="sidebar"):
                yield Static("CONTROL DESK", id="sidebar-title")
                yield Label("League")
                yield Select(
                    self._league_options,
                    prompt="No leagues configured",
                    allow_blank=True,
                    disabled=not self._league_options,
                    id="league-select",
                )
                yield Label("Team")
                yield Input(
                    placeholder="Enter a team name",
                    id="team-input",
                )
                yield Button("Refresh data", variant="primary", id="refresh-button")
                yield Static(
                    "Configure league_selections in config/settings.yaml.",
                    id="selection-help",
                    classes="empty-state",
                )

            with Vertical(id="workspace"):
                yield Static(
                    "Select a configured league and team to prepare a refresh request.",
                    id="connection-summary",
                )
                with TabbedContent(initial="fixtures-tab", id="dashboard-tabs"):
                    with TabPane("Fixtures & Live Odds", id="fixtures-tab"):
                        with Horizontal(id="fixture-odds-layout"):
                            with Vertical(classes="data-panel"):
                                yield Static("FIXTURES", classes="section-title")
                                with VerticalScroll(id="fixture-list"):
                                    yield Static(
                                        "No fixture data loaded yet.",
                                        classes="empty-state",
                                    )
                            with Vertical(classes="data-panel"):
                                yield Static("LIVE & UPCOMING ODDS", classes="section-title")
                                yield OddsTable(id="odds-table", zebra_stripes=True)

                    with TabPane("Value Signals", id="value-tab"):
                        with Vertical(id="value-panel"):
                            yield Static("VALUE SIGNALS", classes="section-title")
                            yield Static(
                                "Value signals will appear here when model probabilities "
                                "and bookmaker prices are available.",
                                classes="empty-state",
                            )
                            yield StakeRecommendation(id="stake-recommendation")

                    with TabPane("Model Performance", id="performance-tab"):
                        with Vertical(id="performance-panel"):
                            yield Static("MODEL PERFORMANCE", classes="section-title")
                            yield Static(
                                "No model evaluation results have been recorded.",
                                classes="empty-state",
                            )

        with Horizontal(id="bottom-panel"):
            with Vertical(id="activity-panel"):
                yield Static("ACTIVITY LOG", classes="section-title")
                yield RichLog(id="activity-log", wrap=True, markup=False)
            with Vertical(id="system-panel"):
                yield Static("SYSTEM STATUS", classes="section-title")
                yield Static("Initialising dashboard…", id="system-status")
                yield Static("No league or team selected.", id="selection-status")
        yield Footer()

    def on_mount(self) -> None:
        activity_log = self.query_one("#activity-log", RichLog)
        activity_log.write("Dashboard initialised.")
        if self._settings_error:
            activity_log.write(f"Configuration issue: {self._settings_error}")
            self.query_one("#system-status", Static).update("Configuration needs attention.")
        else:
            activity_log.write("No live data providers are connected to the dashboard.")
            self.query_one("#system-status", Static).update("Ready · display-only mode")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "league-select":
            return
        self._update_selection_status()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "team-input":
            self._update_selection_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-button":
            self.action_refresh()

    def action_refresh(self) -> None:
        league = self.query_one("#league-select", Select).value
        team = self.query_one("#team-input", Input).value.strip()
        activity_log = self.query_one("#activity-log", RichLog)
        status = self.query_one("#system-status", Static)

        if not isinstance(league, str) or not league:
            message = "Refresh not started: configure and select a league first."
            activity_log.write(message)
            status.update("Waiting for league selection")
            return

        team_label = team if team else "all teams"
        message = (
            f"Refresh requested for league {league}, {team_label}. "
            "Live data providers are not connected."
        )
        activity_log.write(message)
        status.update("Refresh unavailable · no data provider")
        logger.info("%s", message)

    def _update_selection_status(self) -> None:
        league = self.query_one("#league-select", Select).value
        team = self.query_one("#team-input", Input).value.strip()
        if isinstance(league, str) and league:
            selected_league = next(
                (label for label, value in self._league_options if value == league),
                league,
            )
            message = f"League: {selected_league} · Team: {team or 'All teams'}"
        else:
            message = f"No league selected · Team: {team or 'Not specified'}"
        self.query_one("#selection-status", Static).update(Text(message))
