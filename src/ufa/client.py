import requests
import pandas as pd
from datetime import date as date_type
from datetime import datetime, timedelta

BASE_URL = "https://www.backend.ufastats.com/api/v1"


def get_games(date=None):
    params = {}

    if date is not None:
        params["date"] = date

    response = requests.get(
        f"{BASE_URL}/games",
        params=params
    )

    if not response.ok:
        print("URL:", response.url)
        print("Status code:", response.status_code)
        print("Response text:", response.text[:1000])
        response.raise_for_status()

    games_json = response.json()
    return pd.DataFrame(games_json["data"])


def _coerce_date(value):
    if isinstance(value, date_type):
        return value

    return datetime.strptime(value, "%Y-%m-%d").date()


def get_games_for_date_range(start_date, end_date=None):
    start_date = _coerce_date(start_date)
    end_date = _coerce_date(end_date) if end_date is not None else date_type.today()

    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date.")

    games = []
    current_date = start_date

    while current_date <= end_date:
        daily_games = get_games(current_date.isoformat())
        if not daily_games.empty:
            games.append(daily_games)
        current_date += timedelta(days=1)

    if not games:
        return pd.DataFrame()

    return pd.concat(games, ignore_index=True)


def get_games_since_2024(end_date=None):
    return get_games_for_date_range("2024-01-01", end_date=end_date)


def search_games(
    start_date,
    end_date=None,
    team=None,
    status=None,
    location=None,
):
    games = get_games_for_date_range(start_date, end_date=end_date)
    if games.empty:
        return games

    filtered = games.copy()

    if team is not None:
        team = team.lower()
        filtered = filtered[
            filtered["awayTeamID"].str.lower().eq(team)
            | filtered["homeTeamID"].str.lower().eq(team)
            | filtered["gameID"].str.lower().str.contains(team, na=False)
        ]

    if status is not None:
        filtered = filtered[filtered["status"].str.lower().eq(status.lower())]

    if location is not None:
        filtered = filtered[
            filtered["location"].str.lower().str.contains(location.lower(), na=False)
        ]

    columns = [
        "gameID",
        "awayTeamID",
        "homeTeamID",
        "awayScore",
        "homeScore",
        "status",
        "startTimestamp",
        "location",
    ]
    return filtered[[column for column in columns if column in filtered.columns]]


def get_game_events(game_id):
    response = requests.get(
        f"{BASE_URL}/gameEvents",
        params={"gameID": game_id}
    )

    if not response.ok:
        print("URL:", response.url)
        print("Status code:", response.status_code)
        print("Response text:", response.text[:1000])
        response.raise_for_status()

    events_json = response.json()

    home_events = pd.json_normalize(events_json["data"]["homeEvents"])
    away_events = pd.json_normalize(events_json["data"]["awayEvents"])

    home_events["team_side"] = "home"
    away_events["team_side"] = "away"

    return pd.concat([home_events, away_events], ignore_index=True)
