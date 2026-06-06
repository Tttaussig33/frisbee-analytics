import requests
import pandas as pd

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