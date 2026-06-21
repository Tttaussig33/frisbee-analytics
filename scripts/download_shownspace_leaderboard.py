import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://shownspace.com"
SEASONS = ["2026", "2025", "2024", "2023", "2022", "2021", "Career"]


def _leaderboard_url(season):
    if str(season).lower() == "career":
        return f"{BASE_URL}/api/leaderboard/career"
    return f"{BASE_URL}/api/leaderboard?season={season}"


def _flatten_row(row, season):
    flattened = {
        "season": season,
        "player_id": row.get("PlayerID"),
        "player": row.get("full_name"),
        "team_id": row.get("TeamID"),
    }

    for key, value in row.items():
        if key in {"PlayerID", "full_name", "TeamID", "metrics"}:
            continue
        flattened[key] = value

    metrics = row.get("metrics") or {}
    for metric_name, metric in metrics.items():
        if isinstance(metric, dict):
            flattened[f"{metric_name}_display_value"] = metric.get("value")
            flattened[f"{metric_name}_percentile"] = metric.get("percentile")

    return flattened


def fetch_leaderboard(season, timeout=30):
    response = requests.get(
        _leaderboard_url(season),
        timeout=timeout,
        headers={
            "User-Agent": (
                "Ultimate-Frisbee-Analytics educational leaderboard export "
                "(contact: shownspace.analytics@gmail.com)"
            )
        },
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    rows = payload.get("data") or []
    return pd.DataFrame([_flatten_row(row, season) for row in rows])


def fetch_games(season, timeout=30):
    response = requests.get(
        f"{BASE_URL}/api/games",
        params={"year": season, "limit": 500},
        timeout=timeout,
        headers={
            "User-Agent": (
                "Ultimate-Frisbee-Analytics educational game export "
                "(contact: shownspace.analytics@gmail.com)"
            )
        },
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("games") or []


def find_latest_final_game_id(season, timeout=30):
    games = fetch_games(season, timeout=timeout)
    finals = [
        game for game in games
        if game.get("is_final")
        or str(game.get("Status", "")).strip().lower().startswith("final")
    ]
    if not finals:
        raise RuntimeError(f"No final games found for season {season}.")
    latest = max(finals, key=lambda game: str(game.get("StartTimestamp") or ""))
    return latest["GameID"]


def fetch_game_player_stats(game_id, timeout=30):
    response = requests.get(
        f"{BASE_URL}/api/games/{game_id}",
        timeout=timeout,
        headers={
            "User-Agent": (
                "Ultimate-Frisbee-Analytics educational game export "
                "(contact: shownspace.analytics@gmail.com)"
            )
        },
    )
    response.raise_for_status()
    payload = response.json()
    player_stats = payload.get("playerStats") or []
    frame = pd.DataFrame(player_stats)
    if not frame.empty:
        frame.insert(0, "game_id", game_id)
        game = payload.get("game") or {}
        frame["home_team_id"] = game.get("HomeTeamID")
        frame["away_team_id"] = game.get("AwayTeamID")
        frame["home_score"] = game.get("HomeScore")
        frame["away_score"] = game.get("AwayScore")
        frame["status"] = game.get("Status")
        frame["start_timestamp"] = game.get("StartTimestamp")
    return frame


def add_derived_rates(frame):
    frame = frame.copy()

    if "ThrowAttempts" in frame.columns:
        attempts = pd.to_numeric(frame["ThrowAttempts"], errors="coerce")
    else:
        attempts = pd.Series(np.nan, index=frame.index)

    fallback_attempts = pd.Series(0, index=frame.index, dtype="float")
    for column in ["Completions", "Throwaways", "Stalls", "Drops"]:
        if column in frame.columns:
            fallback_attempts += pd.to_numeric(frame[column], errors="coerce").fillna(0)

    attempts = attempts.fillna(fallback_attempts.replace(0, np.nan))
    frame["throw_attempts_for_rate"] = attempts

    if "thrower_aec" in frame.columns:
        frame["thrower_aec_per_throw"] = np.where(
            attempts > 0,
            pd.to_numeric(frame["thrower_aec"], errors="coerce") / attempts,
            np.nan,
        )

    if "total_aec" in frame.columns:
        frame["total_aec_per_throw"] = np.where(
            attempts > 0,
            pd.to_numeric(frame["total_aec"], errors="coerce") / attempts,
            np.nan,
        )

    if "lag_contribution" in frame.columns:
        frame["lag_contribution_per_throw"] = np.where(
            attempts > 0,
            pd.to_numeric(frame["lag_contribution"], errors="coerce") / attempts,
            np.nan,
        )

    return frame


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the public Shown Space player leaderboard and calculate "
            "derived rates such as T-aEC per throw."
        )
    )
    parser.add_argument(
        "--season",
        default="2026",
        help=(
            "Season to download for leaderboard mode, or the season to search "
            "when using --latest-final."
        ),
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Download all available season leaderboards plus Career.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to data/raw/shownspace_leaderboard_<season>.csv.",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        help="Download Shown Space playerStats for one game instead of the season leaderboard.",
    )
    parser.add_argument(
        "--latest-final",
        action="store_true",
        help="Download playerStats for the latest final game in --season.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between season requests when using --all-seasons.",
    )
    args = parser.parse_args()

    if args.game_id or args.latest_final:
        game_id = args.game_id
        if args.latest_final:
            game_id = find_latest_final_game_id(args.season)
            print(f"Latest final game for {args.season}: {game_id}")

        print(f"Fetching Shown Space game player stats for {game_id}...")
        leaderboard = fetch_game_player_stats(game_id)
        leaderboard = add_derived_rates(leaderboard)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path("data/raw") / f"shownspace_game_{game_id}_player_stats.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(output_path, index=False)
        print(f"Saved {len(leaderboard):,} rows to {output_path}")
        return

    seasons = SEASONS if args.all_seasons else [args.season]
    frames = []
    for index, season in enumerate(seasons):
        print(f"Fetching Shown Space leaderboard for {season}...")
        frames.append(fetch_leaderboard(season))
        if index < len(seasons) - 1 and args.delay > 0:
            time.sleep(args.delay)

    leaderboard = pd.concat(frames, ignore_index=True)
    leaderboard = add_derived_rates(leaderboard)

    if args.output:
        output_path = Path(args.output)
    else:
        season_slug = "all_seasons" if args.all_seasons else str(args.season).lower()
        output_path = Path("data/raw") / f"shownspace_leaderboard_{season_slug}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output_path, index=False)
    print(f"Saved {len(leaderboard):,} rows to {output_path}")


if __name__ == "__main__":
    main()
