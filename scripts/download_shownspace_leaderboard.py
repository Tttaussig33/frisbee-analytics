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


def fetch_game_throws(game_id, timeout=30):
    response = requests.get(
        f"{BASE_URL}/api/games/{game_id}",
        timeout=timeout,
        headers={
            "User-Agent": (
                "Ultimate-Frisbee-Analytics educational game throw export "
                "(contact: shownspace.analytics@gmail.com)"
            )
        },
    )
    response.raise_for_status()
    payload = response.json()
    throws = pd.DataFrame(payload.get("throws") or [])
    if not throws.empty:
        game = payload.get("game") or {}
        throws.insert(0, "game_id", game_id)
        throws["home_team_id"] = game.get("HomeTeamID")
        throws["away_team_id"] = game.get("AwayTeamID")
        throws["home_score_final"] = game.get("HomeScore")
        throws["away_score_final"] = game.get("AwayScore")
        throws["status"] = game.get("Status")
        throws["start_timestamp"] = game.get("StartTimestamp")
    return throws


def fetch_season_throws(season, delay=0.5, timeout=30, cache_path=None, cache_dir=None):
    games = fetch_games(season, timeout=timeout)
    finals = [
        game for game in games
        if game.get("is_final")
        or str(game.get("Status", "")).strip().lower().startswith("final")
    ]

    frames = []
    fetched_game_ids = set()
    cache_path = Path(cache_path) if cache_path else None
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir and cache_dir.exists():
        for file_path in sorted(cache_dir.glob("*.csv")):
            cached_game = pd.read_csv(file_path)
            if not cached_game.empty and "game_id" in cached_game.columns:
                frames.append(cached_game)
                fetched_game_ids.update(cached_game["game_id"].dropna().astype(str))
        if fetched_game_ids:
            print(f"Loaded cached throws for {len(fetched_game_ids):,} games from {cache_dir}")
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if not cached.empty and "game_id" in cached.columns:
            frames.append(cached)
            fetched_game_ids = set(cached["game_id"].dropna().astype(str))
            print(f"Loaded {len(cached):,} cached throws from {cache_path}")

    for index, game in enumerate(finals):
        game_id = game.get("GameID")
        if not game_id:
            continue
        if str(game_id) in fetched_game_ids:
            continue
        print(f"Fetching throws for {game_id} ({index + 1}/{len(finals)})...")
        throws = fetch_game_throws(game_id, timeout=timeout)
        if not throws.empty:
            frames.append(throws)
            fetched_game_ids.add(str(game_id))
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                throws.to_csv(cache_dir / f"{game_id}.csv", index=False)
            elif cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(frames, ignore_index=True).drop_duplicates().to_csv(cache_path, index=False)
        if delay > 0 and index < len(finals) - 1:
            time.sleep(delay)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def build_handler_lag_leaderboard(frame, min_throws=300):
    """Return high-volume throwers for handler-focused lag contribution review."""
    if frame.empty:
        return frame.copy()

    leaderboard = add_derived_rates(frame)
    attempts = pd.to_numeric(leaderboard["throw_attempts_for_rate"], errors="coerce")
    handlers = leaderboard[attempts.ge(min_throws)].copy()

    columns = [
        "season",
        "player",
        "team_id",
        "throw_attempts_for_rate",
        "ThrowAttempts",
        "Completions",
        "Throwaways",
        "Stalls",
        "completion_percentage",
        "xcp",
        "cpoe",
        "lag_contribution_per_throw",
        "lag_contribution",
        "thrower_aec_per_throw",
        "thrower_aec",
        "total_aec_per_throw",
        "total_aec",
        "receiver_aec",
        "YardsThrown",
        "Assists",
        "HockeyAssists",
        "hucks_thrown",
        "huck_attempt_percentage",
    ]
    columns = [column for column in columns if column in handlers.columns]
    return (
        handlers.reindex(columns=columns)
        .sort_values(["lag_contribution_per_throw", "thrower_aec_per_throw"], ascending=False)
        .reset_index(drop=True)
    )


def _truthy(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _offense_team_id(row):
    return row["home_team_id"] if _truthy(row.get("is_home_team")) else row["away_team_id"]


def build_higher_aec_losses(throws):
    """Find games where the losing team generated more summed throw aEC."""
    if throws.empty:
        return pd.DataFrame()

    frame = throws.copy()
    frame["team_id"] = frame.apply(_offense_team_id, axis=1)
    frame["aec"] = pd.to_numeric(frame.get("aec"), errors="coerce").fillna(0)
    frame["home_score_final"] = pd.to_numeric(frame.get("home_score_final"), errors="coerce")
    frame["away_score_final"] = pd.to_numeric(frame.get("away_score_final"), errors="coerce")

    team_aec = (
        frame.groupby(
            [
                "game_id",
                "home_team_id",
                "away_team_id",
                "home_score_final",
                "away_score_final",
                "team_id",
            ],
            dropna=False,
        )["aec"]
        .sum()
        .reset_index(name="team_aec")
    )

    rows = []
    for game_id, game in team_aec.groupby("game_id", dropna=False):
        if game.empty:
            continue
        first = game.iloc[0]
        home_team = first["home_team_id"]
        away_team = first["away_team_id"]
        home_score = first["home_score_final"]
        away_score = first["away_score_final"]
        if pd.isna(home_score) or pd.isna(away_score) or home_score == away_score:
            continue

        winner = home_team if home_score > away_score else away_team
        loser = away_team if home_score > away_score else home_team
        winner_aec = game.loc[game["team_id"].eq(winner), "team_aec"]
        loser_aec = game.loc[game["team_id"].eq(loser), "team_aec"]
        if winner_aec.empty or loser_aec.empty:
            continue

        winner_aec_value = float(winner_aec.iloc[0])
        loser_aec_value = float(loser_aec.iloc[0])
        if loser_aec_value <= winner_aec_value:
            continue

        rows.append(
            {
                "game_id": game_id,
                "losing_team_id": loser,
                "winning_team_id": winner,
                "home_team_id": home_team,
                "away_team_id": away_team,
                "home_score_final": home_score,
                "away_score_final": away_score,
                "loser_aec": loser_aec_value,
                "winner_aec": winner_aec_value,
                "aec_margin": loser_aec_value - winner_aec_value,
                "score_margin": abs(home_score - away_score),
            }
        )

    return pd.DataFrame(rows).sort_values("aec_margin", ascending=False).reset_index(drop=True)


def build_reset_thrower_leaderboard(throws, min_reset_throws=25, leaderboard=None):
    """Rank throwers by aEC on backward reset throws."""
    if throws.empty:
        return pd.DataFrame()

    frame = throws.copy()
    for column in ["ThrowerY", "ReceiverY", "aec", "cp"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["team_id"] = frame.apply(_offense_team_id, axis=1)
    if "turnover" in frame.columns:
        frame["is_turnover"] = frame["turnover"].apply(_truthy)
    else:
        frame["is_turnover"] = False

    receiver = frame.get("Receiver", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    reset_throws = frame[
        frame["Thrower"].notna()
        & receiver.ne("")
        & (frame["ReceiverY"] - frame["ThrowerY"]).lt(0)
    ].copy()
    if reset_throws.empty:
        return pd.DataFrame()

    grouped = (
        reset_throws.groupby(["Thrower", "team_id"], dropna=False)
        .agg(
            reset_throws=("aec", "size"),
            reset_aec=("aec", "sum"),
            reset_aec_per_throw=("aec", "mean"),
            avg_reset_cp=("cp", "mean"),
            avg_reset_yards=("ReceiverY", lambda values: np.nan),
        )
        .reset_index()
    )

    reset_throws["reset_yards"] = reset_throws["ReceiverY"] - reset_throws["ThrowerY"]
    avg_yards = (
        reset_throws.groupby(["Thrower", "team_id"], dropna=False)["reset_yards"]
        .mean()
        .reset_index(name="avg_reset_yards")
    )
    grouped = grouped.drop(columns=["avg_reset_yards"]).merge(avg_yards, on=["Thrower", "team_id"], how="left")
    grouped = grouped[grouped["reset_throws"].ge(min_reset_throws)]
    grouped = grouped.rename(columns={"Thrower": "player_id"})

    if leaderboard is not None and not leaderboard.empty:
        player_lookup_columns = [
            column for column in ["player_id", "player", "team_id"] if column in leaderboard.columns
        ]
        player_lookup = leaderboard.reindex(columns=player_lookup_columns).drop_duplicates(
            subset=["player_id", "team_id"]
        )
        grouped = grouped.merge(player_lookup, on=["player_id", "team_id"], how="left")
    if "player" not in grouped.columns:
        grouped["player"] = grouped["player_id"]

    columns = [
        "player",
        "player_id",
        "team_id",
        "reset_throws",
        "reset_aec",
        "reset_aec_per_throw",
        "avg_reset_cp",
        "avg_reset_yards",
    ]
    columns = [column for column in columns if column in grouped.columns]
    return (
        grouped.reindex(columns=columns)
        .sort_values(["reset_aec_per_throw", "reset_aec"], ascending=False)
        .reset_index(drop=True)
    )


def write_analysis_views(leaderboard, output_dir, season_slug, handler_min_throws=300, throws=None, reset_min_throws=25):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handlers = build_handler_lag_leaderboard(leaderboard, min_throws=handler_min_throws)
    handler_path = output_dir / f"shownspace_handlers_min_{handler_min_throws}_throws_{season_slug}.csv"
    handlers.to_csv(handler_path, index=False)
    print(f"Saved {len(handlers):,} handler rows to {handler_path}")

    if throws is None:
        return

    losses = build_higher_aec_losses(throws)
    losses_path = output_dir / f"shownspace_higher_aec_losses_{season_slug}.csv"
    losses.to_csv(losses_path, index=False)
    print(f"Saved {len(losses):,} higher-aEC loss rows to {losses_path}")

    resets = build_reset_thrower_leaderboard(
        throws,
        min_reset_throws=reset_min_throws,
        leaderboard=leaderboard,
    )
    resets_path = output_dir / f"shownspace_reset_throwers_min_{reset_min_throws}_throws_{season_slug}.csv"
    resets.to_csv(resets_path, index=False)
    print(f"Saved {len(resets):,} reset thrower rows to {resets_path}")


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
        "--input",
        default=None,
        help="Existing leaderboard CSV to load instead of downloading a fresh leaderboard.",
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
    parser.add_argument(
        "--write-analysis-views",
        action="store_true",
        help="Write handler, higher-aEC-loss, and reset-thrower analysis CSVs.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default="data/processed",
        help="Directory for --write-analysis-views outputs.",
    )
    parser.add_argument(
        "--handler-min-throws",
        type=int,
        default=300,
        help="Minimum throw attempts for the handler lag contribution leaderboard.",
    )
    parser.add_argument(
        "--with-throw-analysis",
        action="store_true",
        help="Fetch season throw data for higher-aEC-loss and reset-thrower reports.",
    )
    parser.add_argument(
        "--throw-cache",
        default=None,
        help="Optional CSV path used to cache/resume season throw data for throw-level analysis.",
    )
    parser.add_argument(
        "--throw-cache-dir",
        default=None,
        help="Optional directory of per-game throw CSVs used to cache/resume throw-level analysis.",
    )
    parser.add_argument(
        "--reset-min-throws",
        type=int,
        default=25,
        help="Minimum reset throws for the reset thrower leaderboard.",
    )
    args = parser.parse_args()

    def maybe_fetch_throw_analysis():
        if not args.with_throw_analysis:
            return None
        if args.throw_cache_dir:
            return fetch_season_throws(args.season, delay=args.delay, cache_dir=args.throw_cache_dir)
        return fetch_season_throws(args.season, delay=args.delay, cache_path=args.throw_cache)

    if args.input:
        print(f"Loading existing leaderboard from {args.input}...")
        leaderboard = pd.read_csv(args.input)
        leaderboard = add_derived_rates(leaderboard)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            leaderboard.to_csv(output_path, index=False)
            print(f"Saved {len(leaderboard):,} rows to {output_path}")
        if args.write_analysis_views:
            season_slug = str(args.season).lower()
            throws = maybe_fetch_throw_analysis()
            write_analysis_views(
                leaderboard,
                args.analysis_output_dir,
                season_slug,
                handler_min_throws=args.handler_min_throws,
                throws=throws,
                reset_min_throws=args.reset_min_throws,
            )
        return

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

    if args.write_analysis_views:
        season_slug = "all_seasons" if args.all_seasons else str(args.season).lower()
        throws = maybe_fetch_throw_analysis()
        write_analysis_views(
            leaderboard,
            args.analysis_output_dir,
            season_slug,
            handler_min_throws=args.handler_min_throws,
            throws=throws,
            reset_min_throws=args.reset_min_throws,
        )


if __name__ == "__main__":
    main()
