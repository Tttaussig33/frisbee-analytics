from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from .clean import clean_game_events
    from .client import get_game_events, get_games
    from .etv import add_expected_throwing_value
    from .metrics import (
        calculate_receiver_stats,
        calculate_team_stats,
        calculate_thrower_stats,
    )
except ImportError:
    from clean import clean_game_events
    from client import get_game_events, get_games
    from etv import add_expected_throwing_value
    from metrics import (
        calculate_receiver_stats,
        calculate_team_stats,
        calculate_thrower_stats,
    )


@dataclass
class GamePipelineResult:
    game_id: str
    events: pd.DataFrame
    throws: pd.DataFrame
    thrower_stats: pd.DataFrame
    receiver_stats: pd.DataFrame
    team_stats: pd.DataFrame


def resolve_game_id(date=None, game_id=None, game_index=0):
    if game_id is not None:
        return game_id

    if date is None:
        raise ValueError("Pass either game_id or date.")

    games = get_games(date)
    if games.empty:
        raise ValueError(f"No games found for date {date}.")

    if game_index >= len(games):
        raise IndexError(
            f"game_index {game_index} is out of range for {len(games)} games on {date}."
        )

    return games.loc[game_index, "gameID"]


def build_game_throws(date=None, game_id=None, game_index=0, etv_model=None):
    game_id = resolve_game_id(date=date, game_id=game_id, game_index=game_index)
    events = get_game_events(game_id)
    throws = clean_game_events(events)
    if etv_model is not None:
        throws = add_expected_throwing_value(throws, etv_model)

    return GamePipelineResult(
        game_id=game_id,
        events=events,
        throws=throws,
        thrower_stats=calculate_thrower_stats(throws),
        receiver_stats=calculate_receiver_stats(throws),
        team_stats=calculate_team_stats(throws),
    )


def build_date_throws(date, etv_model=None):
    games = get_games(date)
    if games.empty:
        raise ValueError(f"No games found for date {date}.")

    return [
        build_game_throws(game_id=game_id, etv_model=etv_model)
        for game_id in games["gameID"]
    ]


def save_game_pipeline_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "events": output_dir / f"game_{result.game_id}_events.csv",
        "throws": output_dir / f"game_{result.game_id}_throws.csv",
        "thrower_stats": output_dir / f"game_{result.game_id}_thrower_stats.csv",
        "receiver_stats": output_dir / f"game_{result.game_id}_receiver_stats.csv",
        "team_stats": output_dir / f"game_{result.game_id}_team_stats.csv",
    }

    result.events.to_csv(paths["events"], index=False)
    result.throws.to_csv(paths["throws"], index=False)
    result.thrower_stats.to_csv(paths["thrower_stats"], index=False)
    result.receiver_stats.to_csv(paths["receiver_stats"], index=False)
    result.team_stats.to_csv(paths["team_stats"], index=False)

    return paths
