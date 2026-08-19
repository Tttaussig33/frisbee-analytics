import time
import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://shownspace.com"
FIELD_X_MIN = -26.65
FIELD_X_MAX = 26.65
FIELD_Y_MIN = 0
FIELD_Y_MAX = 120
ENDZONE_LOW_Y = 20
ENDZONE_HIGH_Y = 100


def _filter_regular_season_games(games):
    """Keep games marked as regular season by Shown Space schedule metadata."""
    if games.empty:
        return games

    has_regular_week = "RegularSeasonWeek" in games
    has_playoff_round = "PlayoffRound" in games
    if not has_regular_week and not has_playoff_round:
        return games

    if has_regular_week:
        regular_week = pd.to_numeric(games["RegularSeasonWeek"], errors="coerce")
        regular_mask = regular_week.notna()
    else:
        regular_mask = pd.Series(True, index=games.index)

    if has_playoff_round:
        playoff_round = (
            games["PlayoffRound"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        regular_mask &= playoff_round.isin({"", "none", "nan", "null"})

    return games[regular_mask].reset_index(drop=True)


def fetch_shownspace_games(
    season=2026,
    final_only=True,
    limit=500,
    timeout=30,
    regular_season_only=True,
):
    response = requests.get(
        f"{BASE_URL}/api/games",
        params={"year": season, "limit": limit},
        timeout=timeout,
        headers={"User-Agent": "Ultimate-Frisbee-Analytics educational analysis"},
    )
    response.raise_for_status()
    games = pd.DataFrame(response.json().get("games") or [])
    if games.empty:
        return games

    if final_only:
        status = games.get("Status", pd.Series("", index=games.index)).fillna("")
        is_final = games.get("is_final", pd.Series(False, index=games.index)).fillna(False)
        final_mask = is_final.astype(bool) | status.astype(str).str.strip().str.lower().str.startswith(
            "final"
        )
        games = games[final_mask].reset_index(drop=True)

    if regular_season_only:
        games = _filter_regular_season_games(games)
    return games.reset_index(drop=True)


def fetch_shownspace_game_data(game_id, timeout=30):
    response = requests.get(
        f"{BASE_URL}/api/games/{game_id}",
        timeout=timeout,
        headers={"User-Agent": "Ultimate-Frisbee-Analytics educational analysis"},
    )
    response.raise_for_status()
    return response.json()


def fetch_shownspace_throws_for_games(game_ids, delay=0.15, timeout=30):
    frames = []
    for index, game_id in enumerate(game_ids):
        payload = fetch_shownspace_game_data(game_id, timeout=timeout)
        throws = pd.DataFrame(payload.get("throws") or [])
        if not throws.empty:
            game = payload.get("game") or {}
            throws["game_id"] = game_id
            throws["home_team_id"] = game.get("HomeTeamID")
            throws["away_team_id"] = game.get("AwayTeamID")
            throws["home_score_final"] = game.get("HomeScore")
            throws["away_score_final"] = game.get("AwayScore")
            throws["start_timestamp"] = game.get("StartTimestamp")
            frames.append(throws)
        if delay > 0 and index < len(game_ids) - 1:
            time.sleep(delay)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_shownspace_season_throws(
    season=2026,
    team_id=None,
    max_games=None,
    delay=0.15,
    regular_season_only=True,
):
    games = fetch_shownspace_games(
        season=season,
        final_only=True,
        regular_season_only=regular_season_only,
    )
    if team_id is not None and not games.empty:
        team_id = team_id.lower()
        games = games[
            games["HomeTeamID"].str.lower().eq(team_id)
            | games["AwayTeamID"].str.lower().eq(team_id)
        ].reset_index(drop=True)
    if max_games is not None:
        games = games.head(max_games)

    throws = fetch_shownspace_throws_for_games(games["GameID"].tolist(), delay=delay)
    return games, throws


def _offense_team_id(frame):
    home_team_id = frame["home_team_id"].iloc[0]
    away_team_id = frame["away_team_id"].iloc[0]
    is_home = bool(frame["is_home_team"].iloc[0])
    return home_team_id if is_home else away_team_id


def _possession_line_type(frame):
    if "o_line" not in frame:
        return "unknown"

    values = frame["o_line"].dropna()
    if values.empty:
        return "unknown"

    value = values.iloc[0]
    if isinstance(value, str):
        is_o_line = value.strip().lower() in {"true", "1", "yes", "o_line"}
    else:
        is_o_line = bool(value)
    return "o_line" if is_o_line else "d_line"


def _truthy_value(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _has_text_value(value):
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def _throw_receiver_value(throw):
    for column in ["Receiver", "receiver", "receiver_id"]:
        if column in throw:
            return throw.get(column)
    return None


def _is_turnover_throw(throw):
    if "turnover" in throw:
        return _truthy_value(throw.get("turnover"))
    if "Turnover" in throw:
        return _truthy_value(throw.get("Turnover"))

    receiver = _throw_receiver_value(throw)
    return not _has_text_value(receiver)


def _possession_outcome(path):
    if path.empty:
        return "unknown"

    final_throw = path.iloc[-1]
    if _is_turnover_throw(final_throw):
        return "turnover"

    final_y = pd.to_numeric(pd.Series([final_throw.get("ReceiverY")]), errors="coerce").iloc[0]
    # Shown Space coordinates include the end-zone boundary itself. Scores
    # can finish in either end zone, so treat both boundaries as goals.
    if pd.notna(final_y) and (
        final_y <= ENDZONE_LOW_Y or final_y >= ENDZONE_HIGH_Y
    ):
        return "goal"

    return "unknown"


def _path_points(path):
    path = path.sort_values("possession_throw")
    if path.empty:
        return pd.DataFrame(
            columns=["x", "y", "aec", "cumulative_aec", "cp", "win_prob"]
        )

    cumulative_aec = 0.0
    points = [
        {
            "x": path["ThrowerX"].iloc[0],
            "y": path["ThrowerY"].iloc[0],
            "aec": 0.0,
            "cumulative_aec": 0.0,
            "cp": np.nan,
            "win_prob": path["win_prob"].iloc[0] if "win_prob" in path else np.nan,
        }
    ]
    for _, throw in path.iterrows():
        throw_aec = throw.get("aec", np.nan)
        if pd.notna(throw_aec):
            cumulative_aec += throw_aec
        points.append(
            {
                "x": throw["ReceiverX"],
                "y": throw["ReceiverY"],
                "aec": throw_aec,
                "cumulative_aec": cumulative_aec,
                "cp": throw.get("cp", np.nan),
                "win_prob": throw.get("win_prob", np.nan),
            }
        )
    return pd.DataFrame(points)


def _resample_path(points, checkpoints):
    points = points.dropna(subset=["x", "y"]).copy()
    if points.empty:
        return pd.DataFrame()

    progress = points["y"].to_numpy(dtype=float)
    if progress[-1] == progress[0]:
        normalized = np.linspace(0, 1, len(points))
    else:
        normalized = (progress - progress[0]) / (progress[-1] - progress[0])
    normalized = np.maximum.accumulate(np.clip(normalized, 0, 1))

    dedup = pd.DataFrame(
        {
            "progress": normalized,
            "x": points["x"].to_numpy(dtype=float),
            "y": points["y"].to_numpy(dtype=float),
            "aec": points["aec"].to_numpy(dtype=float),
            "cumulative_aec": points["cumulative_aec"].to_numpy(dtype=float),
            "cp": points["cp"].to_numpy(dtype=float),
            "win_prob": points["win_prob"].to_numpy(dtype=float),
        }
    ).drop_duplicates("progress", keep="last")

    if len(dedup) == 1:
        x_values = np.repeat(dedup["x"].iloc[0], len(checkpoints))
        y_values = np.repeat(dedup["y"].iloc[0], len(checkpoints))
    else:
        x_values = np.interp(checkpoints, dedup["progress"], dedup["x"])
        y_values = np.interp(checkpoints, dedup["progress"], dedup["y"])

    return pd.DataFrame(
        {
            "checkpoint": checkpoints,
            "x": x_values,
            "y": y_values,
            "cumulative_aec": np.interp(
                checkpoints,
                dedup["progress"],
                dedup["cumulative_aec"],
            ),
            "cp": np.interp(
                checkpoints,
                dedup["progress"],
                dedup["cp"].ffill().bfill(),
            ),
            "win_prob": np.interp(
                checkpoints,
                dedup["progress"],
                dedup["win_prob"].ffill().bfill(),
            ),
        }
    )


def _path_lookup(paths):
    return {path["possession_id"].iloc[0]: path for path in paths if not path.empty}


def _style_features(possessions):
    feature_columns = [
        "throw_count",
        "aec_per_throw",
        "mean_cp",
        "yards_per_throw",
        "max_throw_distance",
        "huck_count",
        "reset_count",
        "lateral_yards",
    ]
    return (
        possessions.reindex(columns=feature_columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def _path_shape_feature_row(path, checkpoints):
    path = path.sort_values("possession_throw").copy()
    points = _path_points(path).dropna(subset=["x", "y"])
    if points.empty:
        return {}

    sampled = _resample_path(points, checkpoints)
    if sampled.empty:
        return {}

    x_values = points["x"].to_numpy(dtype=float)
    y_values = points["y"].to_numpy(dtype=float)
    thrower_x = pd.to_numeric(path["ThrowerX"], errors="coerce")
    receiver_x = pd.to_numeric(path["ReceiverX"], errors="coerce")
    receiver_y = pd.to_numeric(path["ReceiverY"], errors="coerce")
    x_diff = pd.to_numeric(path.get("x_diff"), errors="coerce")
    y_diff = pd.to_numeric(path.get("y_diff"), errors="coerce")
    throw_distance = pd.to_numeric(path.get("throw_distance"), errors="coerce")

    total_distance = throw_distance.sum()
    net_distance = float(
        np.hypot(x_values[-1] - x_values[0], y_values[-1] - y_values[0])
    )
    directness = net_distance / total_distance if total_distance else np.nan
    field_progress = y_values[-1] - y_values[0]
    lateral_yards = x_diff.abs().sum()

    point_x = pd.Series(np.r_[thrower_x.to_numpy(), receiver_x.to_numpy()])
    point_y = pd.Series(np.r_[path["ThrowerY"].to_numpy(), receiver_y.to_numpy()])
    valid_points = pd.DataFrame({"x": point_x, "y": point_y}).dropna()
    middle_third_share = valid_points["x"].abs().le(8.88).mean()
    sideline_share = valid_points["x"].abs().ge(17.77).mean()
    left_side_share = valid_points["x"].lt(-8.88).mean()
    right_side_share = valid_points["x"].gt(8.88).mean()

    signs = np.sign(x_values)
    non_zero_signs = signs[signs != 0]
    side_switches = (
        np.count_nonzero(non_zero_signs[1:] != non_zero_signs[:-1])
        if len(non_zero_signs) > 1
        else 0
    )

    red_zone_rows = path[receiver_y.ge(ENDZONE_HIGH_Y - 20)]
    red_zone_entry_x = (
        pd.to_numeric(red_zone_rows["ReceiverX"], errors="coerce").iloc[0]
        if not red_zone_rows.empty
        else np.nan
    )

    features = {
        "shape_start_x": x_values[0],
        "shape_start_y": y_values[0],
        "shape_end_x": x_values[-1],
        "shape_end_y": y_values[-1],
        "shape_width": np.nanmax(x_values) - np.nanmin(x_values),
        "shape_directness": directness,
        "shape_lateral_per_yard": lateral_yards / abs(field_progress)
        if field_progress
        else np.nan,
        "shape_side_switches": side_switches,
        "shape_middle_third_share": middle_third_share,
        "shape_sideline_share": sideline_share,
        "shape_left_side_share": left_side_share,
        "shape_right_side_share": right_side_share,
        "shape_red_zone_entry_x": red_zone_entry_x,
        "shape_red_zone_throws": receiver_y.ge(ENDZONE_HIGH_Y - 20).sum(),
        "shape_backwards_share": y_diff.lt(0).mean(),
        "shape_large_gain_share": throw_distance.ge(25).mean(),
    }
    for _, point in sampled.iterrows():
        label = int(round(point["checkpoint"] * 100))
        features[f"shape_x_{label:03d}"] = point["x"]
        features[f"shape_y_{label:03d}"] = point["y"]

    return features


def calculate_possession_shape_features(possessions, paths, checkpoints=None):
    """Return one row of geometry-first features per scoring possession."""
    if possessions.empty:
        return pd.DataFrame()

    checkpoints = np.asarray(
        checkpoints if checkpoints is not None else np.linspace(0, 1, 8),
        dtype=float,
    )
    rows = []
    for path in paths:
        if path.empty or "possession_id" not in path:
            continue
        possession_id = path["possession_id"].iloc[0]
        feature_row = _path_shape_feature_row(path, checkpoints)
        if feature_row:
            feature_row["possession_id"] = possession_id
            rows.append(feature_row)

    if not rows:
        return pd.DataFrame(index=possessions.index)

    shape_features = pd.DataFrame(rows).set_index("possession_id")
    aligned = possessions[["possession_id"]].join(shape_features, on="possession_id")
    return (
        aligned.drop(columns=["possession_id"])
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def add_possession_shape_features(possessions, paths, checkpoints=None):
    """Attach geometry-first possession features to the possession table."""
    if possessions.empty:
        return possessions.copy()

    enriched = possessions.copy()
    shape_features = calculate_possession_shape_features(
        enriched, paths, checkpoints=checkpoints
    )
    for column in shape_features:
        enriched[column] = shape_features[column].to_numpy()
    return enriched


def build_possessions(throws, team_id=None, outcomes=None):
    """Build possession-level rows and paths from Shown Space throw data.

    Parameters
    ----------
    throws:
        Throw-level Shown Space data.
    team_id:
        Optional offensive team id filter.
    outcomes:
        Optional iterable of outcomes to keep. Supported labels are
        ``goal``, ``turnover``, and ``unknown``. ``None`` keeps all outcomes.
    """
    if throws.empty:
        return pd.DataFrame(), []

    outcome_filter = None
    if outcomes is not None:
        outcome_filter = {str(outcome).lower() for outcome in outcomes}

    throws = throws.copy()
    for receiver_column, turnover_candidates in {
        "ReceiverX": ["TurnoverX", "turnoverX", "turnover_x"],
        "ReceiverY": ["TurnoverY", "turnoverY", "turnover_y"],
    }.items():
        if receiver_column not in throws:
            continue
        for turnover_column in turnover_candidates:
            if turnover_column in throws:
                throws[receiver_column] = throws[receiver_column].fillna(
                    throws[turnover_column]
                )
                break

    throws = throws.dropna(
        subset=["ThrowerX", "ThrowerY", "ReceiverX", "ReceiverY", "possession_throw"]
    )
    group_columns = [
        "GameID",
        "game_quarter",
        "quarter_point",
        "possession_num",
        "is_home_team",
    ]

    possession_rows = []
    paths = []
    for key, group in throws.groupby(group_columns, dropna=False):
        path = group.sort_values("possession_throw").copy()
        outcome = _possession_outcome(path)
        if outcome_filter is not None and outcome not in outcome_filter:
            continue

        offense_team = _offense_team_id(path)
        if team_id is not None and offense_team.lower() != team_id.lower():
            continue

        line_type = _possession_line_type(path)
        total_aec = pd.to_numeric(path.get("aec"), errors="coerce").sum()
        throw_count = len(path)
        start_timestamp = path["start_timestamp"].iloc[0] if "start_timestamp" in path else None
        start_x = pd.to_numeric(path["ThrowerX"], errors="coerce").iloc[0]
        start_y = pd.to_numeric(path["ThrowerY"], errors="coerce").iloc[0]
        end_x = pd.to_numeric(path["ReceiverX"], errors="coerce").iloc[-1]
        end_y = pd.to_numeric(path["ReceiverY"], errors="coerce").iloc[-1]
        x_diff = pd.to_numeric(path.get("x_diff"), errors="coerce")
        y_diff = pd.to_numeric(path.get("y_diff"), errors="coerce")
        throw_distance = pd.to_numeric(path.get("throw_distance"), errors="coerce")
        cp = pd.to_numeric(path.get("cp"), errors="coerce")

        possession_id = "|".join(str(value) for value in key)
        possession_rows.append(
            {
                "possession_id": possession_id,
                "GameID": key[0],
                "team_id": offense_team,
                "start_timestamp": start_timestamp,
                "game_quarter": key[1],
                "quarter_point": key[2],
                "possession_num": key[3],
                "is_home_team": key[4],
                "line_type": line_type,
                "outcome": outcome,
                "is_goal": outcome == "goal",
                "is_turnover": outcome == "turnover",
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "field_progress": end_y - start_y,
                "throw_count": throw_count,
                "total_aec": total_aec,
                "aec_per_throw": total_aec / throw_count if throw_count else np.nan,
                "mean_cp": cp.mean(),
                "risk_adjusted_aec_per_throw": (
                    total_aec / throw_count * cp.mean() if throw_count else np.nan
                ),
                "total_yards": y_diff.sum(),
                "yards_per_throw": y_diff.mean(),
                "total_throw_distance": throw_distance.sum(),
                "avg_throw_distance": throw_distance.mean(),
                "max_throw_distance": throw_distance.max(),
                "huck_count": (throw_distance >= 40).sum(),
                "reset_count": (y_diff < 0).sum(),
                "lateral_yards": x_diff.abs().sum(),
            }
        )
        path["possession_id"] = possession_id
        path["team_id"] = offense_team
        path["line_type"] = line_type
        path["outcome"] = outcome
        paths.append(path)

    possessions = pd.DataFrame(possession_rows)
    return possessions, paths


def build_scoring_possessions(throws, team_id=None):
    return build_possessions(throws, team_id=team_id, outcomes=("goal",))


def add_possession_style_labels(possessions):
    """Add simple, readable style labels to scoring possessions."""
    if possessions.empty:
        return possessions.copy()

    labeled = possessions.copy()
    conditions = [
        labeled["huck_count"].fillna(0).ge(1),
        labeled["throw_count"].fillna(0).le(3),
        labeled["reset_count"].fillna(0).ge(3),
        labeled["throw_count"].fillna(0).ge(8),
    ]
    choices = ["huck score", "quick strike", "reset-heavy", "methodical"]
    labeled["style"] = np.select(conditions, choices, default="mixed")
    return labeled


def _shape_cluster_features(possessions):
    shape_columns = [column for column in possessions if column.startswith("shape_")]
    if not shape_columns:
        return _style_features(possessions)
    support_columns = [
        "huck_count",
        "reset_count",
        "lateral_yards",
        "max_throw_distance",
        "yards_per_throw",
    ]
    feature_columns = shape_columns + [
        column for column in support_columns if column in possessions
    ]
    return (
        possessions.reindex(columns=feature_columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def cluster_scoring_possessions(
    possessions,
    paths=None,
    n_clusters=4,
    random_state=0,
):
    """Cluster scoring possessions by geometry-first path shape features."""
    if possessions.empty:
        return possessions.copy()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    clustered = add_possession_style_labels(possessions)
    if paths is not None:
        clustered = add_possession_shape_features_if_available(clustered, paths)
    features = _shape_cluster_features(clustered)
    cluster_count = min(n_clusters, len(clustered))
    if cluster_count <= 1:
        clustered["path_cluster"] = 0
        return clustered

    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init="auto")
    clustered["path_cluster"] = model.fit_predict(scaled)
    return clustered


def add_possession_shape_features_if_available(possessions, paths):
    """Add path-shape features, preserving clustering if paths are unavailable."""
    if paths is None:
        return possessions.copy()
    enriched = add_possession_shape_features(possessions, paths)
    shape_columns = [column for column in enriched if column.startswith("shape_")]
    if not shape_columns:
        return possessions.copy()
    return enriched


def summarize_path_clusters(possessions):
    if possessions.empty:
        return pd.DataFrame()

    aggregations = {
        "possessions": ("possession_id", "count"),
        "avg_throws": ("throw_count", "mean"),
        "avg_aec_per_throw": ("aec_per_throw", "mean"),
        "avg_cp": ("mean_cp", "mean"),
        "avg_yards_per_throw": ("yards_per_throw", "mean"),
        "avg_max_throw_distance": ("max_throw_distance", "mean"),
        "avg_resets": ("reset_count", "mean"),
        "avg_lateral_yards": ("lateral_yards", "mean"),
    }
    optional_shape_aggs = {
        "avg_width": ("shape_width", "mean"),
        "avg_directness": ("shape_directness", "mean"),
        "avg_side_switches": ("shape_side_switches", "mean"),
        "avg_middle_third_share": ("shape_middle_third_share", "mean"),
        "avg_sideline_share": ("shape_sideline_share", "mean"),
        "avg_red_zone_entry_x": ("shape_red_zone_entry_x", "mean"),
    }
    for output_column, (input_column, function_name) in optional_shape_aggs.items():
        if input_column in possessions:
            aggregations[output_column] = (input_column, function_name)

    summary = (
        possessions.groupby(["path_cluster", "style"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["possessions", "avg_aec_per_throw"], ascending=[False, False])
    )
    return summary


PLAYSTYLE_SUMMARY_COLUMNS = [
    "team_id",
    "possessions",
    "primary_shapes",
    "attack_spaces",
    "pace_style",
    "field_width_style",
    "huck_usage",
    "reset_usage",
    "efficiency_note",
    "playstyle_summary",
]


def _mean_numeric(frame, column):
    if column not in frame:
        return np.nan
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).mean()


def _count_numeric(frame, column):
    if column not in frame:
        return 0
    return pd.to_numeric(frame[column], errors="coerce").notna().sum()


def _prepare_playstyle_possessions(
    possessions,
    paths=None,
    n_shape_clusters=8,
    random_state=0,
):
    if possessions.empty:
        return possessions.copy()

    prepared = possessions.copy()
    if paths is not None:
        prepared = add_possession_shape_features_if_available(prepared, paths)

    if "path_cluster" not in prepared:
        prepared = cluster_scoring_possessions(
            prepared,
            paths,
            n_clusters=n_shape_clusters,
            random_state=random_state,
        )
    elif "style" not in prepared:
        prepared = add_possession_style_labels(prepared)

    if "style" not in prepared:
        prepared = add_possession_style_labels(prepared)

    return _add_browser_shape_cluster_labels(prepared)


def _top_shape_text(possessions, limit=3):
    if possessions.empty:
        return "no scoring shapes"

    label_column = "shape_cluster_label" if "shape_cluster_label" in possessions else "style"
    if label_column not in possessions:
        return "unlabeled scoring shapes"

    counts = (
        possessions[label_column]
        .fillna("Unlabeled")
        .astype(str)
        .value_counts()
        .head(limit)
    )
    return ", ".join(f"{label} ({int(count)})" for label, count in counts.items())


def _attack_spaces_text(stats):
    middle = stats["avg_middle_usage"]
    sideline = stats["avg_sideline_usage"]
    left = stats["avg_left_usage"]
    right = stats["avg_right_usage"]

    if pd.notna(middle) and middle >= 0.55 and (
        pd.isna(sideline) or sideline < 0.25
    ):
        lane_text = "attack the middle third often"
    elif pd.notna(sideline) and sideline >= 0.25 and (
        pd.isna(middle) or middle < 0.55
    ):
        lane_text = "work through sideline channels often"
    elif (
        pd.notna(middle)
        and pd.notna(sideline)
        and middle >= 0.50
        and sideline >= 0.22
    ):
        lane_text = "mix middle-lane attacks with sideline channels"
    else:
        lane_text = "use a balanced mix of field spaces"

    side_text = ""
    if pd.notna(left) and pd.notna(right):
        if left > right + 0.12:
            side_text = " with a left-side lean"
        elif right > left + 0.12:
            side_text = " with a right-side lean"

    return f"{lane_text}{side_text}"


def _pace_style_text(stats):
    avg_throws = stats["avg_throws"]
    directness = stats["avg_directness"]

    if pd.notna(avg_throws) and avg_throws <= 4:
        tempo = "quick-strike"
    elif pd.notna(avg_throws) and avg_throws >= 10:
        tempo = "methodical"
    else:
        tempo = "moderate-tempo"

    if pd.notna(directness) and directness >= 0.75:
        shape = "direct"
    elif pd.notna(directness) and directness <= 0.50:
        shape = "winding"
    else:
        shape = "mixed-directness"

    return f"{tempo}, {shape}"


def _field_width_style_text(stats):
    width = stats["avg_width"]
    side_switches = stats["avg_side_switches"]

    if pd.notna(width) and width <= 18:
        return "narrow field usage"
    if pd.notna(width) and width >= 34:
        if pd.notna(side_switches) and side_switches >= 3:
            return "wide and switch-heavy"
        return "wide field usage"
    if pd.notna(side_switches) and side_switches >= 3:
        return "switch-heavy with moderate width"
    return "moderate field width"


def _huck_usage_text(stats):
    hucks = stats["avg_hucks"]
    huck_rate = stats["huck_possession_rate"]
    if pd.notna(hucks) and (hucks >= 0.5 or huck_rate >= 0.35):
        return "lean on huck or large-gain scoring patterns"
    if pd.notna(hucks) and (hucks >= 0.25 or huck_rate >= 0.18):
        return "use hucks as a regular scoring option"
    return "do not rely heavily on hucks"


def _reset_usage_text(stats):
    resets = stats["avg_resets"]
    if pd.notna(resets) and resets >= 3:
        return "are comfortable extending possessions with resets"
    if pd.notna(resets) and resets >= 1.5:
        return "use resets at a moderate rate"
    return "keep reset volume relatively low"


def _efficiency_note_text(stats):
    aec_per_throw = stats["avg_aec_per_throw"]
    if pd.isna(aec_per_throw):
        return "aEC per throw is unavailable"
    if aec_per_throw >= 0.14:
        return "high-value scoring possessions by aEC per throw"
    if aec_per_throw >= 0.09:
        return "solid scoring value by aEC per throw"
    if aec_per_throw >= 0.05:
        return "moderate scoring value by aEC per throw"
    return "lower aEC per throw despite scoring"


def _team_name(team_id):
    if team_id is None or pd.isna(team_id):
        return "This team"
    return str(team_id).strip().title()


def summarize_team_playstyle(
    possessions,
    paths=None,
    team_id=None,
    n_shape_clusters=8,
    random_state=0,
):
    """Summarize one team's scoring-possession playstyle with rule-based text."""
    if possessions.empty:
        return pd.Series(
            {
                "team_id": team_id,
                "possessions": 0,
                "primary_shapes": "no scoring possessions",
                "attack_spaces": "no scoring possessions",
                "pace_style": "no scoring possessions",
                "field_width_style": "no scoring possessions",
                "huck_usage": "no scoring possessions",
                "reset_usage": "no scoring possessions",
                "efficiency_note": "no scoring possessions",
                "playstyle_summary": "No scoring possessions are available for this team.",
            }
        )

    frame = possessions.copy()
    if team_id is not None and "team_id" in frame:
        frame = frame[frame["team_id"].str.lower().eq(str(team_id).lower())].copy()
    if frame.empty:
        return summarize_team_playstyle(pd.DataFrame(), team_id=team_id)

    if team_id is None and "team_id" in frame and frame["team_id"].nunique() == 1:
        team_id = frame["team_id"].iloc[0]

    team_paths = paths
    if paths is not None:
        possession_ids = set(frame["possession_id"])
        team_paths = [
            path
            for path in paths
            if not path.empty and path["possession_id"].iloc[0] in possession_ids
        ]

    prepared = _prepare_playstyle_possessions(
        frame,
        team_paths,
        n_shape_clusters=n_shape_clusters,
        random_state=random_state,
    )

    huck_counts = pd.to_numeric(
        prepared.get("huck_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    stats = {
        "team_id": team_id,
        "possessions": int(len(prepared)),
        "avg_throws": _mean_numeric(prepared, "throw_count"),
        "avg_width": _mean_numeric(prepared, "shape_width"),
        "avg_side_switches": _mean_numeric(prepared, "shape_side_switches"),
        "avg_directness": _mean_numeric(prepared, "shape_directness"),
        "avg_middle_usage": _mean_numeric(prepared, "shape_middle_third_share"),
        "avg_sideline_usage": _mean_numeric(prepared, "shape_sideline_share"),
        "avg_left_usage": _mean_numeric(prepared, "shape_left_side_share"),
        "avg_right_usage": _mean_numeric(prepared, "shape_right_side_share"),
        "avg_hucks": _mean_numeric(prepared, "huck_count"),
        "avg_resets": _mean_numeric(prepared, "reset_count"),
        "avg_aec_per_throw": _mean_numeric(prepared, "aec_per_throw"),
        "avg_cp": _mean_numeric(prepared, "mean_cp"),
        "huck_possession_rate": huck_counts.gt(0).mean() if len(prepared) else np.nan,
        "shape_metric_possessions": _count_numeric(prepared, "shape_directness"),
    }

    primary_shapes = _top_shape_text(prepared)
    attack_spaces = _attack_spaces_text(stats)
    pace_style = _pace_style_text(stats)
    field_width_style = _field_width_style_text(stats)
    huck_usage = _huck_usage_text(stats)
    reset_usage = _reset_usage_text(stats)
    efficiency_note = _efficiency_note_text(stats)

    name = _team_name(team_id)
    summary = (
        f"{name}'s scoring possessions are generally {pace_style}. "
        f"They {attack_spaces}. Their most common shape groups are {primary_shapes}. "
        f"They {huck_usage} and {reset_usage}, with {field_width_style}. "
        f"Overall, the scoring possessions grade as {efficiency_note}."
    )

    stats.update(
        {
            "primary_shapes": primary_shapes,
            "attack_spaces": attack_spaces,
            "pace_style": pace_style,
            "field_width_style": field_width_style,
            "huck_usage": huck_usage,
            "reset_usage": reset_usage,
            "efficiency_note": efficiency_note,
            "playstyle_summary": summary,
        }
    )
    return pd.Series(stats)


def summarize_team_playstyles(
    possessions,
    paths=None,
    team_ids=None,
    n_shape_clusters=8,
    random_state=0,
):
    """Return one rule-based playstyle summary row per team."""
    if possessions.empty:
        return pd.DataFrame(columns=PLAYSTYLE_SUMMARY_COLUMNS)

    if team_ids is None:
        if "team_id" in possessions:
            team_ids = sorted(
                team for team in possessions["team_id"].dropna().astype(str).unique()
            )
        else:
            team_ids = [None]

    rows = [
        summarize_team_playstyle(
            possessions,
            paths=paths,
            team_id=team,
            n_shape_clusters=n_shape_clusters,
            random_state=random_state,
        )
        for team in team_ids
    ]
    table = pd.DataFrame(rows)
    requested = [column for column in PLAYSTYLE_SUMMARY_COLUMNS if column in table]
    remaining = [column for column in table.columns if column not in requested]
    return table[requested + remaining]


def _format_playstyle_metric(value, digits=1, percent=False):
    if value is None or pd.isna(value):
        return "-"
    if percent:
        return f"{float(value):.{digits}%}"
    return f"{float(value):.{digits}f}"


def _shorten_text(value, max_length=96):
    text = "" if value is None or pd.isna(value) else str(value)
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1]}..."


def render_team_playstyle_report(
    team_playstyle,
    team_playstyle_table=None,
    title="Team playstyle report",
):
    """Return browser-style HTML for one-team and comparison playstyle summaries."""
    row = (
        team_playstyle.to_dict()
        if hasattr(team_playstyle, "to_dict")
        else dict(team_playstyle)
    )
    team_name = _team_name(row.get("team_id"))
    possessions = int(row.get("possessions", 0) or 0)
    summary = escape(str(row.get("playstyle_summary", "")))

    metric_rows = [
        ("Possessions", f"{possessions:,}"),
        ("Primary shapes", escape(str(row.get("primary_shapes", "-")))),
        ("Attack spaces", escape(str(row.get("attack_spaces", "-")))),
        ("Pace", escape(str(row.get("pace_style", "-")))),
        ("Width", escape(str(row.get("field_width_style", "-")))),
        ("Hucks", escape(str(row.get("huck_usage", "-")))),
        ("Resets", escape(str(row.get("reset_usage", "-")))),
        ("Efficiency", escape(str(row.get("efficiency_note", "-")))),
        ("Avg throws", _format_playstyle_metric(row.get("avg_throws"))),
        ("Avg width", _format_playstyle_metric(row.get("avg_width"))),
        ("Side switches", _format_playstyle_metric(row.get("avg_side_switches"))),
        ("Directness", _format_playstyle_metric(row.get("avg_directness"), percent=True)),
        ("Middle usage", _format_playstyle_metric(row.get("avg_middle_usage"), percent=True)),
        ("Sideline usage", _format_playstyle_metric(row.get("avg_sideline_usage"), percent=True)),
        ("Avg hucks", _format_playstyle_metric(row.get("avg_hucks"))),
        ("Avg resets", _format_playstyle_metric(row.get("avg_resets"))),
        ("aEC / throw", _format_playstyle_metric(row.get("avg_aec_per_throw"), digits=3)),
        ("Avg CP", _format_playstyle_metric(row.get("avg_cp"), percent=True)),
    ]

    metric_html = "".join(
        "<div class='ufa-playstyle-row'>"
        f"<span>{label}</span><b>{value}</b>"
        "</div>"
        for label, value in metric_rows
    )

    comparison_html = ""
    if team_playstyle_table is not None and not pd.DataFrame(team_playstyle_table).empty:
        table = pd.DataFrame(team_playstyle_table).copy()
        table_rows = []
        for _, table_row in table.iterrows():
            table_rows.append(
                "<tr>"
                f"<td>{escape(_team_name(table_row.get('team_id')))}</td>"
                f"<td>{int(table_row.get('possessions', 0) or 0):,}</td>"
                f"<td>{escape(_shorten_text(table_row.get('primary_shapes'), 90))}</td>"
                f"<td>{escape(_shorten_text(table_row.get('attack_spaces'), 75))}</td>"
                f"<td>{escape(str(table_row.get('pace_style', '-')))}</td>"
                f"<td>{escape(str(table_row.get('field_width_style', '-')))}</td>"
                f"<td>{_format_playstyle_metric(table_row.get('avg_aec_per_throw'), digits=3)}</td>"
                f"<td>{_format_playstyle_metric(table_row.get('avg_directness'), percent=True)}</td>"
                f"<td>{_format_playstyle_metric(table_row.get('avg_middle_usage'), percent=True)}</td>"
                "</tr>"
            )

        comparison_html = f"""
        <div class="ufa-playstyle-panel ufa-playstyle-comparison">
          <h3>Team comparison</h3>
          <div class="ufa-playstyle-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Team</th><th>N</th><th>Primary shapes</th>
                  <th>Attack spaces</th><th>Pace</th><th>Width</th>
                  <th>aEC/T</th><th>Dir</th><th>Mid</th>
                </tr>
              </thead>
              <tbody>{''.join(table_rows)}</tbody>
            </table>
          </div>
        </div>
        """

    return f"""
    <div class="ufa-playstyle-browser">
      <style>
        .ufa-playstyle-browser {{
          color: #0b1a33;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1180px;
        }}
        .ufa-playstyle-browser h2 {{
          margin: 0 0 12px;
          color: #223a5e;
          font-size: 22px;
        }}
        .ufa-playstyle-layout {{
          display: grid;
          grid-template-columns: minmax(320px, 420px) minmax(520px, 1fr);
          gap: 18px;
          align-items: start;
        }}
        .ufa-playstyle-panel {{
          border: 1px solid #d9e1ea;
          background: #fbfdff;
          border-radius: 4px;
          padding: 14px;
          box-sizing: border-box;
        }}
        .ufa-playstyle-panel h3 {{
          margin: 0 0 8px;
          color: #0b1a33;
          font-size: 18px;
        }}
        .ufa-playstyle-summary {{
          color: #263a58;
          line-height: 1.45;
          margin: 8px 0 12px;
        }}
        .ufa-playstyle-metrics {{
          border-top: 1px solid #d9e1ea;
          padding-top: 8px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
          font-size: 12px;
        }}
        .ufa-playstyle-row {{
          display: flex;
          justify-content: space-between;
          gap: 18px;
          border-bottom: 1px solid #edf1f5;
          padding: 4px 0;
        }}
        .ufa-playstyle-row span {{ color: #637188; }}
        .ufa-playstyle-row b {{ text-align: right; }}
        .ufa-playstyle-comparison {{
          overflow: hidden;
        }}
        .ufa-playstyle-table-wrap {{
          overflow-x: auto;
          max-height: 520px;
        }}
        .ufa-playstyle-comparison table {{
          border-collapse: collapse;
          min-width: 920px;
          width: 100%;
          font-size: 12px;
        }}
        .ufa-playstyle-comparison th {{
          position: sticky;
          top: 0;
          background: #e9eef5;
          color: #223a5e;
          text-align: left;
          padding: 7px 8px;
          border-bottom: 1px solid #d9e1ea;
        }}
        .ufa-playstyle-comparison td {{
          padding: 7px 8px;
          border-bottom: 1px solid #edf1f5;
          vertical-align: top;
        }}
        .ufa-playstyle-comparison tr:nth-child(even) td {{
          background: #f6f9fc;
        }}
      </style>
      <h2>{escape(str(title))}</h2>
      <div class="ufa-playstyle-layout">
        <div class="ufa-playstyle-panel">
          <h3>{escape(team_name)} scoring profile</h3>
          <div class="ufa-playstyle-summary">{summary}</div>
          <div class="ufa-playstyle-metrics">{metric_html}</div>
        </div>
        {comparison_html}
      </div>
    </div>
    """


def create_team_playstyle_report_browser(
    team_playstyle,
    team_playstyle_table=None,
    title="Team playstyle report",
):
    """Create a notebook HTML widget for rule-based team playstyle summaries."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError(
            "ipywidgets is required for create_team_playstyle_report_browser. "
            "Install it with `pip install ipywidgets` and restart the notebook kernel."
        ) from exc

    return widgets.HTML(
        render_team_playstyle_report(
            team_playstyle,
            team_playstyle_table=team_playstyle_table,
            title=title,
        )
    )


def select_top_paths(possessions, paths, metric="aec_per_throw", n=5, ascending=False):
    """Return real possession paths ranked by a possession-level metric."""
    if possessions.empty:
        return []

    lookup = _path_lookup(paths)
    ranked = possessions.sort_values(metric, ascending=ascending).head(n)
    return [
        lookup[possession_id]
        for possession_id in ranked["possession_id"]
        if possession_id in lookup
    ]


def select_representative_paths(
    possessions,
    paths,
    group_column="path_cluster",
    unique_games=False,
):
    """Pick one real possession nearest each group's feature median."""
    if possessions.empty or group_column not in possessions:
        return {}

    lookup = _path_lookup(paths)
    features = _shape_cluster_features(add_possession_shape_features_if_available(possessions, paths))
    normalized = (features - features.mean()) / features.std(ddof=0).replace(0, 1)
    normalized = normalized.fillna(0)

    representatives = {}
    used_games = set()
    for group_value, group in possessions.groupby(group_column, dropna=False):
        group_features = normalized.loc[group.index]
        center = group_features.median()
        distances = ((group_features - center) ** 2).sum(axis=1)
        ranked_indices = distances.sort_values().index
        chosen_index = ranked_indices[0]
        if unique_games and "GameID" in possessions:
            for candidate_index in ranked_indices:
                candidate_game = possessions.loc[candidate_index, "GameID"]
                if candidate_game not in used_games:
                    chosen_index = candidate_index
                    break
        chosen = possessions.loc[chosen_index]
        possession_id = chosen["possession_id"]
        if possession_id in lookup:
            if unique_games and "GameID" in chosen:
                used_games.add(chosen["GameID"])
            label = f"{group_column} {group_value}"
            if "style" in chosen:
                label = f"{chosen['style']} ({label})"
            representatives[label] = lookup[possession_id]
    return representatives


def average_scoring_path(paths, checkpoints=None):
    checkpoints = np.asarray(
        checkpoints if checkpoints is not None else np.linspace(0, 1, 6),
        dtype=float,
    )
    resampled = []
    for path in paths:
        sampled = _resample_path(_path_points(path), checkpoints)
        if not sampled.empty:
            sampled["possession_id"] = path["possession_id"].iloc[0]
            resampled.append(sampled)

    if not resampled:
        return pd.DataFrame()

    all_points = pd.concat(resampled, ignore_index=True)
    return (
        all_points
        .groupby("checkpoint")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            mean_cumulative_aec=("cumulative_aec", "mean"),
            mean_cp=("cp", "mean"),
            mean_win_prob=("win_prob", "mean"),
            possessions=("possession_id", "nunique"),
        )
        .reset_index()
    )


def _path_hover_text(path):
    thrower = path.get("thrower")
    receiver = path.get("receiver")
    thrower = thrower if thrower is not None else pd.Series("", index=path.index)
    receiver = receiver if receiver is not None else pd.Series("", index=path.index)
    aec = pd.to_numeric(path.get("aec"), errors="coerce")
    cp = pd.to_numeric(path.get("cp"), errors="coerce")
    yards = pd.to_numeric(path.get("throw_distance"), errors="coerce")

    text = []
    for i, (_, row) in enumerate(path.iterrows(), start=1):
        parts = [f"Throw {i}"]
        if thrower.loc[row.name] or receiver.loc[row.name]:
            parts.append(f"{thrower.loc[row.name]} -> {receiver.loc[row.name]}")
        if pd.notna(aec.loc[row.name]):
            parts.append(f"aEC: {aec.loc[row.name]:.3f}")
        if pd.notna(cp.loc[row.name]):
            parts.append(f"CP: {cp.loc[row.name]:.1%}")
        if pd.notna(yards.loc[row.name]):
            parts.append(f"Distance: {yards.loc[row.name]:.1f}")
        text.append("<br>".join(parts))
    return text


def _format_browser_number(value, digits=2):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_browser_percent(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def _browser_path_lookup(paths):
    return {path["possession_id"].iloc[0]: path for path in paths if not path.empty}


def _sort_browser_possessions(possessions):
    sort_columns = [
        column
        for column in [
            "start_timestamp",
            "GameID",
            "game_quarter",
            "quarter_point",
            "possession_num",
            "is_home_team",
        ]
        if column in possessions
    ]
    return possessions.sort_values(sort_columns).reset_index(drop=True)


def _line_type_label(value):
    if value == "o_line":
        return "O-line"
    if value == "d_line":
        return "D-line"
    return "Unknown"


def _shape_cluster_label(row):
    browser_label = row.get("shape_cluster_label")
    if browser_label is not None and not pd.isna(browser_label):
        return str(browser_label)
    if "path_cluster" not in row or pd.isna(row.get("path_cluster")):
        return "Unclustered"
    cluster = int(row["path_cluster"])
    style = str(row.get("style", "shape")).strip()
    if not style or style.lower() == "nan":
        return f"Shape {cluster}"
    return f"Shape {cluster}: {style}"


def _cluster_average(group, column):
    if column not in group:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce")
    return values.mean()


def _cluster_primary_style(group):
    avg_hucks = _cluster_average(group, "huck_count")
    avg_resets = _cluster_average(group, "reset_count")
    avg_side_switches = _cluster_average(group, "shape_side_switches")

    if pd.notna(avg_hucks) and avg_hucks >= 0.5:
        return "huck-heavy"
    if pd.notna(avg_resets) and avg_resets >= 3:
        return "reset-heavy"
    if pd.notna(avg_side_switches) and avg_side_switches >= 3:
        return "switch-heavy"
    return "balanced"


def _cluster_field_usage(group):
    width = _cluster_average(group, "shape_width")
    middle_usage = _cluster_average(group, "shape_middle_third_share")
    sideline_usage = _cluster_average(group, "shape_sideline_share")

    if pd.notna(width) and width >= 34:
        return "full-width"
    if pd.notna(middle_usage) and middle_usage >= 0.55:
        return "middle"
    if pd.notna(sideline_usage) and sideline_usage >= 0.25:
        return "sideline"
    return None


def _cluster_route_shape(group):
    directness = _cluster_average(group, "shape_directness")
    if pd.notna(directness):
        if directness >= 0.75:
            return "direct"
        elif directness <= 0.50:
            return "circuitous"
    return "neutral"


def _cluster_progress_quality(group):
    avg_throws = _cluster_average(group, "throw_count")
    avg_yards_per_throw = _cluster_average(group, "yards_per_throw")
    if (
        pd.notna(avg_throws)
        and pd.notna(avg_yards_per_throw)
        and avg_throws >= 6
        and avg_yards_per_throw <= 5
    ):
        return "low-progress"
    return "flowing"


def _cluster_shape_components(group):
    descriptors = []
    primary_style = _cluster_primary_style(group)
    if primary_style:
        descriptors.append(primary_style)
    field_usage = _cluster_field_usage(group)
    if field_usage:
        descriptors.append(field_usage)
    descriptors.append(_cluster_route_shape(group))
    descriptors.append(_cluster_progress_quality(group))
    return descriptors


def _describe_shape_cluster(cluster, group):
    cluster_id = int(cluster)
    descriptors = _cluster_shape_components(group)
    descriptor_text = " | ".join(descriptors)
    return f"Shape {cluster_id}: {descriptor_text}"


def _describe_single_possession_shape(possession):
    width = pd.to_numeric(pd.Series([possession.get("shape_width")]), errors="coerce").iloc[0]
    directness = pd.to_numeric(
        pd.Series([possession.get("shape_directness")]),
        errors="coerce",
    ).iloc[0]
    side_switches = pd.to_numeric(
        pd.Series([possession.get("shape_side_switches")]),
        errors="coerce",
    ).iloc[0]
    hucks = pd.to_numeric(pd.Series([possession.get("huck_count")]), errors="coerce").iloc[0]
    resets = pd.to_numeric(pd.Series([possession.get("reset_count")]), errors="coerce").iloc[0]
    throws = pd.to_numeric(pd.Series([possession.get("throw_count")]), errors="coerce").iloc[0]
    middle_usage = pd.to_numeric(
        pd.Series([possession.get("shape_middle_third_share")]),
        errors="coerce",
    ).iloc[0]
    sideline_usage = pd.to_numeric(
        pd.Series([possession.get("shape_sideline_share")]),
        errors="coerce",
    ).iloc[0]
    yards_per_throw = pd.to_numeric(
        pd.Series([possession.get("yards_per_throw")]),
        errors="coerce",
    ).iloc[0]

    tags = []
    if pd.notna(hucks) and hucks >= 1:
        tags.append("huck-heavy")
    elif pd.notna(resets) and resets >= 3:
        tags.append("reset-heavy")
    elif pd.notna(side_switches) and side_switches >= 3:
        tags.append("switch-heavy")
    else:
        tags.append("balanced")

    if pd.notna(width) and width >= 34:
        tags.append("full-width")
    elif pd.notna(middle_usage) and middle_usage >= 0.55:
        tags.append("middle")
    elif pd.notna(sideline_usage) and sideline_usage >= 0.25:
        tags.append("sideline")

    if pd.notna(directness):
        if directness >= 0.75:
            tags.append("direct")
        elif directness <= 0.50:
            tags.append("circuitous")
        else:
            tags.append("neutral")
    else:
        tags.append("neutral")

    if (
        pd.notna(throws)
        and pd.notna(yards_per_throw)
        and throws >= 6
        and yards_per_throw <= 5
    ):
        tags.append("low-progress")
    else:
        tags.append("flowing")

    return " | ".join(tags)


def _add_browser_shape_cluster_labels(possessions):
    """Add one stable readable label per shape cluster for browser filters."""
    if possessions.empty or "path_cluster" not in possessions:
        return possessions.copy()

    labeled = possessions.copy()
    label_by_cluster = {}
    for cluster, group in labeled.groupby("path_cluster", dropna=False):
        if pd.isna(cluster):
            continue
        label_by_cluster[cluster] = _describe_shape_cluster(cluster, group)

    labeled["shape_cluster_label"] = labeled["path_cluster"].map(label_by_cluster)
    return labeled


def _format_overview_number(value, digits=1, percent=False):
    if value is None or pd.isna(value):
        return "-"
    if percent:
        return f"{float(value):.{digits}%}"
    return f"{float(value):.{digits}f}"


def render_shape_cluster_overview(possessions, selected_shape="all"):
    """Return a compact HTML summary of the current browser shape groups."""
    if possessions.empty or "path_cluster" not in possessions:
        return """
        <div style="font-family:system-ui;color:#637188;font-size:12px">
          No shape groups are available for the current filters.
        </div>
        """

    possessions = possessions.copy()
    for column in [
        "shape_width",
        "shape_side_switches",
        "shape_directness",
        "shape_middle_third_share",
        "shape_sideline_share",
        "huck_count",
        "reset_count",
        "aec_per_throw",
    ]:
        if column not in possessions:
            possessions[column] = np.nan

    summary = (
        possessions.groupby(["path_cluster", "shape_cluster_label"], dropna=False)
        .agg(
            possessions=("possession_id", "count"),
            avg_throws=("throw_count", "mean"),
            avg_width=("shape_width", "mean"),
            avg_side_switches=("shape_side_switches", "mean"),
            avg_directness=("shape_directness", "mean"),
            avg_middle_usage=("shape_middle_third_share", "mean"),
            avg_sideline_usage=("shape_sideline_share", "mean"),
            avg_hucks=("huck_count", "mean"),
            avg_resets=("reset_count", "mean"),
            avg_aec_per_throw=("aec_per_throw", "mean"),
        )
        .reset_index()
        .sort_values(["possessions", "path_cluster"], ascending=[False, True])
    )

    rows = []
    for _, row in summary.iterrows():
        selected = selected_shape != "all" and int(row["path_cluster"]) == selected_shape
        row_style = "background:#eef5ff;" if selected else ""
        rows.append(
            f"<tr style='{row_style}'>"
            f"<td>{escape(str(row['shape_cluster_label']))}</td>"
            f"<td>{int(row['possessions'])}</td>"
            f"<td>{_format_overview_number(row['avg_throws'])}</td>"
            f"<td>{_format_overview_number(row['avg_width'])}</td>"
            f"<td>{_format_overview_number(row['avg_side_switches'])}</td>"
            f"<td>{_format_overview_number(row['avg_directness'], percent=True)}</td>"
            f"<td>{_format_overview_number(row['avg_middle_usage'], percent=True)}</td>"
            f"<td>{_format_overview_number(row['avg_sideline_usage'], percent=True)}</td>"
            f"<td>{_format_overview_number(row['avg_hucks'])}</td>"
            f"<td>{_format_overview_number(row['avg_resets'])}</td>"
            f"<td>{_format_overview_number(row['avg_aec_per_throw'], digits=3)}</td>"
            "</tr>"
        )

    return f"""
    <div class="ufa-shape-overview">
      <style>
        .ufa-shape-overview {{
          width: 430px;
          margin: 6px 0 8px;
          color: #0b1a33;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 11px;
        }}
        .ufa-shape-overview details {{
          border: 1px solid #d9e1ea;
          border-radius: 4px;
          padding: 6px 8px;
          background: #fbfdff;
        }}
        .ufa-shape-overview summary {{
          cursor: pointer;
          font-weight: 700;
          color: #223a5e;
        }}
        .ufa-shape-overview p {{
          margin: 6px 0;
          color: #506078;
          line-height: 1.35;
        }}
        .ufa-shape-overview .guide {{
          margin-top: 8px;
          color: #506078;
          line-height: 1.35;
        }}
        .ufa-shape-overview .guide b {{
          color: #223a5e;
        }}
        .ufa-shape-overview .guide-title {{
          margin: 8px 0 4px;
          color: #223a5e;
          font-weight: 700;
        }}
        .ufa-shape-overview .guide-grid {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px 12px;
          margin-top: 6px;
        }}
        .ufa-shape-overview .guide-section {{
          padding: 6px 7px;
          border: 1px solid #edf1f5;
          border-radius: 4px;
          background: #ffffff;
        }}
        .ufa-shape-overview .guide-section h5 {{
          margin: 0 0 4px;
          color: #223a5e;
          font-size: 11px;
        }}
        .ufa-shape-overview .guide ul {{
          margin: 0;
          padding-left: 15px;
        }}
        .ufa-shape-overview .guide li {{
          margin: 2px 0;
        }}
        .ufa-shape-overview .scroll {{
          overflow-x: auto;
          overflow-y: visible;
          border-top: 1px solid #edf1f5;
          margin-top: 6px;
        }}
        .ufa-shape-overview table {{
          border-collapse: collapse;
          width: 100%;
          font-variant-numeric: tabular-nums;
        }}
        .ufa-shape-overview th,
        .ufa-shape-overview td {{
          border-bottom: 1px solid #edf1f5;
          padding: 4px 5px;
          text-align: right;
          white-space: nowrap;
        }}
        .ufa-shape-overview th:first-child,
        .ufa-shape-overview td:first-child {{
          text-align: left;
          min-width: 150px;
        }}
      </style>
      <details>
        <summary>Shape grouping overview</summary>
        <p>
          Groups are made from the possession geometry: resampled x/y path checkpoints,
          width used, side switches, middle/sideline usage, directness, red-zone entry,
          hucks, resets, and yardage style.
          Shape names describe the group average, so individual possessions inside a
          group can still look more central, wider, cleaner, or messier than the label.
        </p>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>Shape</th><th>N</th><th>Thr</th><th>Width</th>
                <th>Sw</th><th>Dir</th><th>Mid</th><th>Side</th>
                <th>Hu</th><th>Re</th><th>aEC/T</th>
              </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div class="guide">
          <div class="guide-title">Column guide</div>
          <ul>
            <li><b>N:</b> number of possessions in the shape group</li>
            <li><b>Thr:</b> average throws per possession</li>
            <li><b>Width:</b> average field width used</li>
            <li><b>Sw:</b> average side switches</li>
            <li><b>Dir:</b> directness</li>
            <li><b>Mid:</b> middle-third usage</li>
            <li><b>Side:</b> sideline usage</li>
            <li><b>Hu:</b> average hucks</li>
            <li><b>Re:</b> average resets</li>
            <li><b>aEC/T:</b> average aEC per throw</li>
          </ul>
        </div>
        <div class="guide">
          <div class="guide-title">How to read shape names</div>
          <p>
            Each label summarizes the average possession in that group using throw
            profile, field area, route shape, and progress quality.
          </p>
          <div class="guide-grid">
            <div class="guide-section">
              <h5>Throw profile</h5>
              <ul>
                <li><b>huck-heavy:</b> uses hucks often</li>
                <li><b>reset-heavy:</b> uses several resets</li>
                <li><b>switch-heavy:</b> changes sides often</li>
                <li><b>balanced:</b> no throw pattern dominates</li>
              </ul>
            </div>
            <div class="guide-section">
              <h5>Field area</h5>
              <ul>
                <li><b>middle:</b> attacks the middle third often</li>
                <li><b>sideline:</b> works near the sideline often</li>
                <li><b>full-width:</b> uses a large amount of field width</li>
              </ul>
            </div>
            <div class="guide-section">
              <h5>Route shape</h5>
              <ul>
                <li><b>direct:</b> moves upfield efficiently</li>
                <li><b>neutral:</b> neither clearly direct nor circuitous</li>
                <li><b>circuitous:</b> takes a less direct route upfield</li>
              </ul>
            </div>
            <div class="guide-section">
              <h5>Progress quality</h5>
              <ul>
                <li><b>flowing:</b> keeps gaining useful yardage</li>
                <li><b>low-progress:</b> uses several throws without much net gain</li>
              </ul>
            </div>
          </div>
        </div>
      </details>
    </div>
    """


def render_shownspace_possession_svg(path, width=260, height=560):
    """Return a Shown Space-style SVG field for one scoring possession."""
    path = path.sort_values("possession_throw").copy()
    possession_id = str(path["possession_id"].iloc[0]) if "possession_id" in path else "path"
    possession_key = hashlib.sha1(possession_id.encode("utf-8")).hexdigest()[:12]
    wrapper_id = f"ufa-browser-field-{possession_key}"
    detail_id = f"ufa-throw-detail-{possession_key}"
    field_width = width - 34
    field_height = height - 34
    left = (width - field_width) / 2
    top = 17

    def sx(value):
        value = float(value)
        scale = (value - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)
        return left + scale * field_width

    def sy(value):
        value = float(value)
        scale = (FIELD_Y_MAX - value) / (FIELD_Y_MAX - FIELD_Y_MIN)
        return top + scale * field_height

    def throw_detail_html(index, throw):
        thrower = throw.get("thrower") or throw.get("Thrower") or ""
        receiver = throw.get("receiver") or throw.get("Receiver") or ""
        distance = _format_browser_number(throw.get("throw_distance"), digits=1)
        cp = _format_browser_percent(throw.get("cp"))
        aec = _format_browser_number(throw.get("aec"), digits=3)
        quarter = throw.get("game_quarter", "-")
        quarter_point = throw.get("quarter_point", "-")
        return f"""
        <div class="ufa-detail-kicker">Throw {index}</div>
        <div class="ufa-detail-title">{escape(str(thrower))} -> {escape(str(receiver))}</div>
        <div class="ufa-detail-row"><span>Distance</span><b>{distance}</b></div>
        <div class="ufa-detail-row"><span>CP</span><b>{cp}</b></div>
        <div class="ufa-detail-row"><span>aEC</span><b>{aec}</b></div>
        <div class="ufa-detail-row"><span>Context</span><b>Q{quarter}, point {quarter_point}</b></div>
        """

    shapes = [
        f'<rect class="ufa-field" x="{left:.2f}" y="{top:.2f}" '
        f'width="{field_width:.2f}" height="{field_height:.2f}" />'
    ]
    for y_value in [ENDZONE_LOW_Y, ENDZONE_HIGH_Y]:
        shapes.append(
            f'<line class="ufa-yard-line" x1="{left:.2f}" y1="{sy(y_value):.2f}" '
            f'x2="{left + field_width:.2f}" y2="{sy(y_value):.2f}" />'
        )
    for y_value in [40, 80]:
        shapes.append(
            f'<circle class="ufa-center-dot" cx="{sx(0):.2f}" '
            f'cy="{sy(y_value):.2f}" r="2.5" />'
        )

    throw_shapes = []
    first_throw_detail_html = None
    for index, (_, throw) in enumerate(path.iterrows(), start=1):
        x1 = sx(throw["ThrowerX"])
        y1 = sy(throw["ThrowerY"])
        x2 = sx(throw["ReceiverX"])
        y2 = sy(throw["ReceiverY"])
        detail_html = throw_detail_html(index, throw)
        if first_throw_detail_html is None:
            first_throw_detail_html = detail_html
        update_detail = (
            f"document.getElementById({json.dumps(detail_id)}).innerHTML = "
            "this.getAttribute('data-detail-html');"
        )
        select_throw = (
            f"const wrapper = document.getElementById({json.dumps(wrapper_id)});"
            "wrapper.focus();"
            "wrapper.querySelectorAll('.ufa-throw.selected').forEach(function(node) {"
            "node.classList.remove('selected');"
            "});"
            "this.classList.add('selected');"
            "wrapper.dataset.selectedThrow = this.dataset.throwIndex;"
            f"{update_detail}"
        )
        selected_class = " selected" if index == 1 else ""
        throw_shapes.append(
            f'<g class="ufa-throw{selected_class}" '
            f'data-throw-index="{index}" '
            f'data-detail-html="{escape(detail_html, quote=True)}" '
            f'onmouseover="{escape(update_detail, quote=True)}" '
            f'onclick="{escape(select_throw, quote=True)}">'
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" />'
            f'<circle class="throw-start" cx="{x1:.2f}" cy="{y1:.2f}" r="2.8" />'
            f'<circle class="throw-end" cx="{x2:.2f}" cy="{y2:.2f}" r="3.1" />'
            "</g>"
        )

    keydown_handler = (
        "if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') { return; }"
        "event.preventDefault();"
        f"const wrapper = document.getElementById({json.dumps(wrapper_id)});"
        "const throws = Array.from(wrapper.querySelectorAll('.ufa-throw'));"
        "if (!throws.length) { return; }"
        "let selected = Number(wrapper.dataset.selectedThrow || 0);"
        "selected += event.key === 'ArrowRight' ? 1 : -1;"
        "selected = Math.max(1, Math.min(throws.length, selected));"
        "const nextThrow = wrapper.querySelector("
        "'.ufa-throw[data-throw-index=\"' + selected + '\"]'"
        ");"
        "if (!nextThrow) { return; }"
        "wrapper.querySelectorAll('.ufa-throw.selected').forEach(function(node) {"
        "node.classList.remove('selected');"
        "});"
        "nextThrow.classList.add('selected');"
        "wrapper.dataset.selectedThrow = selected;"
        f"document.getElementById({json.dumps(detail_id)}).innerHTML = "
        "nextThrow.getAttribute('data-detail-html');"
    )

    css = """
    <style>
      .ufa-browser-svg {
        background: #f5f8f5;
        display: block;
        border-radius: 4px;
      }
      .ufa-field { fill: #86d973; stroke: #071019; stroke-width: 2.2; }
      .ufa-yard-line { stroke: #071019; stroke-width: 1.6; }
      .ufa-center-dot { fill: #071019; stroke: none; }
      .ufa-throw line { stroke: #071019; stroke-width: 2.2; stroke-linecap: round; }
      .ufa-throw circle { fill: #071019; stroke: #071019; stroke-width: 1.2; }
      .ufa-throw:hover line { stroke: #c3482b; stroke-width: 3.2; }
      .ufa-throw:hover circle { fill: #c3482b; stroke: #071019; stroke-width: 1.8; }
      .ufa-throw.selected line { stroke: #c3482b; stroke-width: 3.4; }
      .ufa-throw.selected circle { fill: #c3482b; stroke: #071019; stroke-width: 1.9; }
      .ufa-throw { cursor: pointer; }
      .ufa-browser-field-wrap {
        display: flex;
        align-items: flex-start;
        gap: 18px;
        border: 1px solid #d9e1ea;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        padding: 16px;
      }
      .ufa-browser-field-wrap:focus {
        outline: 2px solid #9fc0e8;
        outline-offset: 4px;
      }
      .ufa-throw-detail {
        box-sizing: border-box;
        width: 220px;
        min-height: 150px;
        border-left: 1px solid #d8e0e8;
        padding: 10px 0 8px 16px;
        color: #0b1a33;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 12px;
        line-height: 1.4;
      }
      .ufa-detail-placeholder {
        color: #637188;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .ufa-detail-kicker {
        color: #637188;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
      }
      .ufa-detail-title {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-weight: 800;
        margin-bottom: 10px;
        color: #10233f;
      }
      .ufa-detail-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        border-bottom: 1px solid #edf1f5;
        padding: 5px 0;
      }
      .ufa-detail-row span { color: #637188; }
    </style>
    """
    detail_content = (
        first_throw_detail_html
        if first_throw_detail_html is not None
        else '<div class="ufa-detail-placeholder">Hover or click a throw on the field.</div>'
    )
    return (
        f'<div id="{wrapper_id}" class="ufa-browser-field-wrap" tabindex="0" '
        f'data-selected-throw="1" '
        f'aria-label="Possession throw browser" '
        f'onclick="this.focus()" '
        f'onkeydown="{escape(keydown_handler, quote=True)}">'
        f"{css}"
        f'<svg class="ufa-browser-svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f"{''.join(shapes)}{''.join(throw_shapes)}</svg>"
        f'<div id="{detail_id}" class="ufa-throw-detail">'
        f"{detail_content}"
        f"</div></div>"
    )


def render_mini_possession_svg(path, width=108, height=230):
    """Return a compact static field SVG for gallery-style possession scans."""
    path = path.sort_values("possession_throw").copy()
    field_width = width - 18
    field_height = height - 18
    left = (width - field_width) / 2
    top = 9

    def sx(value):
        value = float(value)
        scale = (value - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)
        return left + scale * field_width

    def sy(value):
        value = float(value)
        scale = (FIELD_Y_MAX - value) / (FIELD_Y_MAX - FIELD_Y_MIN)
        return top + scale * field_height

    shapes = [
        f'<rect class="ufa-mini-field" x="{left:.2f}" y="{top:.2f}" '
        f'width="{field_width:.2f}" height="{field_height:.2f}" />'
    ]
    for y_value in [ENDZONE_LOW_Y, ENDZONE_HIGH_Y]:
        shapes.append(
            f'<line class="ufa-mini-yard-line" x1="{left:.2f}" y1="{sy(y_value):.2f}" '
            f'x2="{left + field_width:.2f}" y2="{sy(y_value):.2f}" />'
        )
    for y_value in [40, 80]:
        shapes.append(
            f'<circle class="ufa-mini-center-dot" cx="{sx(0):.2f}" '
            f'cy="{sy(y_value):.2f}" r="1.7" />'
        )

    throw_shapes = []
    for _, throw in path.iterrows():
        x1 = sx(throw["ThrowerX"])
        y1 = sy(throw["ThrowerY"])
        x2 = sx(throw["ReceiverX"])
        y2 = sy(throw["ReceiverY"])
        throw_shapes.append(
            f'<line class="ufa-mini-throw-line" x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" />'
            f'<circle class="ufa-mini-dot" cx="{x1:.2f}" cy="{y1:.2f}" r="1.9" />'
            f'<circle class="ufa-mini-dot" cx="{x2:.2f}" cy="{y2:.2f}" r="2.1" />'
        )

    return (
        f'<svg class="ufa-mini-svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f"{''.join(shapes)}{''.join(throw_shapes)}</svg>"
    )


def render_possession_overlay_group(
    path,
    color="#c3482b",
    width=260,
    height=560,
    group_id=None,
    hidden=False,
    anchor_x=0,
    normalize_shape=False,
    target_y_low=18,
    target_y_high=104,
    max_scale=4,
):
    """Return only the throw layers for overlaying one possession on a shared field."""
    path = path.sort_values("possession_throw").copy()
    field_width = width - 34
    field_height = height - 34
    left = (width - field_width) / 2
    top = 17

    def sx(value):
        value = float(value)
        scale = (value - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)
        return left + scale * field_width

    def sy(value):
        value = float(value)
        scale = (FIELD_Y_MAX - value) / (FIELD_Y_MAX - FIELD_Y_MIN)
        return top + scale * field_height

    shape_scale = 1
    x_offset = 0
    y_offset = 0
    x_values = pd.Series(dtype="float64")
    y_values = pd.Series(dtype="float64")
    if not path.empty:
        x_values = pd.concat(
            [
                pd.to_numeric(path["ThrowerX"], errors="coerce"),
                pd.to_numeric(path["ReceiverX"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        y_values = pd.concat(
            [
                pd.to_numeric(path["ThrowerY"], errors="coerce"),
                pd.to_numeric(path["ReceiverY"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()

    if normalize_shape and not x_values.empty and not y_values.empty:
        x_min = float(x_values.min())
        x_max = float(x_values.max())
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        x_span = max(x_max - x_min, 1)
        y_span = max(y_max - y_min, 1)
        target_y_span = max(target_y_high - target_y_low, 1)
        shape_scale = min(max_scale, target_y_span / y_span)
        if x_span * shape_scale > (FIELD_X_MAX - FIELD_X_MIN) * 0.84:
            shape_scale = ((FIELD_X_MAX - FIELD_X_MIN) * 0.84) / x_span
        scaled_x_mid = ((x_min + x_max) / 2) * shape_scale
        scaled_y_mid = ((y_min + y_max) / 2) * shape_scale
        target_y_mid = (target_y_low + target_y_high) / 2
        x_offset = anchor_x - scaled_x_mid
        y_offset = target_y_mid - scaled_y_mid
    elif not x_values.empty and not y_values.empty:
        x_offset = 0
        y_offset = 0

    def overlay_x(value):
        value = float(value)
        if not normalize_shape:
            return value
        return value * shape_scale + x_offset

    def overlay_y(value):
        value = float(value)
        if not normalize_shape:
            return value
        return value * shape_scale + y_offset

    throw_shapes = []
    for _, throw in path.iterrows():
        x1 = sx(overlay_x(throw["ThrowerX"]))
        y1 = sy(overlay_y(throw["ThrowerY"]))
        x2 = sx(overlay_x(throw["ReceiverX"]))
        y2 = sy(overlay_y(throw["ReceiverY"]))
        throw_shapes.append(
            f'<line class="ufa-overlay-throw-line" x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" />'
            f'<circle class="ufa-overlay-dot" cx="{x1:.2f}" cy="{y1:.2f}" r="2.0" />'
            f'<circle class="ufa-overlay-dot" cx="{x2:.2f}" cy="{y2:.2f}" r="2.2" />'
        )

    group_attrs = 'class="ufa-overlay-path"'
    if group_id is not None:
        group_attrs += f' id="{escape(str(group_id), quote=True)}"'
    display_style = "display: none; " if hidden else ""
    group_attrs += (
        f' style="{display_style}--overlay-color: {escape(color, quote=True)};"'
    )
    return f"<g {group_attrs}>{''.join(throw_shapes)}</g>"


def render_empty_overlay_field(layer_id, width=260, height=560, overlay_groups=""):
    """Return a shared field SVG whose layer group can be filled by selected cards."""
    field_width = width - 34
    field_height = height - 34
    left = (width - field_width) / 2
    top = 17

    def sx(value):
        value = float(value)
        scale = (value - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)
        return left + scale * field_width

    def sy(value):
        value = float(value)
        scale = (FIELD_Y_MAX - value) / (FIELD_Y_MAX - FIELD_Y_MIN)
        return top + scale * field_height

    shapes = [
        f'<rect class="ufa-mini-field" x="{left:.2f}" y="{top:.2f}" '
        f'width="{field_width:.2f}" height="{field_height:.2f}" />'
    ]
    for y_value in [ENDZONE_LOW_Y, ENDZONE_HIGH_Y]:
        shapes.append(
            f'<line class="ufa-mini-yard-line" x1="{left:.2f}" y1="{sy(y_value):.2f}" '
            f'x2="{left + field_width:.2f}" y2="{sy(y_value):.2f}" />'
        )
    for y_value in [40, 80]:
        shapes.append(
            f'<circle class="ufa-mini-center-dot" cx="{sx(0):.2f}" '
            f'cy="{sy(y_value):.2f}" r="2.5" />'
        )

    return (
        f'<svg class="ufa-overlay-svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f"{''.join(shapes)}"
        f'<g id="{layer_id}" class="ufa-overlay-layers">{overlay_groups}</g>'
        "</svg>"
    )


def render_possession_shape_gallery(
    possessions,
    path_lookup,
    max_shapes=6,
    max_per_shape=4,
):
    """Render a grouped mini-field gallery for comparing possession shapes."""
    if possessions.empty:
        return """
        <div class="ufa-shape-gallery">
          <div class="empty">No possessions match these filters.</div>
        </div>
        """

    label_column = (
        "shape_cluster_label"
        if "shape_cluster_label" in possessions
        else "style"
        if "style" in possessions
        else None
    )
    if label_column is None:
        possessions = possessions.copy()
        possessions["shape_gallery_label"] = "All possessions"
        label_column = "shape_gallery_label"

    groups = (
        possessions.groupby(label_column, dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(max_shapes)
    )

    sections = []
    for shape_label, count in groups.items():
        group = possessions[possessions[label_column].eq(shape_label)].head(max_per_shape)
        cards = []
        for _, row in group.iterrows():
            path = path_lookup.get(row["possession_id"])
            if path is None or path.empty:
                continue
            aec_per_throw = _format_browser_number(row.get("aec_per_throw"), digits=3)
            total_aec = _format_browser_number(row.get("total_aec"), digits=3)
            outcome = escape(str(row.get("outcome", "unknown")).title())
            line_type = escape(_line_type_label(row.get("line_type")))
            meta = (
                f"{escape(str(row.get('GameID', '-')))} | "
                f"Q{row.get('game_quarter', '-')} P{row.get('quarter_point', '-')} | "
                f"{int(row.get('throw_count', len(path)))} throws"
            )
            cards.append(
                f"""
                <div class="ufa-gallery-card">
                  {render_mini_possession_svg(path)}
                  <div class="ufa-gallery-card-meta">
                    <b>{outcome}</b> <span>{line_type}</span>
                    <div>{escape(meta)}</div>
                    <div>aEC/T <b>{aec_per_throw}</b> &middot; total <b>{total_aec}</b></div>
                  </div>
                </div>
                """
            )
        if not cards:
            continue
        sections.append(
            f"""
            <section class="ufa-gallery-section">
              <div class="ufa-gallery-section-title">
                <h3>{escape(str(shape_label))}</h3>
                <span>{len(cards)} shown of {int(count)}</span>
              </div>
              <div class="ufa-gallery-grid">{''.join(cards)}</div>
            </section>
            """
        )

    return f"""
    <div class="ufa-shape-gallery">
      <div class="ufa-gallery-kicker">Shape Gallery</div>
      {''.join(sections)}
    </div>
    """


def render_possession_free_board(
    possessions,
    path_lookup,
    max_cards=24,
    include_overlay=True,
    return_overlay=False,
    external_controls=False,
    persistence_key=None,
):
    """Render a draggable board of mini possession cards."""
    if possessions.empty:
        board_html = """
        <div class="ufa-free-board-wrap">
          <div class="empty">No possessions match these filters.</div>
        </div>
        """
        if return_overlay:
            return board_html, ""
        return board_html

    frame = possessions.head(max_cards).copy()
    board_key = hashlib.sha1(
        "|".join(frame["possession_id"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()[:12]
    persistence_token = str(persistence_key or board_key).strip()
    board_id = f"ufa-free-board-{board_key}"
    wrap_id = f"ufa-free-wrap-{board_key}"
    overlay_layer_id = f"ufa-free-overlay-layer-{board_key}"
    overlay_count_id = f"ufa-free-overlay-count-{board_key}"
    export_status_id = f"ufa-free-export-status-{board_key}"
    export_text_id = f"ufa-free-export-text-{board_key}"
    insert_position_id = f"ufa-free-insert-position-{board_key}"
    arrange_selection_count_id = f"ufa-free-arrange-selection-count-{board_key}"
    undo_button_id = f"ufa-free-undo-arrangement-{board_key}"
    recovery_history_id = f"ufa-free-recovery-history-{board_key}"
    recovery_restore_id = f"ufa-free-recovery-restore-{board_key}"
    recovery_status_id = f"ufa-free-recovery-status-{board_key}"
    storage_key = f"ufa-free-arrange:{persistence_token}"
    legacy_storage_key = f"ufa-free-arrange:{board_key}"
    recovery_storage_key = f"ufa-free-recovery:{persistence_token}"
    overlay_palette = [
        "#c3482b",
        "#0d4f94",
        "#2f7d3b",
        "#7f3fbf",
        "#d9891b",
        "#008c8c",
        "#b83280",
        "#57606f",
    ]
    overlay_width = 230
    overlay_height = 496
    update_overlay = (
        f"const wrap = document.getElementById({json.dumps(wrap_id)});"
        f"const count = document.getElementById({json.dumps(overlay_count_id)});"
        f"const layer = document.getElementById({json.dumps(overlay_layer_id)});"
        "if (!wrap || !count || !layer) { return; }"
        "layer.querySelectorAll('.ufa-overlay-path').forEach(function(path) {"
        "path.style.display = 'none';"
        "});"
        "const selected = Array.from("
        "wrap.querySelectorAll('.ufa-free-overlay-check:checked')"
        ").filter(function(input) {"
        "const card = input.closest('.ufa-free-card');"
        "return Boolean(card && !card.hidden);"
        "});"
        "selected.forEach(function(input) {"
        "const path = document.getElementById(input.dataset.overlayPath);"
        "if (path) { path.style.display = ''; }"
        "});"
        "count.textContent = selected.length ? selected.length + ' selected' : 'Select cards to overlay';"
        "wrap.querySelectorAll('.ufa-free-card.overlay-selected').forEach(function(card) {"
        "card.classList.remove('overlay-selected');"
        "});"
        "selected.forEach(function(input) {"
        "const card = input.closest('.ufa-free-card');"
        "if (card) { card.classList.add('overlay-selected'); }"
        "});"
    )
    clear_overlay = (
        f"const clearWrap = document.getElementById({json.dumps(wrap_id)});"
        "if (!clearWrap) { return; }"
        "clearWrap.querySelectorAll('.ufa-free-overlay-check').forEach(function(input) {"
        "input.checked = false;"
        "});"
        f"{update_overlay}"
    )
    early_undo = (
        "if (!board._ufaUndoStack) { board._ufaUndoStack = []; }"
        "if (!board._ufaPushUndo) {"
        "board._ufaPushUndo = function() {"
        "if (board._ufaUndoRestoring || board._ufaSuppressUndo) { return; }"
        "const snapshot = {"
        "children: Array.from(board.children).map(function(item) {"
        "const titleInput = item.querySelector('.ufa-free-row-break input');"
        "return {"
        "node: item,"
        "className: item.className,"
        "hidden: item.hidden,"
        "title: titleInput ? titleInput.value : null,"
        "overlay: Array.from(item.querySelectorAll('.ufa-free-overlay-check'))"
        ".map(function(input) { return input.checked; })"
        "};"
        "}),"
        "selectionCommitted: board.dataset.selectionCommitted || 'false'"
        "};"
        "board._ufaUndoStack.push(snapshot);"
        "if (board._ufaUndoStack.length > 50) { board._ufaUndoStack.shift(); }"
        f"const undoButton = document.getElementById({json.dumps(undo_button_id)});"
        "if (undoButton) { undoButton.disabled = false; }"
        "};"
        "}"
        "board._ufaPushUndo();"
    )
    drag_start = (
        "const dragOrigin = event.target && event.target.closest("
        "'button, input, select, textarea, label'"
        ");"
        "if (dragOrigin && dragOrigin !== this) {"
        "event.preventDefault();"
        "event.stopPropagation();"
        "return;"
        "}"
        "event.dataTransfer.setData('text/plain', this.id);"
        "event.dataTransfer.effectAllowed = 'move';"
        "this.classList.add('dragging');"
        "const dragBoard = this.closest('.ufa-free-board');"
        "if (dragBoard) {"
        "dragBoard._ufaDragSnapshot = Array.from(dragBoard.children);"
        "dragBoard._ufaDropCommitted = false;"
        "}"
        "if (dragBoard && this.classList.contains('ufa-free-card')"
        "&& this.classList.contains('row-selected')) {"
        "const selectedCards = Array.from("
        "dragBoard.querySelectorAll('.ufa-free-card.row-selected')"
        ");"
        "if (selectedCards.length > 1) {"
        "selectedCards.forEach(function(card) {"
        "card.classList.add('group-dragging');"
        "});"
        "document.querySelectorAll('.ufa-multi-drag-preview').forEach(function(item) {"
        "if (item._ufaMoveHandler) {"
        "document.removeEventListener('dragover', item._ufaMoveHandler, true);"
        "}"
        "item.remove();"
        "});"
        "const preview = document.createElement('div');"
        "preview.className = 'ufa-multi-drag-preview';"
        "const cardCount = selectedCards.length;"
        "const fieldWidth = 108;"
        "const fieldHeight = 230;"
        "const previewGap = 6;"
        "const previewPadding = 12;"
        "const availableWidth = Math.max(fieldWidth, window.innerWidth * 0.72);"
        "const availableHeight = Math.max(fieldHeight, window.innerHeight * 0.82);"
        "const maxFullColumns = Math.max(1, Math.floor("
        "(availableWidth - previewPadding + previewGap) / (fieldWidth + previewGap)"
        "));"
        "const columns = Math.min(cardCount, 6, maxFullColumns);"
        "const rows = Math.ceil(cardCount / columns);"
        "const widthScale = (availableWidth - previewPadding"
        "- ((columns - 1) * previewGap)) / (columns * fieldWidth);"
        "const heightScale = (availableHeight - previewPadding"
        "- ((rows - 1) * previewGap)) / (rows * fieldHeight);"
        "const previewScale = Math.min(1, widthScale, heightScale);"
        "const thumbWidth = Math.max(18, Math.floor(fieldWidth * previewScale));"
        "const thumbHeight = Math.max(38, Math.round(fieldHeight * previewScale));"
        "preview.style.gridTemplateColumns = 'repeat(' + columns + ', '"
        "+ thumbWidth + 'px)';"
        "selectedCards.forEach(function(card) {"
        "const field = card.querySelector('.ufa-mini-svg');"
        "if (!field) { return; }"
        "const fieldClone = field.cloneNode(true);"
        "fieldClone.setAttribute('width', thumbWidth);"
        "fieldClone.setAttribute('height', thumbHeight);"
        "fieldClone.style.margin = '0';"
        "preview.appendChild(fieldClone);"
        "});"
        "document.body.appendChild(preview);"
        "const draggedIndex = Math.max(0, selectedCards.indexOf(this));"
        "const draggedColumn = draggedIndex % columns;"
        "const draggedRow = Math.floor(draggedIndex / columns);"
        "const sourceField = this.querySelector('.ufa-mini-svg');"
        "const sourceRect = sourceField ? sourceField.getBoundingClientRect() : null;"
        "const pointerXRatio = sourceRect && sourceRect.width"
        "? Math.max(0, Math.min(1, (event.clientX - sourceRect.left) / sourceRect.width))"
        ": 0.5;"
        "const pointerYRatio = sourceRect && sourceRect.height"
        "? Math.max(0, Math.min(1, (event.clientY - sourceRect.top) / sourceRect.height))"
        ": 0.5;"
        "const previewInset = 8;"
        "const anchorX = previewInset + (draggedColumn * (thumbWidth + previewGap))"
        "+ (pointerXRatio * thumbWidth);"
        "const anchorY = previewInset + (draggedRow * (thumbHeight + previewGap))"
        "+ (pointerYRatio * thumbHeight);"
        "const positionPreview = function(moveEvent) {"
        "if (!moveEvent || (!moveEvent.clientX && !moveEvent.clientY)) { return; }"
        "preview.style.left = (moveEvent.clientX - anchorX) + 'px';"
        "preview.style.top = (moveEvent.clientY - anchorY) + 'px';"
        "};"
        "preview._ufaMoveHandler = positionPreview;"
        "document.addEventListener('dragover', positionPreview, true);"
        "positionPreview(event);"
        "const nativeGhost = document.createElement('canvas');"
        "nativeGhost.className = 'ufa-native-drag-ghost';"
        "nativeGhost.width = 1;"
        "nativeGhost.height = 1;"
        "document.body.appendChild(nativeGhost);"
        "event.dataTransfer.setDragImage(nativeGhost, 0, 0);"
        "window.setTimeout(function() { nativeGhost.remove(); }, 0);"
        "}"
        "}"
    )
    protect_row_controls = (
        "breaker.querySelectorAll('button, input').forEach(function(control) {"
        "control.draggable = false;"
        "control.onpointerdown = function(event) {"
        "event.stopPropagation();"
        "breaker.draggable = false;"
        "if (control.tagName === 'BUTTON' && control.setPointerCapture"
        "&& typeof event.pointerId === 'number') {"
        "try { control.setPointerCapture(event.pointerId); } catch (error) {}"
        "}"
        "const restoreRowDrag = function() {"
        "breaker.draggable = true;"
        "document.removeEventListener('pointerup', restoreRowDrag, true);"
        "document.removeEventListener('pointercancel', restoreRowDrag, true);"
        "};"
        "document.addEventListener('pointerup', restoreRowDrag, true);"
        "document.addEventListener('pointercancel', restoreRowDrag, true);"
        "};"
        "control.onmousedown = function(event) { event.stopPropagation(); };"
        "control.ondragstart = function(event) {"
        "event.preventDefault();"
        "event.stopPropagation();"
        "};"
        "});"
    )
    drag_end = (
        "const dragBoard = this.closest('.ufa-free-board');"
        "this.classList.remove('dragging');"
        "if (dragBoard) {"
        "if (!dragBoard._ufaDropCommitted"
        "&& Array.isArray(dragBoard._ufaDragSnapshot)) {"
        "const restoreFragment = document.createDocumentFragment();"
        "dragBoard._ufaDragSnapshot.forEach(function(item) {"
        "restoreFragment.appendChild(item);"
        "});"
        "dragBoard.appendChild(restoreFragment);"
        "}"
        "dragBoard._ufaDragSnapshot = null;"
        "dragBoard._ufaDropCommitted = false;"
        "dragBoard.querySelectorAll('.group-dragging').forEach(function(item) {"
        "item.classList.remove('group-dragging');"
        "});"
        "dragBoard.querySelectorAll('.drop-target').forEach(function(item) {"
        "item.classList.remove('drop-target');"
        "});"
        "}"
        "document.querySelectorAll('.ufa-multi-drag-preview').forEach(function(item) {"
        "if (item._ufaMoveHandler) {"
        "document.removeEventListener('dragover', item._ufaMoveHandler, true);"
        "}"
        "item.remove();"
        "});"
        "document.querySelectorAll('.ufa-native-drag-ghost').forEach(function(item) {"
        "item.remove();"
        "});"
    )
    drag_over = (
        "event.preventDefault();"
        "if (event.dataTransfer) { event.dataTransfer.dropEffect = 'move'; }"
        "this.classList.add('drop-target');"
    )
    drag_leave = "this.classList.remove('drop-target');"
    drop = (
        "event.preventDefault();"
        "this.classList.remove('drop-target');"
        f"const board = document.getElementById({json.dumps(board_id)});"
        "const dragged = document.getElementById("
        "event.dataTransfer.getData('text/plain')"
        ");"
        "if (!board || !dragged || dragged === this) { return; }"
        "const movingItems = dragged.classList.contains('ufa-free-card')"
        "&& dragged.classList.contains('row-selected')"
        "? Array.from(board.querySelectorAll('.ufa-free-card.row-selected'))"
        ": [dragged];"
        "if (movingItems.includes(this)) { return; }"
        "const items = Array.from(board.querySelectorAll('.ufa-free-board-item'));"
        "const fromIndex = items.indexOf(dragged);"
        "const toIndex = items.indexOf(this);"
        "if (fromIndex < 0 || toIndex < 0) { return; }"
        f"{early_undo}"
        "const fragment = document.createDocumentFragment();"
        "movingItems.forEach(function(item) { fragment.appendChild(item); });"
        "if (fromIndex < toIndex) { this.after(fragment); }"
        "else { this.before(fragment); }"
        "board._ufaDropCommitted = true;"
    )
    select_arrange_target = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "if (!board) { return; }"
        "board.querySelectorAll('.arrange-target').forEach(function(item) {"
        "item.classList.remove('arrange-target');"
        "});"
        "this.classList.add('arrange-target');"
    )
    update_arrange_selection = (
        f"const selectionWrap = document.getElementById({json.dumps(wrap_id)});"
        f"const selectionCount = document.getElementById({json.dumps(arrange_selection_count_id)});"
        "if (!selectionWrap || !selectionCount) { return; }"
        "const arrangeSelected = Array.from(selectionWrap.querySelectorAll('.ufa-free-card.row-selected'));"
        "selectionCount.textContent = arrangeSelected.length"
        "? arrangeSelected.length + ' selected'"
        ": '0 selected';"
    )
    ensure_undo = (
        "if (!board._ufaUndoStack) { board._ufaUndoStack = []; }"
        "if (!board._ufaUpdateUndoButton) {"
        "board._ufaUpdateUndoButton = function() {"
        f"const undoButton = document.getElementById({json.dumps(undo_button_id)});"
        "if (!undoButton) { return; }"
        "const canUndo = Array.isArray(board._ufaUndoStack)"
        "&& board._ufaUndoStack.length > 0;"
        "undoButton.disabled = !canUndo;"
        "undoButton.title = canUndo"
        "? 'Undo last arrangement change'"
        ": 'Nothing to undo';"
        "};"
        "}"
        "if (!board._ufaPushUndo) {"
        "board._ufaPushUndo = function() {"
        "if (board._ufaUndoRestoring || board._ufaSuppressUndo) { return; }"
        "const snapshot = {"
        "children: Array.from(board.children).map(function(item) {"
        "const titleInput = item.querySelector('.ufa-free-row-break input');"
        "return {"
        "node: item,"
        "className: item.className,"
        "hidden: item.hidden,"
        "title: titleInput ? titleInput.value : null,"
        "overlay: Array.from(item.querySelectorAll('.ufa-free-overlay-check'))"
        ".map(function(input) { return input.checked; })"
        "};"
        "}),"
        "selectionCommitted: board.dataset.selectionCommitted || 'false'"
        "};"
        "board._ufaUndoStack.push(snapshot);"
        "if (board._ufaUndoStack.length > 50) {"
        "board._ufaUndoStack.shift();"
        "}"
        "board._ufaUpdateUndoButton();"
        "};"
        "}"
        "if (!board._ufaUndo) {"
        "board._ufaUndo = function() {"
        "if (!Array.isArray(board._ufaUndoStack)"
        "|| !board._ufaUndoStack.length) {"
        "board._ufaUpdateUndoButton();"
        "return;"
        "}"
        "const snapshot = board._ufaUndoStack.pop();"
        "board._ufaUndoRestoring = true;"
        "Array.from(board.children).forEach(function(item) { item.remove(); });"
        "snapshot.children.forEach(function(entry) {"
        "entry.node.className = entry.className;"
        "entry.node.hidden = entry.hidden;"
        "const titleInput = entry.node.querySelector('.ufa-free-row-break input');"
        "if (titleInput && entry.title !== null) {"
        "titleInput.value = entry.title;"
        "}"
        "const inputs = Array.from(entry.node.querySelectorAll('.ufa-free-overlay-check'));"
        "inputs.forEach(function(input, index) {"
        "input.checked = Boolean(entry.overlay[index]);"
        "});"
        "board.appendChild(entry.node);"
        "});"
        "board.dataset.selectionCommitted = snapshot.selectionCommitted || 'false';"
        "board._ufaUndoRestoring = false;"
        "board._ufaUpdateUndoButton();"
        f"{update_arrange_selection}"
        f"{update_overlay}"
        "};"
        "}"
        "board._ufaUpdateUndoButton();"
    )
    undo_arrangement = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "if (!board) { return; }"
        f"{ensure_undo}"
        "board._ufaUndo();"
    )
    toggle_arrange_selection = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "if (!board) { return; }"
        "board.querySelectorAll('.ufa-free-board-item.arrange-target').forEach(function(item) {"
        "item.classList.remove('arrange-target');"
        "});"
        "const startsNewSelection = board.dataset.selectionCommitted === 'true'"
        "&& !this.classList.contains('row-selected');"
        "if (startsNewSelection) {"
        "board.querySelectorAll('.ufa-free-card.row-selected').forEach(function(card) {"
        "card.classList.remove('row-selected');"
        "});"
        "}"
        "board.dataset.selectionCommitted = 'false';"
        "this.classList.toggle('row-selected');"
        f"{update_arrange_selection}"
    )
    clear_arrange_selection = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "if (!board) { return; }"
        "board.querySelectorAll('.ufa-free-card.row-selected').forEach(function(card) {"
        "card.classList.remove('row-selected');"
        "});"
        "board.dataset.selectionCommitted = 'false';"
        f"{update_arrange_selection}"
    )

    overlay_row = (
        f"const rowWrap = document.getElementById({json.dumps(wrap_id)});"
        "const rowBreak = this.closest('.ufa-free-row-break');"
        "if (!rowWrap || !rowBreak) { return; }"
        "rowWrap.querySelectorAll('.ufa-free-overlay-check').forEach(function(input) {"
        "input.checked = false;"
        "});"
        "let item = rowBreak.nextElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "if (item.classList.contains('ufa-free-card') && !item.hidden) {"
        "const checkbox = item.querySelector('.ufa-free-overlay-check');"
        "if (checkbox) { checkbox.checked = true; }"
        "}"
        "item = item.nextElementSibling;"
        "}"
        f"{update_overlay}"
    )
    clear_overlay_row = (
        f"const rowWrap = document.getElementById({json.dumps(wrap_id)});"
        "const rowBreak = this.closest('.ufa-free-row-break');"
        "if (!rowWrap || !rowBreak) { return; }"
        "let item = rowBreak.nextElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "if (item.classList.contains('ufa-free-card')) {"
        "const checkbox = item.querySelector('.ufa-free-overlay-check');"
        "if (checkbox) { checkbox.checked = false; }"
        "}"
        "item = item.nextElementSibling;"
        "}"
        f"{update_overlay}"
    )

    move_row_up = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "const rowBreak = this.closest('.ufa-free-row-break');"
        "if (!board || !rowBreak) { return; }"
        "if (rowBreak === board.firstElementChild) { return; }"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "const group = [rowBreak];"
        "let item = rowBreak.nextElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "group.push(item);"
        "item = item.nextElementSibling;"
        "}"
        "let previousBreak = rowBreak.previousElementSibling;"
        "while (previousBreak && !previousBreak.classList.contains('ufa-free-row-break')) {"
        "previousBreak = previousBreak.previousElementSibling;"
        "}"
        "const insertionPoint = previousBreak || board.firstElementChild;"
        "const fragment = document.createDocumentFragment();"
        "group.forEach(function(node) { fragment.appendChild(node); });"
        "if (insertionPoint) { board.insertBefore(fragment, insertionPoint); }"
        "else { board.appendChild(fragment); }"
    )

    move_row_down = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "const rowBreak = this.closest('.ufa-free-row-break');"
        "if (!board || !rowBreak) { return; }"
        "const group = [rowBreak];"
        "let item = rowBreak.nextElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "group.push(item);"
        "item = item.nextElementSibling;"
        "}"
        "const nextBreak = item;"
        "if (!nextBreak) { return; }"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "item = nextBreak.nextElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "item = item.nextElementSibling;"
        "}"
        "const insertionPoint = item;"
        "const fragment = document.createDocumentFragment();"
        "group.forEach(function(node) { fragment.appendChild(node); });"
        "if (insertionPoint) { board.insertBefore(fragment, insertionPoint); }"
        "else { board.appendChild(fragment); }"
    )

    move_row_to_top = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "const rowBreak = this.closest('.ufa-free-row-break');"
        "if (!board || !rowBreak) { return; }"
        "if (rowBreak === board.firstElementChild) { return; }"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "const group = [rowBreak];"
        "let item = rowBreak.nextElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "group.push(item);"
        "item = item.nextElementSibling;"
        "}"
        "const fragment = document.createDocumentFragment();"
        "group.forEach(function(node) { fragment.appendChild(node); });"
        "board.insertBefore(fragment, board.firstElementChild);"
    )

    move_row_to_bottom = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "const rowBreak = this.closest('.ufa-free-row-break');"
        "if (!board || !rowBreak) { return; }"
        "let item = rowBreak.nextElementSibling;"
        "const group = [rowBreak];"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "group.push(item);"
        "item = item.nextElementSibling;"
        "}"
        "if (group[group.length - 1] === board.lastElementChild) { return; }"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "const fragment = document.createDocumentFragment();"
        "group.forEach(function(node) { fragment.appendChild(node); });"
        "board.appendChild(fragment);"
    )

    bind_row_move_controls = (
        "const bindRowMoveButton = function(button, moveRow) {"
        "if (!button) { return; }"
        "button.onpointerup = function(event) {"
        "if (event.isPrimary === false"
        "|| (event.pointerType === 'mouse' && event.button !== 0)) { return; }"
        "event.preventDefault();"
        "event.stopPropagation();"
        "this._ufaPointerMoveHandled = true;"
        "const handledButton = this;"
        "window.setTimeout(function() {"
        "handledButton._ufaPointerMoveHandled = false;"
        "}, 0);"
        "moveRow.call(this);"
        "};"
        "button.onclick = function(event) {"
        "event.preventDefault();"
        "event.stopPropagation();"
        "if (this._ufaPointerMoveHandled) { return; }"
        "moveRow.call(this);"
        "};"
        "};"
        "bindRowMoveButton("
        "breaker.querySelector('.ufa-free-row-move-up'),"
        f"function() {{ {move_row_up} }}"
        ");"
        "bindRowMoveButton("
        "breaker.querySelector('.ufa-free-row-move-down'),"
        f"function() {{ {move_row_down} }}"
        ");"
        "bindRowMoveButton("
        "breaker.querySelector('.ufa-free-row-move-top'),"
        f"function() {{ {move_row_to_top} }}"
        ");"
        "bindRowMoveButton("
        "breaker.querySelector('.ufa-free-row-move-bottom'),"
        f"function() {{ {move_row_to_bottom} }}"
        ");"
    )

    insert_row_break = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        f"const select = document.getElementById({json.dumps(insert_position_id)});"
        "if (!board || !select) { return; }"
        "const allSelectedCards = Array.from(board.querySelectorAll('.ufa-free-card.row-selected'));"
        "const selectedCards = allSelectedCards.filter(function(card) { return !card.hidden; });"
        "let position = select.value;"
        "let target = board.querySelector('.ufa-free-board-item.arrange-target');"
        "let previousBreak = null;"
        "let nextBreak = null;"
        "let restoreViewportAnchor = function() {};"
        "if (selectedCards.length) {"
        "const items = Array.from(board.querySelectorAll('.ufa-free-board-item'));"
        "selectedCards.sort(function(left, right) {"
        "return items.indexOf(left) - items.indexOf(right);"
        "});"
        "const viewportCenter = window.innerHeight / 2;"
        "const nearestViewportCard = selectedCards.reduce(function(best, card) {"
        "const cardRect = card.getBoundingClientRect();"
        "const distance = Math.abs((cardRect.top + (cardRect.height / 2)) - viewportCenter);"
        "return !best || distance < best.distance"
        "? { card: card, distance: distance }"
        ": best;"
        "}, null);"
        "const viewportAnchorCard = position === 'above'"
        "? selectedCards[selectedCards.length - 1]"
        ": position === 'below'"
        "? selectedCards[0]"
        ": nearestViewportCard.card;"
        "const viewportAnchor = {"
        "card: viewportAnchorCard,"
        "top: viewportAnchorCard.getBoundingClientRect().top"
        "};"
        "restoreViewportAnchor = function() {"
        "if (!viewportAnchor || !viewportAnchor.card.isConnected) { return; }"
        "const currentTop = viewportAnchor.card.getBoundingClientRect().top;"
        "const scrollDelta = currentTop - viewportAnchor.top;"
        "if (Math.abs(scrollDelta) < 1) { return; }"
        "const root = document.documentElement;"
        "const previousScrollBehavior = root.style.scrollBehavior;"
        "root.style.scrollBehavior = 'auto';"
        "window.scrollBy(0, scrollDelta);"
        "root.style.scrollBehavior = previousScrollBehavior;"
        "};"
        "let item = selectedCards[0].previousElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "item = item.previousElementSibling;"
        "}"
        "previousBreak = item;"
        "item = selectedCards[0].nextElementSibling;"
        "while (item && !item.classList.contains('ufa-free-row-break')) {"
        "item = item.nextElementSibling;"
        "}"
        "nextBreak = item;"
        "target = null;"
        "const selectionStartsRow = selectedCards[0].previousElementSibling === previousBreak;"
        "const selectionEndsRow = selectedCards[selectedCards.length - 1].nextElementSibling === nextBreak;"
        "const selectedSet = new Set(selectedCards);"
        "const rowCards = [];"
        "item = previousBreak ? previousBreak.nextElementSibling : board.firstElementChild;"
        "while (item && item !== nextBreak) {"
        "if (item.classList.contains('ufa-free-card')) { rowCards.push(item); }"
        "item = item.nextElementSibling;"
        "}"
        "const selectionFillsRow = rowCards.length === selectedCards.length"
        "&& rowCards.every(function(card) { return selectedSet.has(card); });"
        "const existingBoundary = position === 'above' && previousBreak && selectionStartsRow"
        "&& selectionEndsRow && selectionFillsRow"
        "? previousBreak"
        ": position === 'below' && nextBreak && selectionStartsRow"
        "&& selectionEndsRow && selectionFillsRow"
        "? nextBreak"
        ": null;"
        "if (existingBoundary) {"
        "board.querySelectorAll('.arrange-target').forEach(function(item) {"
        "item.classList.remove('arrange-target');"
        "});"
        "existingBoundary.classList.add('arrange-target');"
        "board.dataset.selectionCommitted = 'true';"
        f"{update_arrange_selection}"
        "return;"
        "}"
        "}"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "board.dataset.suppressFilterMutation = 'true';"
        "board.dataset.suppressFilterRun = 'true';"
        "window.requestAnimationFrame(function() {"
        "if (board.dataset.suppressFilterMutation === 'true') {"
        "board.dataset.suppressFilterMutation = 'false';"
        "}"
        "if (board.dataset.suppressFilterRun === 'true') {"
        "board.dataset.suppressFilterRun = 'false';"
        "}"
        "});"
        "const nextIndex = board.querySelectorAll('.ufa-free-row-break').length + 1;"
        "const breaker = document.createElement('div');"
        f"breaker.id = {json.dumps(board_id + '-row-break-')} + nextIndex + '-' + Date.now();"
        "breaker.className = 'ufa-free-board-item ufa-free-row-break arrange-target';"
        "breaker.dataset.arrangeLabel = 'Row break ' + nextIndex;"
        "board.querySelectorAll('.arrange-target').forEach(function(item) {"
        "item.classList.remove('arrange-target');"
        "});"
        "breaker.draggable = true;"
        "breaker.title = 'Drag this divider between possession groups';"
         "breaker.innerHTML = '<input type=\"text\" value=\"Pattern group ' + nextIndex + '\" aria-label=\"Row title\" />"
         "<button type=\"button\" class=\"ufa-free-row-overlay\" aria-label=\"Overlay all possessions in this row\">Overlay all</button>"
         "<button type=\"button\" class=\"ufa-free-row-overlay-clear\" aria-label=\"Clear overlay for all possessions in this row\">Clear overlay</button>"
        "<button type=\"button\" class=\"ufa-free-row-move-up\" aria-label=\"Move this row up\" title=\"Move row up\"><span class=\"ufa-row-icon ufa-row-icon-up ufa-row-icon-single\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-move-down\" aria-label=\"Move this row down\" title=\"Move row down\"><span class=\"ufa-row-icon ufa-row-icon-down ufa-row-icon-single\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-move-top\" aria-label=\"Move this row to the top\" title=\"Move row to top\"><span class=\"ufa-row-icon ufa-row-icon-up\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-move-bottom\" aria-label=\"Move this row to the bottom\" title=\"Move row to bottom\"><span class=\"ufa-row-icon ufa-row-icon-down\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-remove\" aria-label=\"Remove row break\">Remove</button>';"
        f"breaker.ondragstart = function(event) {{ {drag_start} }};"
        f"breaker.ondragend = function() {{ {drag_end} }};"
        f"breaker.ondragover = function(event) {{ {drag_over} }};"
        f"breaker.ondragleave = function() {{ {drag_leave} }};"
        f"breaker.ondrop = function(event) {{ {drop} }};"
        f"breaker.onclick = function() {{ {select_arrange_target} }};"
        f"{protect_row_controls}"
        "breaker.querySelector('input').onclick = function(event) { event.stopPropagation(); };"
        "breaker.querySelector('.ufa-free-row-overlay').onclick = function(event) {"
        "event.preventDefault();"
        "event.stopPropagation();"
         f"{overlay_row}"
         "};"
         "breaker.querySelector('.ufa-free-row-overlay-clear').onclick = function(event) {"
         "event.preventDefault();"
         "event.stopPropagation();"
         f"{clear_overlay_row}"
         "};"
        f"{bind_row_move_controls}"
        "breaker.querySelector('.ufa-free-row-remove').onclick = function(event) {"
        "event.preventDefault();"
        "event.stopPropagation();"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "breaker.remove();"
        "};"
        "if (selectedCards.length) {"
        "const firstSelectedCard = selectedCards[0];"
        "const lastSelectedCard = selectedCards[selectedCards.length - 1];"
        "const selectedFragment = document.createDocumentFragment();"
        "if (position === 'above') {"
        "board.insertBefore(breaker, firstSelectedCard);"
        "selectedCards.forEach(function(card) { selectedFragment.appendChild(card); });"
        "breaker.after(selectedFragment);"
        "} else if (position === 'below') {"
        "lastSelectedCard.after(breaker);"
        "selectedCards.forEach(function(card) { selectedFragment.appendChild(card); });"
        "board.insertBefore(selectedFragment, breaker);"
        "} else {"
        "selectedFragment.appendChild(breaker);"
        "selectedCards.forEach(function(card) { selectedFragment.appendChild(card); });"
        "board.appendChild(selectedFragment);"
        "}"
        "} else {"
        "if (position === 'above') {"
        "if (target) { board.insertBefore(breaker, target); }"
        "else { board.insertBefore(breaker, board.firstChild); }"
        "} else if (position === 'below') {"
        "if (target) { target.after(breaker); }"
        "else { board.appendChild(breaker); }"
        "} else {"
        "board.appendChild(breaker);"
        "}"
        "}"
        "allSelectedCards.forEach(function(card) {"
        "if (card.hidden) { card.classList.remove('row-selected'); }"
        "});"
        "if (selectedCards.length) { board.dataset.selectionCommitted = 'true'; }"
        f"{update_arrange_selection}"
        "if (selectedCards.length) {"
        "restoreViewportAnchor();"
        "window.requestAnimationFrame(restoreViewportAnchor);"
        "}"
    )
    clear_row_breaks = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        "if (!board) { return; }"
        "const rowBreaks = Array.from(board.querySelectorAll('.ufa-free-row-break'));"
        "if (!rowBreaks.length) { return; }"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "rowBreaks.forEach(function(breaker) {"
        "breaker.remove();"
        "});"
    )
    save_arrangement = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        f"const status = document.getElementById({json.dumps(export_status_id)});"
        f"const output = document.getElementById({json.dumps(export_text_id)});"
        "if (!board) { return; }"
        "const groups = [];"
        "let currentGroup = {"
        "title: 'Group 1', possessions: [], has_break: false, auto_unsorted: false"
        "};"
        "Array.from(board.children).forEach(function(item) {"
        "if (item.classList.contains('ufa-free-row-break')) {"
        "if (currentGroup.has_break || currentGroup.possessions.length) {"
        "groups.push(currentGroup);"
        "}"
        "const titleInput = item.querySelector('input');"
        "const title = titleInput && titleInput.value.trim() ? titleInput.value.trim() : item.dataset.arrangeLabel;"
        "currentGroup = {"
        "title: title || 'Pattern group ' + (groups.length + 2),"
        "possessions: [],"
        "has_break: true,"
        "auto_unsorted: item.dataset.autoUnsorted === 'true'"
        "};"
        "return;"
        "}"
        "if (!item.classList.contains('ufa-free-card')) { return; }"
        "const checkbox = item.querySelector('.ufa-free-overlay-check');"
        "currentGroup.possessions.push({"
        "possession_id: item.dataset.possessionId,"
        "label: item.dataset.cardLabel,"
        "overlay_selected: Boolean(checkbox && checkbox.checked)"
        "});"
        "});"
        "if (currentGroup.has_break || currentGroup.possessions.length) {"
        "groups.push(currentGroup);"
        "}"
        "const cardCount = groups.reduce(function(total, group) {"
        "return total + group.possessions.length;"
        "}, 0);"
        "const payload = {"
        "saved_at: new Date().toISOString(),"
        "cards_shown: cardCount,"
        f"filtered_possessions: {len(possessions)},"
        "groups: groups.map(function(group, index) {"
        "return {"
        "group_index: index + 1,"
        "title: group.title,"
        "auto_unsorted: Boolean(group.auto_unsorted),"
        "possessions: group.possessions"
        "};"
        "})"
        "};"
        "const text = JSON.stringify(payload, null, 2);"
        "if (output) {"
        "output.textContent = text;"
        "output.style.display = 'block';"
        "}"
        "let localSaveMessage = 'Saved in this browser';"
        "try {"
        f"window.localStorage.setItem({json.dumps(storage_key)}, text);"
        "} catch (error) {"
        "localSaveMessage = 'Browser storage blocked; exported JSON';"
        "}"
        "if (status) {"
        "status.textContent = localSaveMessage + ': ' + cardCount + ' cards across ' + groups.length + ' groups. JSON shown below.';"
        "status.scrollIntoView({block: 'nearest'});"
        "}"
        "try {"
        "const blob = new Blob([text], {type: 'application/json'});"
        "const url = URL.createObjectURL(blob);"
        "const link = document.createElement('a');"
        "link.href = url;"
        f"link.download = 'ufa-free-arrange-{board_key}-' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.json';"
        "document.body.appendChild(link);"
        "link.click();"
        "link.remove();"
        "URL.revokeObjectURL(url);"
        "} catch (error) {"
        "if (status) {"
        "status.textContent = localSaveMessage + ': ' + cardCount + ' cards across ' + groups.length + ' groups. Download was blocked; JSON shown below.';"
        "}"
        "}"
    )
    load_arrangement = (
        f"const board = document.getElementById({json.dumps(board_id)});"
        f"const status = document.getElementById({json.dumps(export_status_id)});"
        f"const output = document.getElementById({json.dumps(export_text_id)});"
        "if (!board) { return; }"
        "let text = null;"
        "try {"
        f"text = window.localStorage.getItem({json.dumps(storage_key)});"
        f"if (!text && {json.dumps(legacy_storage_key)} !== {json.dumps(storage_key)}) {{"
        f"text = window.localStorage.getItem({json.dumps(legacy_storage_key)});"
        "if (text) {"
        f"window.localStorage.setItem({json.dumps(storage_key)}, text);"
        "}"
        "}"
        "} catch (error) {"
        "text = null;"
        "}"
        "if (!text) {"
        "if (status) { status.textContent = 'No saved arrangement found for this board.'; }"
        "return;"
        "}"
        "let payload = null;"
        "try { payload = JSON.parse(text); }"
        "catch (error) {"
        "if (status) { status.textContent = 'Saved arrangement could not be read.'; }"
        "return;"
        "}"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "const cardMap = {};"
        "board.querySelectorAll('.ufa-free-card').forEach(function(card) {"
        "cardMap[card.dataset.possessionId] = card;"
        "card.classList.remove('arrange-target');"
        "card.classList.remove('row-selected');"
        "});"
        "board.dataset.selectionCommitted = 'false';"
        "board.innerHTML = '';"
        "function makeBreak(title, index, autoUnsorted) {"
        "const breaker = document.createElement('div');"
        f"breaker.id = {json.dumps(board_id + '-row-break-')} + index + '-' + Date.now();"
        "breaker.className = 'ufa-free-board-item ufa-free-row-break';"
        "breaker.dataset.arrangeLabel = title || ('Pattern group ' + index);"
        "if (autoUnsorted) { breaker.dataset.autoUnsorted = 'true'; }"
        "breaker.draggable = true;"
        "breaker.title = 'Drag this divider between possession groups';"
         "breaker.innerHTML = '<input type=\"text\" aria-label=\"Row title\" />"
         "<button type=\"button\" class=\"ufa-free-row-overlay\" aria-label=\"Overlay all possessions in this row\">Overlay all</button>"
         "<button type=\"button\" class=\"ufa-free-row-overlay-clear\" aria-label=\"Clear overlay for all possessions in this row\">Clear overlay</button>"
         "<button type=\"button\" class=\"ufa-free-row-move-up\" aria-label=\"Move this row up\" title=\"Move row up\"><span class=\"ufa-row-icon ufa-row-icon-up ufa-row-icon-single\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-move-down\" aria-label=\"Move this row down\" title=\"Move row down\"><span class=\"ufa-row-icon ufa-row-icon-down ufa-row-icon-single\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-move-top\" aria-label=\"Move this row to the top\" title=\"Move row to top\"><span class=\"ufa-row-icon ufa-row-icon-up\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-move-bottom\" aria-label=\"Move this row to the bottom\" title=\"Move row to bottom\"><span class=\"ufa-row-icon ufa-row-icon-down\" aria-hidden=\"true\"></span></button>"
        "<button type=\"button\" class=\"ufa-free-row-remove\" aria-label=\"Remove row break\">Remove</button>';"
        "breaker.querySelector('input').value = title || ('Pattern group ' + index);"
        f"breaker.ondragstart = function(event) {{ {drag_start} }};"
        f"breaker.ondragend = function() {{ {drag_end} }};"
        f"breaker.ondragover = function(event) {{ {drag_over} }};"
        f"breaker.ondragleave = function() {{ {drag_leave} }};"
        f"breaker.ondrop = function(event) {{ {drop} }};"
        f"breaker.onclick = function() {{ {select_arrange_target} }};"
        f"{protect_row_controls}"
        "breaker.querySelector('input').onclick = function(event) { event.stopPropagation(); };"
        "breaker.querySelector('.ufa-free-row-overlay').onclick = function(event) {"
        "event.preventDefault();"
        "event.stopPropagation();"
         f"{overlay_row}"
         "};"
         "breaker.querySelector('.ufa-free-row-overlay-clear').onclick = function(event) {"
         "event.preventDefault();"
         "event.stopPropagation();"
         f"{clear_overlay_row}"
         "};"
        f"{bind_row_move_controls}"
        "breaker.querySelector('.ufa-free-row-remove').onclick = function(event) {"
        "event.preventDefault();"
        "event.stopPropagation();"
        f"{ensure_undo}"
        "board._ufaPushUndo();"
        "breaker.remove();"
        "};"
        "return breaker;"
        "}"
        "const usedIds = new Set();"
        "(payload.groups || []).forEach(function(group, groupIndex) {"
        "const title = group.title || ('Pattern group ' + (groupIndex + 1));"
        "if (groupIndex > 0 || title !== 'Group 1') {"
        "board.appendChild(makeBreak(title, groupIndex + 1, Boolean(group.auto_unsorted)));"
        "}"
        "(group.possessions || []).forEach(function(possession) {"
        "const card = cardMap[possession.possession_id];"
        "if (!card) { return; }"
        "const checkbox = card.querySelector('.ufa-free-overlay-check');"
        "if (checkbox) { checkbox.checked = Boolean(possession.overlay_selected); }"
        "board.appendChild(card);"
        "usedIds.add(possession.possession_id);"
        "});"
        "});"
        "let stagingBreak = Array.from(board.querySelectorAll('.ufa-free-row-break')).find(function(candidate) {"
        "return candidate.dataset.autoUnsorted === 'true';"
        "}) || null;"
        "if (stagingBreak) {"
        "const stagingGroup = [stagingBreak];"
        "let stagingItem = stagingBreak.nextElementSibling;"
        "while (stagingItem && !stagingItem.classList.contains('ufa-free-row-break')) {"
        "stagingGroup.push(stagingItem);"
        "stagingItem = stagingItem.nextElementSibling;"
        "}"
        "const stagingFragment = document.createDocumentFragment();"
        "stagingGroup.forEach(function(item) { stagingFragment.appendChild(item); });"
        "board.appendChild(stagingFragment);"
        "} else {"
        "stagingBreak = makeBreak('Unsorted: new cards', (payload.groups || []).length + 1, true);"
        "board.appendChild(stagingBreak);"
        "}"
        "Object.keys(cardMap).forEach(function(possessionId) {"
        "if (!usedIds.has(possessionId)) { board.appendChild(cardMap[possessionId]); }"
        "});"
        "if (output) {"
        "output.textContent = text;"
        "output.style.display = 'block';"
        "}"
        "if (status) {"
        "status.textContent = 'Loaded saved arrangement from this browser.';"
        "}"
        f"{update_overlay}"
        f"{update_arrange_selection}"
    )

    cards = []
    overlay_groups = []
    for card_index, (_, row) in enumerate(frame.iterrows(), start=1):
        path = path_lookup.get(row["possession_id"])
        if path is None or path.empty:
            continue

        card_id = f"{board_id}-card-{card_index}"
        overlay_path_id = f"{card_id}-overlay"
        overlay_input_id = f"{card_id}-overlay-check"
        aec_per_throw = _format_browser_number(row.get("aec_per_throw"), digits=3)
        total_aec = _format_browser_number(row.get("total_aec"), digits=3)
        outcome = escape(str(row.get("outcome", "unknown")).title())
        line_type = escape(_line_type_label(row.get("line_type")))
        outcome_value = escape(
            str(row.get("outcome", "unknown")).strip().lower(), quote=True
        )
        line_type_value = escape(
            str(row.get("line_type", "unknown")).strip().lower(), quote=True
        )
        throw_count_value = int(row.get("throw_count", len(path)))
        huck_count_value = pd.to_numeric(row.get("huck_count", 0), errors="coerce")
        huck_count_value = 0 if pd.isna(huck_count_value) else int(huck_count_value)
        shape_label = escape(_shape_cluster_label(row))
        meta = (
            f"{escape(str(row.get('GameID', '-')))} | "
            f"Q{row.get('game_quarter', '-')} P{row.get('quarter_point', '-')} | "
            f"{int(row.get('throw_count', len(path)))} throws"
        )
        card_title = (
            f"{outcome} {line_type} - {meta} - "
            f"aEC/T {aec_per_throw} - total {total_aec} - {shape_label}"
        )
        card_label = (
            f"{card_index}. {outcome} {line_type} | {meta} | "
            f"aEC/T {aec_per_throw} | total {total_aec} | {shape_label}"
        )
        overlay_color = overlay_palette[(card_index - 1) % len(overlay_palette)]
        overlay_groups.append(
            render_possession_overlay_group(
                path,
                color=overlay_color,
                width=overlay_width,
                height=overlay_height,
                group_id=overlay_path_id,
                hidden=True,
                normalize_shape=False,
            )
        )
        cards.append(
            f"""
            <div id="{card_id}" class="ufa-free-board-item ufa-free-card" draggable="true"
              title="{escape(card_title, quote=True)}"
              data-possession-id="{escape(str(row['possession_id']), quote=True)}"
              data-card-label="{escape(card_label, quote=True)}"
              data-arrange-label="{escape(card_label, quote=True)}"
              data-outcome="{outcome_value}"
              data-line-type="{line_type_value}"
              data-throw-count="{throw_count_value}"
              data-huck-count="{huck_count_value}"
              data-aec-per-throw="{escape(str(row.get('aec_per_throw', '')), quote=True)}"
              data-total-aec="{escape(str(row.get('total_aec', '')), quote=True)}"
              ondragstart="{escape(drag_start, quote=True)}"
              ondragend="{escape(drag_end, quote=True)}"
              ondragover="{escape(drag_over, quote=True)}"
              ondragleave="{escape(drag_leave, quote=True)}"
              ondrop="{escape(drop, quote=True)}"
              onclick="{escape(toggle_arrange_selection, quote=True)}">
              {render_mini_possession_svg(path)}
              <div class="ufa-free-card-meta">
                <label class="ufa-free-overlay-toggle"
                  for="{overlay_input_id}"
                  onclick="event.stopPropagation();">
                  <input id="{overlay_input_id}" class="ufa-free-overlay-check" type="checkbox"
                    data-overlay-path="{overlay_path_id}"
                    onchange="{escape(update_overlay, quote=True)}" />
                  Overlay
                </label>
              </div>
            </div>
            """
        )

    shown_count = len(cards)
    recovery_controls = ""
    recovery_data_attributes = ""
    if persistence_key:
        recovery_controls = f"""
            <details class="ufa-free-recovery-menu">
              <summary>Recovery</summary>
              <div class="ufa-free-recovery-popover">
                <label class="ufa-free-recovery-control">
                  Version
                  <select id="{recovery_history_id}" class="ufa-free-recovery-history"
                    aria-label="Recovery history" disabled>
                    <option value="">No recovery snapshots</option>
                  </select>
                </label>
                <button id="{recovery_restore_id}" class="ufa-free-recovery-restore"
                  type="button" disabled>
                  Restore
                </button>
                <span id="{recovery_status_id}" class="ufa-free-recovery-status"
                  role="status" aria-live="polite" title="Autosave is ready">
                  Saved
                </span>
              </div>
            </details>
        """
        recovery_data_attributes = (
            f' data-checkpoint-storage-key="{escape(storage_key, quote=True)}"'
            f' data-recovery-storage-key="{escape(recovery_storage_key, quote=True)}"'
        )
    overlay_panel = f"""
      <aside class="ufa-free-overlay-panel">
        <div class="ufa-gallery-kicker">Selected Overlay</div>
        <div id="{overlay_count_id}" class="ufa-free-board-meta">
          Select cards to overlay
        </div>
        <div class="ufa-free-overlay-note">
          Exact field overlay of selected possessions.
        </div>
        {render_empty_overlay_field(
            overlay_layer_id,
            width=overlay_width,
            height=overlay_height,
            overlay_groups=''.join(overlay_groups),
        )}
        <button class="ufa-free-overlay-clear"
          type="button"
          onclick="{escape(clear_overlay, quote=True)}">
          Clear overlay
        </button>
      </aside>
    """
    board_wrap_class = "ufa-free-board-wrap"
    if external_controls:
        board_wrap_class += " with-external-controls"
    board_html = f"""
    <div id="{wrap_id}" class="{board_wrap_class}">
      <div class="ufa-free-board-layout{' with-inline-overlay' if include_overlay else ''}">
        <div class="ufa-free-board-content">
          <div class="ufa-gallery-kicker">Free Arrange</div>
          <div class="ufa-free-board-meta">
            {shown_count} cards shown of {len(possessions)} filtered possessions
          </div>
          <div class="ufa-free-board-actions">
            <label>
              <span class="ufa-free-placement-label">Placement</span>
              <select id="{insert_position_id}" class="ufa-free-insert-position"
                aria-label="Row break placement">
                <option value="above">Above selected</option>
                <option value="below">Below selected</option>
                <option value="bottom">At bottom</option>
              </select>
            </label>
            <button class="ufa-free-insert-row-break" type="button"
              onclick="{escape(insert_row_break, quote=True)}">
              Insert row break
            </button>
            <button class="ufa-free-clear-selection" type="button"
              onclick="{escape(clear_arrange_selection, quote=True)}">
              Clear selection
            </button>
            <button class="primary ufa-free-save-arrangement" type="button"
              onclick="{escape(save_arrangement, quote=True)}">
              Save arrangement
            </button>
            <button class="ufa-free-load-arrangement" type="button"
              onclick="{escape(load_arrangement, quote=True)}">
              Load saved
            </button>
            <button class="ufa-free-clear-row-breaks" type="button"
              onclick="{escape(clear_row_breaks, quote=True)}">
              Clear row breaks
            </button>
            <button id="{undo_button_id}" class="ufa-free-undo-arrangement"
              type="button" disabled title="Nothing to undo"
              aria-label="Undo last arrangement change"
              onclick="{escape(undo_arrangement, quote=True)}">
              Undo
            </button>
            {recovery_controls}
            <span id="{arrange_selection_count_id}"
              class="ufa-free-arrange-selection-count"
              title="Click cards to select for a row break">
              0 selected
            </span>
          </div>
          <div id="{export_status_id}" class="ufa-free-export-status"></div>
          <pre id="{export_text_id}" class="ufa-free-export-text"
            aria-label="Saved arrangement JSON"></pre>
          <div id="{board_id}" class="ufa-free-board"{recovery_data_attributes}
            ondragover="event.preventDefault();">
            {''.join(cards)}
          </div>
        </div>
        {f'<div class="ufa-free-overlay-shell">{overlay_panel}</div>' if include_overlay else ''}
      </div>
    </div>
    """
    if return_overlay:
        return board_html, overlay_panel
    return board_html


def render_possession_browser_summary(possession, path):
    game_id = escape(str(possession.get("GameID", "-")))
    team_id = escape(str(possession.get("team_id", "-")).title())
    side = "Home" if bool(possession.get("is_home_team", False)) else "Away"
    line_type = escape(_line_type_label(possession.get("line_type")))
    outcome_raw = str(possession.get("outcome", "unknown")).lower()
    outcome = escape(outcome_raw.title())
    outcome_class = f"outcome-{escape(outcome_raw)}"
    quarter = possession.get("game_quarter", "-")
    quarter_point = possession.get("quarter_point", "-")
    possession_num = possession.get("possession_num", "-")
    start_y = _format_browser_number(possession.get("start_y"), digits=1)
    end_y = _format_browser_number(possession.get("end_y"), digits=1)
    field_progress = _format_browser_number(possession.get("field_progress"), digits=1)
    total_yards = _format_browser_number(possession.get("total_yards"), digits=1)
    total_throw_distance = _format_browser_number(
        possession.get("total_throw_distance"),
        digits=1,
    )
    throw_count = int(possession.get("throw_count", len(path)))
    total_aec = _format_browser_number(possession.get("total_aec"), digits=3)
    aec_per_throw = _format_browser_number(possession.get("aec_per_throw"), digits=3)
    huck_count = int(possession.get("huck_count", 0))
    reset_count = int(possession.get("reset_count", 0))
    shape_label = escape(_shape_cluster_label(possession))
    possession_shape_label = escape(_describe_single_possession_shape(possession))
    width_used = _format_browser_number(possession.get("shape_width"), digits=1)
    side_switches = _format_browser_number(
        possession.get("shape_side_switches"),
        digits=0,
    )
    directness = _format_browser_percent(possession.get("shape_directness"))
    middle_usage = _format_browser_percent(possession.get("shape_middle_third_share"))
    sideline_usage = _format_browser_percent(possession.get("shape_sideline_share"))

    return f"""
    <div class="ufa-browser-summary">
      <style>
        .ufa-browser-summary {{
          color: #0b1a33;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 13px;
          line-height: 1.45;
          width: 100%;
        }}
        .ufa-browser-summary h3 {{
          font-size: 20px;
          margin: 0 0 6px;
          color: #0b1a33;
          letter-spacing: 0;
        }}
        .ufa-browser-summary .summary-subtitle {{
          color: #52637a;
          font-size: 12px;
          margin-bottom: 8px;
        }}
        .ufa-browser-summary .chips {{
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
          margin: 8px 0;
        }}
        .ufa-browser-summary .chip {{
          display: inline-flex;
          align-items: center;
          border: 1px solid #d9e1ea;
          border-radius: 999px;
          padding: 2px 8px;
          background: #f7fafc;
          color: #253b58;
          font-size: 11px;
          font-weight: 700;
        }}
        .ufa-browser-summary .outcome-goal {{
          border-color: #b9dfc6;
          background: #ecf8f0;
          color: #196236;
        }}
        .ufa-browser-summary .outcome-turnover {{
          border-color: #f1c2b8;
          background: #fff1ee;
          color: #a33b27;
        }}
        .ufa-browser-summary .shape-line {{
          margin: 6px 0;
          color: #263c58;
          font-size: 12px;
        }}
        .ufa-browser-summary .meta {{
          border-top: 1px solid #d9e1ea;
          margin-top: 12px;
          padding-top: 8px;
        }}
        .ufa-browser-summary .row {{
          display: flex;
          justify-content: space-between;
          gap: 16px;
          border-bottom: 1px solid #edf1f5;
          padding: 5px 0;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        }}
        .ufa-browser-summary .label {{ color: #637188; }}
        .ufa-browser-summary .value {{ font-weight: 700; text-align: right; }}
      </style>
      <h3>{team_id}</h3>
      <div class="summary-subtitle">{game_id} - Q{quarter}, point {quarter_point}, possession {possession_num}</div>
      <div class="chips">
        <span class="chip {outcome_class}">{outcome}</span>
        <span class="chip">{side}</span>
        <span class="chip">{line_type}</span>
      </div>
      <div class="shape-line"><b>Group:</b> {shape_label}</div>
      <div class="shape-line"><b>This possession:</b> {possession_shape_label}</div>
      <div class="meta">
        <div class="row"><span class="label">Throws</span><span class="value">{throw_count}</span></div>
        <div class="row"><span class="label">Start Y</span><span class="value">{start_y}</span></div>
        <div class="row"><span class="label">End Y</span><span class="value">{end_y}</span></div>
        <div class="row"><span class="label">Net Y progress</span><span class="value">{field_progress}</span></div>
        <div class="row"><span class="label">Field yards</span><span class="value">{total_yards}</span></div>
        <div class="row"><span class="label">Total pass distance</span><span class="value">{total_throw_distance}</span></div>
        <div class="row"><span class="label">Total aEC</span><span class="value">{total_aec}</span></div>
        <div class="row"><span class="label">aEC / throw</span><span class="value">{aec_per_throw}</span></div>
        <div class="row"><span class="label">Hucks</span><span class="value">{huck_count}</span></div>
        <div class="row"><span class="label">Resets</span><span class="value">{reset_count}</span></div>
        <div class="row"><span class="label">Width used</span><span class="value">{width_used}</span></div>
        <div class="row"><span class="label">Side switches</span><span class="value">{side_switches}</span></div>
        <div class="row"><span class="label">Directness</span><span class="value">{directness}</span></div>
        <div class="row"><span class="label">Middle usage</span><span class="value">{middle_usage}</span></div>
        <div class="row"><span class="label">Sideline usage</span><span class="value">{sideline_usage}</span></div>
      </div>
    </div>
    """


def _possession_browser_css():
    return """
    <style>
      .ufa-possession-browser-shell {
        box-sizing: border-box;
        width: 100%;
        max-width: 1680px;
        overflow: visible !important;
        border: 1px solid #d8e1eb;
        border-radius: 8px;
        background: #f5f8fb;
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.10);
        padding: 14px;
      }
      .ufa-browser-topbar {
        box-sizing: border-box;
        max-width: 1680px;
        margin: 0 0 8px;
        justify-content: flex-start;
      }
      .ufa-browser-topbar .widget-button {
        border-radius: 6px;
      }
      .ufa-browser-sticky-toolbar {
        position: sticky !important;
        top: 8px;
        z-index: 2000;
        box-sizing: border-box;
        align-self: flex-start;
        width: calc(100% - 286px) !important;
        max-width: 1380px;
        margin: 0 0 8px;
        overflow: visible !important;
      }
      .ufa-browser-sticky-toolbar .widget-html-content {
        width: 100% !important;
        overflow: visible !important;
      }
      .ufa-free-external-actions {
        box-sizing: border-box;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        width: 100%;
        padding: 8px 12px;
        border: 1px solid #cbd6e2;
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.98);
        box-shadow: 0 5px 14px rgba(15, 23, 42, 0.12);
        color: #52637a;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .ufa-free-external-actions label {
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }
      .ufa-free-external-actions select,
      .ufa-free-external-actions button {
        border: 1px solid #cbd6e2;
        border-radius: 6px;
        background: #ffffff;
        color: #10233f;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        padding: 6px 10px;
        text-transform: none;
      }
      .ufa-free-external-actions button {
        cursor: pointer;
      }
      .ufa-free-external-actions button.primary {
        border-color: #2b8ce6;
        background: #2b8ce6;
        color: #ffffff;
      }
      .ufa-browser-controls {
        box-sizing: border-box;
        border: 1px solid #d8e1eb;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        padding: 16px;
      }
      .ufa-browser-title {
        margin-bottom: 10px;
        color: #10233f;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .ufa-browser-title .kicker {
        color: #667792;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .ufa-browser-title h2 {
        margin: 2px 0 2px;
        color: #10233f;
        font-size: 22px;
        letter-spacing: 0;
      }
      .ufa-browser-title .subtitle {
        color: #5a6b82;
        font-size: 13px;
      }
      .ufa-possession-browser-shell .widget-label {
        color: #42526a;
        font-weight: 700;
      }
      .ufa-possession-browser-shell select,
      .ufa-possession-browser-shell input {
        border-color: #cbd6e2;
        border-radius: 6px;
      }
      .ufa-browser-nav {
        align-items: center;
        margin: 2px 0 8px;
      }
      .ufa-browser-nav .widget-button {
        border-radius: 6px;
      }
      .ufa-browser-count {
        color: #10233f;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      }
      .ufa-browser-field-panel {
        min-width: 540px;
        width: 100%;
        flex: 1 1 auto;
        overflow: visible !important;
      }
      .ufa-browser-field-panel.gallery {
        min-width: 900px;
        width: 100%;
        flex: 1 1 auto;
      }
      .ufa-browser-field-panel .widget-html,
      .ufa-browser-field-panel .widget-html-content {
        width: 100%;
        overflow: visible !important;
      }
      .ufa-browser-main-panel {
        align-items: flex-start;
        gap: 16px;
        flex: 1 1 auto;
        min-width: 0;
        overflow: visible !important;
      }
      .ufa-browser-info-panel {
        box-sizing: border-box;
        width: 340px;
        border: 1px solid #d8e1eb;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        padding: 16px;
      }
      .ufa-browser-status {
        color: #30435f;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 6px 0 10px;
      }
      .ufa-team-browser-controls {
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
      }
      .ufa-shape-gallery {
        box-sizing: border-box;
        border: 1px solid #d8e1eb;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        padding: 16px;
        width: 100%;
        max-width: 1280px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #10233f;
      }
      .ufa-gallery-kicker {
        color: #667792;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
      }
      .ufa-gallery-section {
        border-top: 1px solid #edf1f5;
        padding-top: 12px;
        margin-top: 12px;
      }
      .ufa-gallery-section:first-of-type {
        border-top: none;
        padding-top: 0;
      }
      .ufa-gallery-section-title {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 8px;
      }
      .ufa-gallery-section-title h3 {
        margin: 0;
        font-size: 14px;
        color: #10233f;
      }
      .ufa-gallery-section-title span {
        color: #667792;
        font-size: 12px;
      }
      .ufa-gallery-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(145px, 1fr));
        gap: 10px;
      }
      .ufa-gallery-card {
        min-width: 0;
        border: 1px solid #dfe7ef;
        border-radius: 8px;
        background: #f8fbff;
        padding: 8px;
      }
      .ufa-mini-svg {
        display: block;
        margin: 0 auto 6px;
        background: #f5f8f5;
        border-radius: 4px;
      }
      .ufa-mini-field {
        fill: #86d973;
        stroke: #071019;
        stroke-width: 1.8;
      }
      .ufa-mini-yard-line {
        stroke: #071019;
        stroke-width: 1.2;
      }
      .ufa-mini-center-dot,
      .ufa-mini-dot {
        fill: #071019;
      }
      .ufa-mini-throw-line {
        stroke: #071019;
        stroke-width: 1.7;
        stroke-linecap: round;
      }
      .ufa-gallery-card-meta {
        color: #40516a;
        font-size: 11px;
        line-height: 1.35;
      }
      .ufa-gallery-card-meta b {
        color: #10233f;
      }
      .ufa-gallery-card-meta span {
        color: #667792;
      }
      .ufa-shape-gallery .empty {
        color: #667792;
        padding: 16px;
      }
      .ufa-free-board-wrap {
        box-sizing: border-box;
        border: 1px solid #d8e1eb;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        padding: 16px;
        width: 100%;
        max-width: none;
        overflow: visible;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #10233f;
      }
      .ufa-free-board-meta {
        color: #667792;
        font-size: 12px;
        margin: -4px 0 12px;
      }
      .ufa-free-board-layout {
        box-sizing: border-box;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        position: relative;
        width: 100%;
        overflow: visible;
      }
      .ufa-free-board-layout.with-inline-overlay {
        padding-right: 284px;
      }
      .ufa-free-board-content {
        flex: 1 1 auto;
        min-width: 0;
      }
      .ufa-free-board-actions {
        box-sizing: border-box;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin: -4px 0 12px;
        padding: 8px 0;
        border-bottom: 1px solid #d8e1eb;
        background: #ffffff;
      }
      /* Keep manual checkpoints available to autosave/recovery, but hide the controls for now. */
      .ufa-free-save-arrangement,
      .ufa-free-load-arrangement,
      .ufa-free-external-save-arrangement,
      .ufa-free-external-load-arrangement {
        display: none !important;
      }
      .ufa-free-board-actions label {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #52637a;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .ufa-free-board-actions select {
        border: 1px solid #cbd6e2;
        border-radius: 6px;
        background: #ffffff;
        color: #10233f;
        font-size: 12px;
        letter-spacing: 0;
        padding: 5px 8px;
        text-transform: none;
      }
      .ufa-free-board-actions button {
        border: 1px solid #cbd6e2;
        border-radius: 6px;
        background: #ffffff;
        color: #10233f;
        cursor: pointer;
        font-weight: 700;
        padding: 6px 10px;
      }
      .ufa-free-board-actions button.primary {
        border-color: #2b8ce6;
        background: #2b8ce6;
        color: #ffffff;
      }
      .ufa-free-board-actions .ufa-free-recovery-menu {
        position: relative;
        flex: 0 0 auto;
      }
      .ufa-free-board-actions .ufa-free-recovery-menu summary {
        min-height: 30px;
        box-sizing: border-box;
        display: inline-flex;
        align-items: center;
        border: 1px solid #cbd6e2;
        border-radius: 6px;
        background: #ffffff;
        color: #10233f;
        cursor: pointer;
        font-size: 13px;
        font-weight: 700;
        list-style: none;
        padding: 6px 10px;
      }
      .ufa-free-board-actions .ufa-free-recovery-menu summary::-webkit-details-marker {
        display: none;
      }
      .ufa-free-board-actions .ufa-free-recovery-menu summary::after {
        content: 'v';
        margin-left: 7px;
        color: #52637a;
        font-size: 10px;
      }
      .ufa-free-board-actions .ufa-free-recovery-menu[open] summary::after {
        content: '^';
      }
      .ufa-free-board-actions .ufa-free-recovery-popover {
        position: absolute;
        top: calc(100% + 8px);
        right: 0;
        z-index: 220;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 390px;
        padding: 10px;
        border: 1px solid #c4d0dd;
        border-radius: 7px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(15, 35, 58, 0.18);
      }
      .ufa-free-board-actions .ufa-free-recovery-control {
        padding-left: 0;
        border-left: 0;
      }
      .ufa-free-board-actions .ufa-free-recovery-history {
        min-width: 205px;
        max-width: 240px;
      }
      .ufa-free-board-actions button:disabled,
      .ufa-free-board-actions select:disabled {
        cursor: default;
        opacity: 0.55;
      }
      .ufa-free-recovery-status {
        color: #2f7247;
        font-size: 11px;
        font-weight: 750;
        white-space: nowrap;
      }
      .ufa-free-recovery-status.save-error {
        color: #a23f2c;
      }
      .ufa-free-board-wrap.with-external-controls .ufa-free-board-actions {
        display: none;
      }
      .ufa-free-export-status {
        min-height: 18px;
        color: #52637a;
        font-size: 11px;
        margin: -6px 0 10px;
      }
      .ufa-free-export-text {
        box-sizing: border-box;
        display: none;
        width: min(100%, 980px);
        min-height: 140px;
        margin: 0 0 12px;
        border: 1px solid #cbd6e2;
        border-radius: 6px;
        background: #f8fbff;
        color: #10233f;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 11px;
        line-height: 1.35;
        overflow: auto;
        padding: 8px;
        white-space: pre-wrap;
      }
      .ufa-free-board {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: start;
        justify-content: flex-start;
      }
      .ufa-free-card {
        flex: 0 0 126px;
        min-width: 0;
        border: 1px solid #dfe7ef;
        border-radius: 8px;
        background: #f8fbff;
        padding: 7px;
        cursor: grab;
        transition: box-shadow 120ms ease, transform 120ms ease, border-color 120ms ease;
      }
      .ufa-free-card:active {
        cursor: grabbing;
      }
      .ufa-free-card.dragging {
        opacity: 0.55;
        transform: scale(0.98);
      }
      .ufa-free-card.group-dragging {
        opacity: 0.42;
        transform: scale(0.98);
      }
      .ufa-multi-drag-preview {
        position: fixed;
        top: 0;
        left: 0;
        z-index: 2147483647;
        display: grid;
        gap: 6px;
        padding: 6px;
        border: 2px solid #2b8ce6;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(15, 35, 63, 0.24);
        pointer-events: none;
        will-change: left, top;
      }
      .ufa-multi-drag-preview .ufa-mini-svg {
        display: block;
        margin: 0;
        border-radius: 3px;
      }
      .ufa-multi-drag-preview.ufa-single-drag-preview {
        border-color: rgba(43, 140, 230, 0.78);
        background: transparent;
        box-shadow: none;
        opacity: 0.52;
      }
      .ufa-multi-drag-preview.ufa-single-drag-preview .ufa-mini-svg {
        background: transparent;
      }
      body.ufa-custom-group-drag-active,
      body.ufa-custom-group-drag-active * {
        cursor: grabbing !important;
        user-select: none !important;
      }
      .ufa-free-card.drop-target {
        border-color: #2b8ce6;
        box-shadow: 0 0 0 3px rgba(43, 140, 230, 0.16);
      }
      .ufa-free-board-item.arrange-target {
        border-color: #2b8ce6;
        box-shadow: 0 0 0 3px rgba(43, 140, 230, 0.22);
      }
      .ufa-free-row-break {
        box-sizing: border-box;
        flex: 1 0 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        min-height: 22px;
        border: 1px dashed #9fb2c8;
        border-radius: 6px;
        background: #f3f7fb;
        color: #52637a;
        cursor: grab;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .ufa-free-row-break input {
        flex: 1 1 220px;
        min-width: 220px;
        max-width: min(520px, 70%);
        border: 1px solid #cbd6e2;
        border-radius: 5px;
        background: #ffffff;
        color: #10233f;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0;
        padding: 3px 7px;
        text-transform: none;
      }
      .ufa-free-row-break:active {
        cursor: grabbing;
      }
      .ufa-free-row-break.drop-target {
        border-color: #2b8ce6;
        box-shadow: 0 0 0 3px rgba(43, 140, 230, 0.16);
      }
      .ufa-free-row-break button {
        border: 1px solid #cbd6e2;
        border-radius: 5px;
        background: #ffffff;
        color: #52637a;
        cursor: pointer;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 7px;
        touch-action: manipulation;
        text-transform: none;
        letter-spacing: 0;
        user-select: none;
        -webkit-user-drag: none;
      }
      .ufa-free-row-break .ufa-free-row-overlay {
        border-color: #2b8ce6;
        color: #1c6ea4;
      }
      .ufa-free-row-break .ufa-free-row-overlay:hover {
        background: #edf6ff;
      }
      .ufa-free-row-break .ufa-free-row-overlay-clear {
        border-color: #e0b9b0;
        color: #a23f2c;
      }
      .ufa-free-row-break .ufa-free-row-overlay-clear:hover {
        background: #fff3f0;
      }
      .ufa-free-row-break .ufa-free-row-move-up,
      .ufa-free-row-break .ufa-free-row-move-down,
      .ufa-free-row-break .ufa-free-row-move-top,
      .ufa-free-row-break .ufa-free-row-move-bottom {
        color: #52637a;
      }
      .ufa-free-row-break .ufa-free-row-move-up:hover,
      .ufa-free-row-break .ufa-free-row-move-down:hover,
      .ufa-free-row-break .ufa-free-row-move-top:hover,
      .ufa-free-row-break .ufa-free-row-move-bottom:hover {
        background: #f3f7fb;
      }
      .ufa-free-row-break .ufa-free-row-move-up,
      .ufa-free-row-break .ufa-free-row-move-down,
      .ufa-free-row-break .ufa-free-row-move-top,
      .ufa-free-row-break .ufa-free-row-move-bottom {
        width: 28px;
        min-width: 28px;
        height: 24px;
        box-sizing: border-box;
        padding: 2px 5px;
        line-height: 1;
      }
      .ufa-row-icon {
        position: relative;
        display: inline-block;
        width: 12px;
        height: 14px;
        color: currentColor;
        vertical-align: middle;
      }
      .ufa-row-icon::before,
      .ufa-row-icon::after {
        position: absolute;
        left: 2px;
        width: 7px;
        height: 7px;
        border-top: 2px solid currentColor;
        border-left: 2px solid currentColor;
        content: "";
      }
      .ufa-row-icon::before {
        top: 1px;
      }
      .ufa-row-icon::after {
        top: 7px;
      }
      .ufa-row-icon-up::before,
      .ufa-row-icon-up::after {
        transform: rotate(45deg);
      }
      .ufa-row-icon-down::before,
      .ufa-row-icon-down::after {
        transform: rotate(225deg);
      }
      .ufa-row-icon-single::after {
        display: none;
      }
      .ufa-row-icon-single::before {
        top: 3px;
      }
      .ufa-row-icon-down.ufa-row-icon-single::before {
        top: 0;
      }
      .ufa-free-card.overlay-selected {
        border-color: #c3482b;
        box-shadow: 0 0 0 3px rgba(195, 72, 43, 0.14);
      }
      .ufa-free-card-meta {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 3px;
        color: #40516a;
        font-size: 11px;
        line-height: 1.35;
        text-align: center;
      }
      .ufa-free-card-meta b {
        color: #10233f;
      }
      .ufa-free-card-meta span,
      .ufa-free-card-meta .shape {
        color: #667792;
      }
      .ufa-free-card-meta .shape {
        margin-top: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .ufa-free-overlay-toggle {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        justify-content: center;
        box-sizing: border-box;
        min-height: 26px;
        width: 100%;
        margin-top: 4px;
        padding: 4px 8px;
        border: 1px solid transparent;
        border-radius: 5px;
        color: #10233f;
        font-size: 11px;
        font-weight: 800;
        cursor: pointer;
        user-select: none;
      }
      .ufa-free-overlay-toggle:hover {
        border-color: #b9d9f5;
        background: #edf6ff;
      }
      .ufa-free-overlay-toggle input {
        margin: 0;
      }
      .ufa-free-card.row-selected {
        border-color: #2b8ce6;
        box-shadow: 0 0 0 3px rgba(43, 140, 230, 0.18);
      }
      .ufa-free-arrange-selection-count {
        color: #52637a;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 0;
      }
      .ufa-free-overlay-shell {
        box-sizing: border-box;
        flex: 0 0 270px;
        align-self: flex-start;
        position: sticky;
        top: 12px;
        width: 270px;
        height: fit-content;
        max-height: calc(100vh - 24px);
        overflow: visible;
      }
      .ufa-browser-overlay-widget {
        box-sizing: border-box;
        flex: 0 0 270px;
        align-self: flex-start;
        min-width: 270px;
        width: 270px;
        position: sticky !important;
        top: 12px;
        z-index: 1000;
        overflow: visible !important;
      }
      .ufa-browser-overlay-widget .widget-html,
      .ufa-browser-overlay-widget .widget-html-content {
        width: 100%;
        overflow: visible !important;
      }
      .ufa-free-overlay-panel {
        box-sizing: border-box;
        border: 1px solid #dfe7ef;
        border-radius: 8px;
        background: #f8fbff;
        padding: 10px;
        position: relative;
        width: 100%;
        max-height: calc(100vh - 24px);
        overflow-y: auto;
      }
      .ufa-overlay-svg {
        display: block;
        margin: 0 auto 10px;
        background: #f5f8f5;
        border-radius: 4px;
      }
      .ufa-free-overlay-note {
        color: #52637a;
        font-size: 11px;
        line-height: 1.35;
        margin: -6px 0 10px;
      }
      .ufa-overlay-path {
        opacity: 0.72;
      }
      .ufa-overlay-throw-line {
        stroke: var(--overlay-color);
        stroke-width: 2.4;
        stroke-linecap: round;
        fill: none;
      }
      .ufa-overlay-dot {
        fill: var(--overlay-color);
        stroke: #071019;
        stroke-width: 0.9;
      }
      .ufa-free-overlay-clear {
        border: 1px solid #cbd6e2;
        border-radius: 6px;
        background: #ffffff;
        color: #10233f;
        cursor: pointer;
        font-weight: 700;
        padding: 6px 10px;
        width: 100%;
      }
      @media (max-width: 800px) {
        .ufa-browser-sticky-toolbar {
          top: 4px;
          width: 100% !important;
          max-width: none;
        }
        .ufa-free-board-layout {
          flex-direction: column-reverse;
          padding-right: 0;
        }
        .ufa-free-overlay-shell {
          position: static !important;
          top: 12px;
          width: 100%;
          flex-basis: auto;
          max-height: none;
          overflow: visible;
        }
        .ufa-browser-overlay-widget {
          position: static !important;
          width: 100%;
          min-width: 0;
          flex-basis: auto;
          top: auto;
        }
        .ufa-free-overlay-panel {
          max-height: none;
          overflow: visible;
        }
      }
    </style>
    """


def write_possession_pattern_browser_html(
    possessions,
    paths,
    output_path,
    title="UFA possession pattern browser",
    max_cards=None,
    team_id=None,
    team_options=None,
    persistence_key=None,
):
    """Write a standalone HTML free-arrange possession browser."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path_lookup = _browser_path_lookup(paths)
    browser_possessions = _sort_browser_possessions(possessions)
    browser_possessions = browser_possessions[
        browser_possessions["possession_id"].isin(path_lookup)
    ].reset_index(drop=True)
    if browser_possessions.empty:
        raise ValueError("No possessions with matching paths were provided.")

    if max_cards is None:
        max_cards = len(browser_possessions)
    max_cards = max(1, min(int(max_cards), len(browser_possessions)))
    throw_counts = pd.to_numeric(
        browser_possessions.get("throw_count"), errors="coerce"
    ).dropna()
    min_throw_count = int(throw_counts.min()) if not throw_counts.empty else 1
    max_throw_count = int(throw_counts.max()) if not throw_counts.empty else 50
    throw_count_options_html = "".join(
        f'<label class="standalone-throw-count-option">'
        f'<input type="checkbox" value="{throw_count}" />'
        f'<span>{throw_count}</span></label>'
        for throw_count in range(min_throw_count, max_throw_count + 1)
    )

    board_html = render_possession_free_board(
        browser_possessions,
        path_lookup,
        max_cards=max_cards,
        include_overlay=True,
        persistence_key=persistence_key,
    )
    safe_title = escape(str(title))
    selected_team_id = str(team_id or "").strip().lower()
    team_options_html = ""
    for option in team_options or []:
        if isinstance(option, dict):
            option_id = option.get("id", "")
            option_label = option.get("label", option_id)
            option_href = option.get("href", "")
        else:
            option_id, option_label, option_href = option
        selected = " selected" if str(option_id).lower() == selected_team_id else ""
        team_options_html += (
            f'<option value="{escape(str(option_href), quote=True)}"{selected}>'
            f"{escape(str(option_label))}</option>"
        )
    team_filter_html = ""
    if team_options_html:
        team_filter_html = (
            '<label class="standalone-team-filter">Team'
            '<select id="standaloneTeamFilter" aria-label="Choose team">'
            f"{team_options_html}</select></label>"
        )
    standalone_undo_setup = """
              if (!board._ufaUndoStack) { board._ufaUndoStack = []; }
              function updateUndoButton() {
                const wrap = board.closest('.ufa-free-board-wrap');
                const button = wrap
                  ? wrap.querySelector('.ufa-free-undo-arrangement')
                  : document.querySelector('.ufa-free-undo-arrangement');
                if (!button) { return; }
                const canUndo = board._ufaUndoStack.length > 0;
                button.disabled = !canUndo;
                button.title = canUndo
                  ? 'Undo last arrangement change'
                  : 'Nothing to undo';
              }
              function snapshotBoard() {
                return {
                  children: Array.from(board.children).map(function (item) {
                    const titleInput = item.querySelector('.ufa-free-row-break input');
                    return {
                      node: item,
                      className: item.className,
                      hidden: item.hidden,
                      title: titleInput ? titleInput.value : null,
                      overlay: Array.from(
                        item.querySelectorAll('.ufa-free-overlay-check')
                      ).map(function (input) { return input.checked; }),
                    };
                  }),
                  selectionCommitted: board.dataset.selectionCommitted || 'false',
                };
              }
              board._ufaUpdateUndoButton = updateUndoButton;
              board._ufaPushUndo = function () {
                if (board._ufaUndoRestoring || board._ufaSuppressUndo) { return; }
                board._ufaUndoStack.push(snapshotBoard());
                if (board._ufaUndoStack.length > 50) {
                  board._ufaUndoStack.shift();
                }
                updateUndoButton();
              };
              board._ufaUndo = function () {
                if (!board._ufaUndoStack.length) {
                  updateUndoButton();
                  return;
                }
                const snapshot = board._ufaUndoStack.pop();
                board._ufaUndoRestoring = true;
                Array.from(board.children).forEach(function (item) { item.remove(); });
                snapshot.children.forEach(function (entry) {
                  entry.node.className = entry.className;
                  entry.node.hidden = entry.hidden;
                  const titleInput = entry.node.querySelector('.ufa-free-row-break input');
                  if (titleInput && entry.title !== null) {
                    titleInput.value = entry.title;
                  }
                  const inputs = Array.from(
                    entry.node.querySelectorAll('.ufa-free-overlay-check')
                  );
                  inputs.forEach(function (input, index) {
                    input.checked = Boolean(entry.overlay[index]);
                  });
                  board.appendChild(entry.node);
                });
                board.dataset.selectionCommitted = snapshot.selectionCommitted || 'false';
                board._ufaUndoRestoring = false;
                updateUndoButton();
                const wrap = board.closest('.ufa-free-board-wrap');
                const selectionCount = wrap && wrap.querySelector(
                  '.ufa-free-arrange-selection-count'
                );
                if (selectionCount) {
                  const selectedCount = board.querySelectorAll(
                    '.ufa-free-card.row-selected'
                  ).length;
                  selectionCount.textContent = selectedCount
                    ? selectedCount + ' selected'
                    : '0 selected';
                }
                const layer = wrap && wrap.querySelector('.ufa-overlay-layers');
                const count = wrap && wrap.querySelector(
                  '[id^="ufa-free-overlay-count-"]'
                );
                const selectedInputs = wrap
                  ? Array.from(wrap.querySelectorAll('.ufa-free-overlay-check:checked'))
                  : [];
                if (layer) {
                  layer.querySelectorAll('.ufa-overlay-path').forEach(function (path) {
                    path.style.display = 'none';
                  });
                  selectedInputs.forEach(function (input) {
                    const path = document.getElementById(input.dataset.overlayPath);
                    if (path) { path.style.display = ''; }
                  });
                }
                if (count) {
                  count.textContent = selectedInputs.length
                    ? selectedInputs.length + ' selected'
                    : 'Select cards to overlay';
                }
                board.querySelectorAll('.ufa-free-card.overlay-selected').forEach(
                  function (card) { card.classList.remove('overlay-selected'); }
                );
                selectedInputs.forEach(function (input) {
                  const card = input.closest('.ufa-free-card');
                  if (card) { card.classList.add('overlay-selected'); }
                });
              };
              updateUndoButton();
"""
    document = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"  <title>{safe_title}</title>\n"
        + """
        <script>
          (function () {
            try {
              const storedTheme = window.localStorage.getItem(
                'ufa-possession-browser-theme'
              );
              const prefersDark = window.matchMedia
                && window.matchMedia('(prefers-color-scheme: dark)').matches;
              document.documentElement.dataset.theme = storedTheme === 'dark'
                || (!storedTheme && prefersDark)
                ? 'dark'
                : 'light';
            } catch (error) {
              document.documentElement.dataset.theme = 'light';
            }
          })();
        </script>
        """
        + _possession_browser_css()
        + """
        <style>
          :root {
            color: #142a43;
            color-scheme: light;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #edf2f6;
          }
          * { box-sizing: border-box; }
          html { scroll-behavior: smooth; }
          body { margin: 0; min-width: 760px; background: #edf2f6; }
          [hidden] { display: none !important; }
          .standalone-page {
            width: 100%;
            max-width: 1840px;
            margin: 0 auto;
            padding: 0 14px 20px;
          }
          .standalone-heading {
            display: flex;
            align-items: center;
            min-height: 66px;
            margin: 0 -14px 14px;
            padding: 16px 24px;
            border-bottom: 1px solid #d2dce7;
            background: #edf2f6;
            box-shadow: none;
            color: #10233f;
            font-size: 24px;
            font-weight: 780;
            letter-spacing: 0;
          }
          .standalone-filters {
            display: flex;
            align-items: end;
            flex-wrap: wrap;
            gap: 10px 8px;
            margin-bottom: 14px;
            padding: 12px 14px;
            border: 1px solid #d2dce7;
            border-left: 4px solid #49a365;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 3px 10px rgba(15, 35, 58, 0.06);
          }
          .standalone-filters label {
            display: grid;
            gap: 4px;
            color: #40546e;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .standalone-filters select,
          .standalone-filters input,
          .standalone-filters button {
            min-height: 36px;
            border: 1px solid #c4d0dd;
            border-radius: 6px;
            background: #ffffff;
            color: #142a43;
            font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            padding: 7px 10px;
            transition: border-color 120ms ease, background-color 120ms ease, box-shadow 120ms ease;
          }
          .standalone-filters input { width: 84px; }
          .standalone-throw-count-picker {
            position: relative;
            align-self: end;
          }
          .standalone-throw-count-picker summary {
            display: inline-flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            min-width: 126px;
            min-height: 36px;
            padding: 7px 10px;
            border: 1px solid #c4d0dd;
            border-radius: 6px;
            background: #ffffff;
            color: #142a43;
            cursor: pointer;
            font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            list-style: none;
          }
          .standalone-throw-count-picker summary::-webkit-details-marker {
            display: none;
          }
          .standalone-throw-count-picker summary::after {
            content: "";
            width: 7px;
            height: 7px;
            border-right: 2px solid #52637a;
            border-bottom: 2px solid #52637a;
            transform: rotate(45deg) translate(-2px, -2px);
          }
          .standalone-throw-count-picker[open] summary::after {
            transform: rotate(-135deg) translate(-2px, -2px);
          }
          .standalone-throw-count-picker summary:hover {
            border-color: #879cb2;
          }
          .standalone-throw-count-picker summary:focus-visible {
            outline: 3px solid rgba(44, 125, 204, 0.22);
            outline-offset: 1px;
          }
          .standalone-throw-count-menu {
            position: absolute;
            top: calc(100% + 6px);
            left: 0;
            z-index: 220;
            width: 190px;
            max-height: 270px;
            overflow-y: auto;
            padding: 8px;
            border: 1px solid #c4d0dd;
            border-radius: 7px;
            background: #ffffff;
            box-shadow: 0 8px 20px rgba(15, 35, 58, 0.18);
          }
          .standalone-throw-count-options {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 2px 4px;
          }
          .standalone-throw-count-option {
            display: flex;
            align-items: center;
            gap: 5px;
            min-height: 28px;
            padding: 3px 4px;
            border-radius: 4px;
            color: #344b65;
            cursor: pointer;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: none;
          }
          .standalone-throw-count-option:hover {
            background: #f1f5f8;
          }
          .standalone-throw-count-option input {
            width: auto;
            min-height: 0;
            margin: 0;
            padding: 0;
            accent-color: #49a365;
          }
          .standalone-throw-count-clear {
            width: 100%;
            margin-top: 7px;
            padding: 6px 8px !important;
            min-height: 30px !important;
            background: #f5f7fa !important;
            color: #344b65 !important;
            font-size: 11px !important;
          }
          .standalone-filters .standalone-theme-toggle {
            position: relative;
            display: inline-flex;
            align-items: center;
            align-self: end;
            gap: 8px;
            min-height: 36px;
            padding: 7px 10px;
            border: 1px solid #c4d0dd;
            border-radius: 6px;
            background: #f5f7fa;
            cursor: pointer;
          }
          .standalone-theme-toggle input {
            position: absolute;
            width: 1px;
            height: 1px;
            margin: 0;
            opacity: 0;
            pointer-events: none;
          }
          .standalone-theme-track {
            position: relative;
            flex: 0 0 34px;
            width: 34px;
            height: 18px;
            border: 1px solid #9dacbc;
            border-radius: 9px;
            background: #cbd5df;
            transition: border-color 140ms ease, background-color 140ms ease;
          }
          .standalone-theme-track::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 35, 58, 0.28);
            transition: transform 140ms ease;
          }
          .standalone-theme-toggle input:checked + .standalone-theme-track {
            border-color: #3f8e58;
            background: #49a365;
          }
          .standalone-theme-toggle input:checked + .standalone-theme-track::after {
            transform: translateX(16px);
          }
          .standalone-theme-toggle input:focus-visible + .standalone-theme-track {
            outline: 3px solid rgba(44, 125, 204, 0.28);
            outline-offset: 2px;
          }
          .standalone-theme-text {
            color: #344b65;
            font-size: 12px;
            font-weight: 750;
            letter-spacing: 0;
            text-transform: none;
            white-space: nowrap;
          }
          .standalone-team-filter select {
            min-width: 200px;
            border-color: #83bd95;
            background: #f1f9f3;
            box-shadow: inset 3px 0 0 #49a365;
          }
          .standalone-filters select:hover,
          .standalone-filters input:hover,
          .standalone-filters button:hover {
            border-color: #879cb2;
          }
          .standalone-filters select:focus-visible,
          .standalone-filters input:focus-visible,
          .standalone-filters button:focus-visible,
          .standalone-browser .ufa-free-board-actions select:focus-visible,
          .standalone-browser .ufa-free-board-actions button:focus-visible {
            outline: 3px solid rgba(44, 125, 204, 0.22);
            outline-offset: 1px;
          }
          .standalone-filters button {
            cursor: pointer;
            align-self: end;
          }
          #standaloneResetFilters {
            background: #f5f7fa;
            color: #344b65;
          }
          #standaloneToggleOverlay {
            border-color: #83bd95;
            background: #eaf6ed;
            color: #1f6638;
          }
          .standalone-count {
            margin-left: auto;
            align-self: center;
            min-height: 36px;
            padding: 9px 11px;
            border-left: 3px solid #49a365;
            border-radius: 4px;
            background: #f4f7fa;
            color: #40546e;
            font-size: 12px;
            font-weight: 800;
          }
          .standalone-browser .ufa-free-board-wrap {
            border-color: #d2dce7;
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(15, 35, 58, 0.08);
          }
          .standalone-browser .ufa-free-board-content > .ufa-gallery-kicker {
            display: inline-block;
            margin: 0 10px 10px 0;
            color: #243d5a;
            font-size: 12px;
          }
          .standalone-browser .ufa-free-board-content > .ufa-free-board-meta {
            display: inline-block;
            margin: 0 0 10px;
            color: #697b91;
          }
          .standalone-browser .ufa-free-board-layout.with-inline-overlay {
            padding-right: 0;
          }
          .standalone-browser .ufa-free-board-actions {
            position: sticky;
            top: 8px;
            z-index: 100;
            margin: 0 0 12px;
            padding: 9px 11px;
            border: 1px solid #c4d0dd;
            border-left: 4px solid #49a365;
            border-radius: 7px;
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 6px 16px rgba(15, 35, 58, 0.13);
            gap: 6px;
          }
          .standalone-browser .ufa-free-board-actions label {
            color: #52637a;
          }
          .standalone-browser .ufa-free-board-actions select,
          .standalone-browser .ufa-free-board-actions button {
            min-height: 34px;
            border-color: #b9c7d5;
            background: #ffffff;
            color: #10233f;
          }
          .standalone-browser .ufa-free-board-actions button {
            padding-left: 9px;
            padding-right: 9px;
          }
          .standalone-browser .ufa-free-board-actions button:hover {
            border-color: #879cb2;
            background: #f3f6f9;
          }
          .standalone-browser .ufa-free-board-actions .ufa-free-recovery-menu summary {
            min-height: 34px;
            border-color: #b9c7d5;
            padding-left: 9px;
            padding-right: 9px;
          }
          .standalone-browser .ufa-free-board-actions .ufa-free-recovery-menu summary:hover {
            border-color: #879cb2;
            background: #f3f6f9;
          }
          .standalone-browser .ufa-free-board-actions button.primary {
            border-color: #2c7dcc;
            background: #2c7dcc;
            color: #ffffff;
          }
          .standalone-browser .ufa-free-board-actions button.primary:hover {
            border-color: #4191dd;
            background: #4191dd;
          }
          .standalone-browser .ufa-free-board-actions .ufa-free-sticky-overlay-toggle {
            border-color: #79b98c;
            background: #e7f5eb;
            color: #1f6638;
          }
          .standalone-browser .ufa-free-board-actions .ufa-free-clear-row-breaks {
            border-color: #d8aaa4;
            color: #8a3f37;
          }
          .standalone-browser .ufa-free-board-actions .ufa-free-arrange-selection-count {
            margin-left: auto;
            color: #52637a;
          }
          .standalone-browser .ufa-free-board-content {
            overflow: visible;
          }
          .standalone-browser .ufa-free-overlay-shell {
            top: 12px;
          }
          .standalone-browser.overlay-hidden .ufa-free-board-layout {
            padding-right: 0;
          }
          html[data-theme="dark"] {
            color: #e8f0ea;
            color-scheme: dark;
            background: #111713;
          }
          html[data-theme="dark"] body {
            background: #111713;
          }
          html[data-theme="dark"] .standalone-heading {
            border-color: #35443a;
            background: #151c17;
            color: #f0f5f1;
          }
          html[data-theme="dark"] .standalone-filters {
            border-color: #3b4d41;
            border-left-color: #56a76c;
            background: #18211b;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.24);
          }
          html[data-theme="dark"] .standalone-filters label {
            color: #b9c9be;
          }
          html[data-theme="dark"] .standalone-filters select,
          html[data-theme="dark"] .standalone-filters input,
          html[data-theme="dark"] .standalone-filters button,
          html[data-theme="dark"] .standalone-filters .standalone-theme-toggle {
            border-color: #4a5e50;
            background: #222d25;
            color: #edf4ef;
          }
          html[data-theme="dark"] .standalone-team-filter select {
            border-color: #5c9e70;
            background: #203127;
            box-shadow: inset 3px 0 0 #56a76c;
          }
          html[data-theme="dark"] .standalone-filters select:hover,
          html[data-theme="dark"] .standalone-filters input:hover,
          html[data-theme="dark"] .standalone-filters button:hover,
          html[data-theme="dark"] .standalone-filters .standalone-theme-toggle:hover {
            border-color: #74887a;
            background: #29372d;
          }
          html[data-theme="dark"] #standaloneResetFilters {
            background: #222d25;
            color: #dce7df;
          }
          html[data-theme="dark"] #standaloneToggleOverlay {
            border-color: #518e63;
            background: #203b29;
            color: #bfe6c9;
          }
          html[data-theme="dark"] .standalone-theme-text {
            color: #e3ece6;
          }
          html[data-theme="dark"] .standalone-theme-track {
            border-color: #607166;
            background: #3a4740;
          }
          html[data-theme="dark"] .standalone-count {
            border-left-color: #56a76c;
            background: #202a23;
            color: #c2d1c7;
          }
          html[data-theme="dark"] .standalone-throw-count-picker summary,
          html[data-theme="dark"] .standalone-throw-count-menu {
            border-color: #4a5e50;
            background: #222d25;
            color: #edf4ef;
          }
          html[data-theme="dark"] .standalone-throw-count-picker summary::after {
            border-color: #b7c7bc;
          }
          html[data-theme="dark"] .standalone-throw-count-option {
            color: #dce8df;
          }
          html[data-theme="dark"] .standalone-throw-count-option:hover {
            background: #2a382e;
          }
          html[data-theme="dark"] .standalone-throw-count-clear {
            border-color: #4a5e50 !important;
            background: #29372d !important;
            color: #dce8df !important;
          }
          html[data-theme="dark"] .ufa-free-board-wrap {
            border-color: #35453b;
            background: #151d18;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.26);
            color: #e8f0ea;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-content > .ufa-gallery-kicker {
            color: #dce8df;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-content > .ufa-free-board-meta,
          html[data-theme="dark"] .ufa-free-board-meta,
          html[data-theme="dark"] .ufa-gallery-kicker,
          html[data-theme="dark"] .ufa-free-overlay-note,
          html[data-theme="dark"] .ufa-free-arrange-selection-count,
          html[data-theme="dark"] .ufa-free-export-status {
            color: #aebfb3;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions {
            border-color: #43564a;
            border-left-color: #56a76c;
            background: rgba(24, 33, 27, 0.98);
            box-shadow: 0 7px 18px rgba(0, 0, 0, 0.30);
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions label {
            color: #bdcbc1;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions select,
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions button,
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions .ufa-free-recovery-menu summary {
            border-color: #4a5e50;
            background: #222d25;
            color: #edf4ef;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions button:hover,
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions .ufa-free-recovery-menu summary:hover {
            border-color: #74887a;
            background: #2a382e;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions button.primary {
            border-color: #3d83c4;
            background: #3577b5;
            color: #ffffff;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions button.primary:hover {
            border-color: #61a1db;
            background: #4388c8;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions .ufa-free-sticky-overlay-toggle {
            border-color: #518e63;
            background: #203b29;
            color: #bfe6c9;
          }
          html[data-theme="dark"] .standalone-browser .ufa-free-board-actions .ufa-free-clear-row-breaks {
            border-color: #815b55;
            color: #f0b8ae;
          }
          html[data-theme="dark"] .ufa-free-board-actions .ufa-free-recovery-menu summary::after {
            color: #b7c7bc;
          }
          html[data-theme="dark"] .ufa-free-board-actions .ufa-free-recovery-popover {
            border-color: #4a5e50;
            background: #202a23;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.42);
          }
          html[data-theme="dark"] .ufa-free-recovery-status {
            color: #9ed5ab;
          }
          html[data-theme="dark"] .ufa-free-recovery-status.save-error {
            color: #f0a99c;
          }
          html[data-theme="dark"] .ufa-free-card,
          html[data-theme="dark"] .ufa-free-overlay-panel {
            border-color: #3d4e43;
            background: #1d2721;
          }
          html[data-theme="dark"] .ufa-free-card-meta,
          html[data-theme="dark"] .ufa-free-card-meta span,
          html[data-theme="dark"] .ufa-free-card-meta .shape {
            color: #b2c2b7;
          }
          html[data-theme="dark"] .ufa-free-card-meta b,
          html[data-theme="dark"] .ufa-free-overlay-toggle {
            color: #eef4f0;
          }
          html[data-theme="dark"] .ufa-free-overlay-toggle:hover {
            border-color: #557b64;
            background: #26372c;
          }
          html[data-theme="dark"] .ufa-free-row-break {
            border-color: #607568;
            background: #1b261f;
            color: #bdccc2;
          }
          html[data-theme="dark"] .ufa-free-row-break input,
          html[data-theme="dark"] .ufa-free-row-break button,
          html[data-theme="dark"] .ufa-free-overlay-clear {
            border-color: #4a5e50;
            background: #263229;
            color: #e6eee8;
          }
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-overlay {
            border-color: #4a8dc8;
            color: #a9d2f2;
          }
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-overlay:hover {
            background: #203546;
          }
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-overlay-clear {
            border-color: #815b55;
            color: #f0b8ae;
          }
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-overlay-clear:hover {
            background: #3b2925;
          }
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-up,
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-down,
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-top,
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-bottom {
            color: #bdccc2;
          }
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-up:hover,
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-down:hover,
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-top:hover,
          html[data-theme="dark"] .ufa-free-row-break .ufa-free-row-move-bottom:hover {
            background: #29362d;
          }
          html[data-theme="dark"] .ufa-free-export-text {
            border-color: #4a5e50;
            background: #19221c;
            color: #dce8df;
          }
          html[data-theme="dark"] .ufa-multi-drag-preview {
            border-color: #4b94d5;
            background: #202a23;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.46);
          }
          html[data-theme="dark"] .ufa-multi-drag-preview.ufa-single-drag-preview {
            border-color: rgba(75, 148, 213, 0.82);
            background: transparent;
            box-shadow: none;
          }
          html[data-theme="dark"] .ufa-mini-svg,
          html[data-theme="dark"] .ufa-overlay-svg {
            background: #172019;
          }
          html[data-theme="dark"] .ufa-free-overlay-panel,
          html[data-theme="dark"] .ufa-free-recovery-popover {
            scrollbar-color: #52675a #1a231d;
          }
          html[data-theme="dark"] input[type="checkbox"] {
            accent-color: #4f9f66;
          }
          @media (min-width: 981px) and (max-width: 1499px) {
            .standalone-browser .ufa-free-placement-label {
              display: none;
            }
          }
          @media (min-width: 981px) and (max-width: 1399px) {
            .standalone-browser .ufa-free-board-actions .ufa-free-arrange-selection-count {
              display: none;
            }
          }
          @media (min-width: 981px) and (max-width: 1300px) {
            .standalone-browser .ufa-free-board-actions {
              gap: 4px;
            }
            .standalone-browser .ufa-free-board-actions button,
            .standalone-browser .ufa-free-board-actions .ufa-free-recovery-menu summary {
              padding-left: 7px;
              padding-right: 7px;
              font-size: 12px;
            }
          }
          @media (max-width: 980px) {
            body { min-width: 0; }
            .standalone-page { padding: 0 8px 14px; }
            .standalone-heading {
              min-height: 56px;
              margin: 0 -8px 10px;
              padding: 13px 16px;
              font-size: 20px;
            }
            .standalone-filters { padding: 10px; }
            .standalone-count { width: 100%; margin-left: 0; }
            .standalone-browser .ufa-free-board-actions { top: 4px; }
            .standalone-browser .ufa-free-board-actions .ufa-free-arrange-selection-count {
              width: 100%;
              margin-left: 0;
            }
          }
        </style>
        </head>
        <body>
          <div class="standalone-page">
        """
        + f"<h1 class=\"standalone-heading\">{safe_title}</h1>"
        + f"""
            <section class="standalone-filters" aria-label="Possession filters">
              {team_filter_html}
              <label>
                Line
                <select id="standaloneLineFilter">
                  <option value="all">All lines</option>
                  <option value="o_line">O-line</option>
                  <option value="d_line">D-line</option>
                </select>
              </label>
              <label>
                Outcome
                <select id="standaloneOutcomeFilter">
                  <option value="all">All outcomes</option>
                  <option value="goal">Goals</option>
                  <option value="turnover">Turnovers</option>
                </select>
              </label>
              <label>
                Hucks
                <select id="standaloneHuckFilter">
                  <option value="all">Include all</option>
                  <option value="non_huck">Non-hucks only</option>
                  <option value="huck">Hucks only</option>
                </select>
              </label>
              <label>
                Min throws
                <input id="standaloneMinThrows" type="number" min="0" value="{min_throw_count}" />
              </label>
              <label>
                Max throws
                <input id="standaloneMaxThrows" type="number" min="0" value="{max_throw_count}" />
              </label>
              <details id="standaloneThrowCountPicker" class="standalone-throw-count-picker">
                <summary id="standaloneThrowCountSummary">Exact throws</summary>
                <div class="standalone-throw-count-menu">
                  <div class="standalone-throw-count-options">
                    {throw_count_options_html}
                  </div>
                  <button id="standaloneClearThrowCounts" class="standalone-throw-count-clear" type="button">
                    Clear exact counts
                  </button>
                </div>
              </details>
              <label>
                Cards shown
                <input id="standaloneCardLimit" type="number" min="1" max="{max_cards}" value="{max_cards}" />
              </label>
              <button id="standaloneResetFilters" type="button">Reset filters</button>
              <button id="standaloneToggleOverlay" type="button">Hide overlay</button>
              <label class="standalone-theme-toggle" title="Use dark mode">
                <input id="standaloneThemeToggle" type="checkbox" role="switch"
                  aria-label="Dark mode" />
                <span class="standalone-theme-track" aria-hidden="true"></span>
                <span class="standalone-theme-text">Dark mode</span>
              </label>
              <span id="standaloneVisibleCount" class="standalone-count"></span>
            </section>
            <main id="standaloneBrowser" class="standalone-browser">
              {board_html}
            </main>
          </div>
          <script>
            (function () {{
              const browser = document.getElementById('standaloneBrowser');
              const board = browser.querySelector('.ufa-free-board');
              {standalone_undo_setup}
              const teamFilter = document.getElementById('standaloneTeamFilter');
              const lineFilter = document.getElementById('standaloneLineFilter');
              const outcomeFilter = document.getElementById('standaloneOutcomeFilter');
              const huckFilter = document.getElementById('standaloneHuckFilter');
              const minThrows = document.getElementById('standaloneMinThrows');
              const maxThrows = document.getElementById('standaloneMaxThrows');
              const throwCountSummary = document.getElementById(
                'standaloneThrowCountSummary'
              );
              const exactThrowCountInputs = Array.from(
                document.querySelectorAll(
                  '#standaloneThrowCountPicker .standalone-throw-count-option input'
                )
              );
              const clearExactThrowCounts = document.getElementById(
                'standaloneClearThrowCounts'
              );
              const cardLimit = document.getElementById('standaloneCardLimit');
              const count = document.getElementById('standaloneVisibleCount');
              const reset = document.getElementById('standaloneResetFilters');
              const overlayToggle = document.getElementById('standaloneToggleOverlay');
              const themeToggle = document.getElementById('standaloneThemeToggle');
              const overlayShell = browser.querySelector('.ufa-free-overlay-shell');
              const boardActions = browser.querySelector('.ufa-free-board-actions');
              const loadCheckpointButton = browser.querySelector('.ufa-free-load-arrangement');
              const recoveryHistory = browser.querySelector('.ufa-free-recovery-history');
              const recoveryRestore = browser.querySelector('.ufa-free-recovery-restore');
              const recoveryStatus = browser.querySelector('.ufa-free-recovery-status');
              const recoveryMenu = browser.querySelector('.ufa-free-recovery-menu');
              const recoveryOutput = browser.querySelector('.ufa-free-export-text');
              const checkpointStorageKey = board.dataset.checkpointStorageKey || '';
              const recoveryStorageKey = board.dataset.recoveryStorageKey || '';
              const stickyOverlayToggle = document.createElement('button');
              stickyOverlayToggle.id = 'standaloneStickyToggleOverlay';
              stickyOverlayToggle.className = 'ufa-free-sticky-overlay-toggle';
              stickyOverlayToggle.type = 'button';
              stickyOverlayToggle.textContent = 'Hide overlay';
              if (boardActions) {{
                const selectionCount = boardActions.querySelector(
                  '.ufa-free-arrange-selection-count'
                );
                boardActions.insertBefore(stickyOverlayToggle, selectionCount || null);
              }}

              if (teamFilter) {{
                teamFilter.addEventListener('change', function () {{
                  if (teamFilter.value) {{ window.location.assign(teamFilter.value); }}
                }});
              }}

              function applyTheme(theme, persist) {{
                const useDark = theme === 'dark';
                document.documentElement.dataset.theme = useDark ? 'dark' : 'light';
                if (themeToggle) {{
                  themeToggle.checked = useDark;
                  themeToggle.closest('.standalone-theme-toggle').title = useDark
                    ? 'Use light mode'
                    : 'Use dark mode';
                }}
                if (!persist) {{ return; }}
                try {{
                  window.localStorage.setItem(
                    'ufa-possession-browser-theme',
                    useDark ? 'dark' : 'light'
                  );
                }} catch (error) {{}}
              }}

              applyTheme(document.documentElement.dataset.theme, false);
              if (themeToggle) {{
                themeToggle.addEventListener('change', function () {{
                  applyTheme(themeToggle.checked ? 'dark' : 'light', true);
                }});
              }}

              if (recoveryMenu) {{
                document.addEventListener('click', function (event) {{
                  if (recoveryMenu.open && !recoveryMenu.contains(event.target)) {{
                    recoveryMenu.removeAttribute('open');
                  }}
                }});
              }}

              const dragScrollEdge = 140;
              const dragScrollMaxSpeed = 30;
              let dragScrollSpeed = 0;
              let dragScrollFrame = null;
              let customGroupDrag = null;
              let recoveryTimer = null;
              let recoveryApplying = false;
              let recoveryObserver = null;
              const recoveryLimit = 5;

              function stopDragScroll() {{
                dragScrollSpeed = 0;
                if (dragScrollFrame !== null) {{
                  window.cancelAnimationFrame(dragScrollFrame);
                  dragScrollFrame = null;
                }}
              }}

              function runDragScroll() {{
                const nativeDrag = board.querySelector('.ufa-free-board-item.dragging');
                const groupedDrag = customGroupDrag && customGroupDrag.active;
                if ((!nativeDrag && !groupedDrag) || dragScrollSpeed === 0) {{
                  stopDragScroll();
                  return;
                }}
                window.scrollBy(0, dragScrollSpeed);
                if (groupedDrag) {{
                  updateCustomGroupDropTarget(
                    customGroupDrag.clientX,
                    customGroupDrag.clientY
                  );
                }}
                dragScrollFrame = window.requestAnimationFrame(runDragScroll);
              }}

              function updateDragScroll(clientY) {{
                const viewportHeight = window.innerHeight
                  || document.documentElement.clientHeight;
                const edge = Math.min(dragScrollEdge, viewportHeight * 0.25);
                let nextSpeed = 0;
                if (clientY < edge) {{
                  const strength = Math.max(0, (edge - clientY) / edge);
                  nextSpeed = -Math.max(
                    4,
                    Math.round(dragScrollMaxSpeed * strength)
                  );
                }} else if (clientY > viewportHeight - edge) {{
                  const strength = Math.max(
                    0,
                    (clientY - (viewportHeight - edge)) / edge
                  );
                  nextSpeed = Math.max(
                    4,
                    Math.round(dragScrollMaxSpeed * strength)
                  );
                }}
                dragScrollSpeed = nextSpeed;
                if (dragScrollSpeed === 0) {{
                  stopDragScroll();
                }} else if (dragScrollFrame === null) {{
                  dragScrollFrame = window.requestAnimationFrame(runDragScroll);
                }}
              }}

              document.addEventListener('dragover', function (event) {{
                if (!board.querySelector('.ufa-free-board-item.dragging')) {{ return; }}
                event.preventDefault();
                updateDragScroll(event.clientY);
              }});
              document.addEventListener('dragend', stopDragScroll, true);
              document.addEventListener('drop', stopDragScroll, true);

              function buildCustomGroupPreview(state, event) {{
                const preview = document.createElement('div');
                preview.className = 'ufa-multi-drag-preview';
                const cardCount = state.cards.length;
                if (cardCount === 1) {{
                  preview.classList.add('ufa-single-drag-preview');
                }}
                const fieldWidth = 108;
                const fieldHeight = 230;
                const previewGap = 6;
                const previewPadding = 12;
                const availableWidth = Math.max(
                  fieldWidth,
                  window.innerWidth * 0.72
                );
                const availableHeight = Math.max(
                  fieldHeight,
                  window.innerHeight * 0.82
                );
                const maxFullColumns = Math.max(
                  1,
                  Math.floor(
                    (availableWidth - previewPadding + previewGap)
                    / (fieldWidth + previewGap)
                  )
                );
                const columns = Math.min(cardCount, 6, maxFullColumns);
                const rows = Math.ceil(cardCount / columns);
                const widthScale = (
                  availableWidth - previewPadding
                  - ((columns - 1) * previewGap)
                ) / (columns * fieldWidth);
                const heightScale = (
                  availableHeight - previewPadding
                  - ((rows - 1) * previewGap)
                ) / (rows * fieldHeight);
                const previewScale = Math.min(1, widthScale, heightScale);
                const thumbWidth = Math.max(
                  18,
                  Math.floor(fieldWidth * previewScale)
                );
                const thumbHeight = Math.max(
                  38,
                  Math.round(fieldHeight * previewScale)
                );
                preview.style.gridTemplateColumns = 'repeat('
                  + columns + ', ' + thumbWidth + 'px)';
                state.cards.forEach(function (card) {{
                  const field = card.querySelector('.ufa-mini-svg');
                  if (!field) {{ return; }}
                  const fieldClone = field.cloneNode(true);
                  fieldClone.setAttribute('width', thumbWidth);
                  fieldClone.setAttribute('height', thumbHeight);
                  fieldClone.style.margin = '0';
                  preview.appendChild(fieldClone);
                }});
                document.body.appendChild(preview);

                const draggedIndex = Math.max(0, state.cards.indexOf(state.source));
                const draggedColumn = draggedIndex % columns;
                const draggedRow = Math.floor(draggedIndex / columns);
                const sourceField = state.source.querySelector('.ufa-mini-svg');
                const sourceRect = sourceField
                  ? sourceField.getBoundingClientRect()
                  : state.source.getBoundingClientRect();
                const pointerXRatio = sourceRect.width
                  ? Math.max(
                      0,
                      Math.min(1, (state.startX - sourceRect.left) / sourceRect.width)
                    )
                  : 0.5;
                const pointerYRatio = sourceRect.height
                  ? Math.max(
                      0,
                      Math.min(1, (state.startY - sourceRect.top) / sourceRect.height)
                    )
                  : 0.5;
                const previewInset = 8;
                state.anchorX = previewInset
                  + (draggedColumn * (thumbWidth + previewGap))
                  + (pointerXRatio * thumbWidth);
                state.anchorY = previewInset
                  + (draggedRow * (thumbHeight + previewGap))
                  + (pointerYRatio * thumbHeight);
                state.preview = preview;
                positionCustomGroupPreview(state, event.clientX, event.clientY);
              }}

              function positionCustomGroupPreview(state, clientX, clientY) {{
                if (!state.preview) {{ return; }}
                state.preview.style.left = (clientX - state.anchorX) + 'px';
                state.preview.style.top = (clientY - state.anchorY) + 'px';
              }}

              function updateCustomGroupDropTarget(clientX, clientY) {{
                const state = customGroupDrag;
                if (!state || !state.active) {{ return; }}
                if (state.target) {{ state.target.classList.remove('drop-target'); }}
                state.target = null;
                const pointed = document.elementFromPoint(clientX, clientY);
                const target = pointed && pointed.closest('.ufa-free-board-item');
                if (!target || !board.contains(target) || state.cards.includes(target)) {{
                  return;
                }}
                target.classList.add('drop-target');
                state.target = target;
              }}

              function clearCustomGroupVisuals(state) {{
                state.source.classList.remove('dragging');
                state.cards.forEach(function (card) {{
                  card.classList.remove('group-dragging');
                }});
                if (state.target) {{ state.target.classList.remove('drop-target'); }}
                state.target = null;
                if (state.preview) {{ state.preview.remove(); }}
                state.preview = null;
                document.body.classList.remove('ufa-custom-group-drag-active');
                stopDragScroll();
              }}

              function suppressCustomGroupClick() {{
                board.dataset.suppressCardClick = 'true';
                window.setTimeout(function () {{
                  board.dataset.suppressCardClick = 'false';
                }}, 0);
              }}

              board.addEventListener('click', function (event) {{
                if (board.dataset.suppressCardClick !== 'true'
                    || !event.target.closest('.ufa-free-card')) {{
                  return;
                }}
                event.preventDefault();
                event.stopImmediatePropagation();
                board.dataset.suppressCardClick = 'false';
              }}, true);

              function removeCustomGroupListeners(state) {{
                document.removeEventListener('mousemove', state.onMove, true);
                document.removeEventListener('mouseup', state.onMouseUp, true);
                document.removeEventListener('mousedown', state.onMouseDown, true);
                document.removeEventListener('contextmenu', state.onContextMenu, true);
                document.removeEventListener('keydown', state.onKeyDown, true);
                window.removeEventListener('blur', state.onBlur, true);
              }}

              function finishCustomGroupDrag(state, suppressClick) {{
                clearCustomGroupVisuals(state);
                removeCustomGroupListeners(state);
                state.source.draggable = true;
                customGroupDrag = null;
                if (suppressClick) {{ suppressCustomGroupClick(); }}
              }}

              function restoreCustomGroupSnapshot(state) {{
                board.dataset.suppressRecoveryMutation = 'true';
                const fragment = document.createDocumentFragment();
                state.snapshot.forEach(function (item) {{
                  fragment.appendChild(item);
                }});
                board.appendChild(fragment);
              }}

              function cancelCustomGroupDrag(event) {{
                const state = customGroupDrag;
                if (!state || state.cancelled) {{ return; }}
                if (event) {{
                  event.preventDefault();
                  event.stopImmediatePropagation();
                }}
                state.cancelled = true;
                restoreCustomGroupSnapshot(state);
                clearCustomGroupVisuals(state);
              }}

              function beginCustomGroupDrag(event) {{
                if (event.button !== 0) {{ return; }}
                const source = event.target.closest('.ufa-free-card');
                if (!source || !board.contains(source)
                    || event.target.closest('button, input, select, textarea, label')) {{
                  return;
                }}
                const selectedCards = Array.from(
                  board.querySelectorAll('.ufa-free-card.row-selected')
                );
                const draggedCards = source.classList.contains('row-selected')
                    && selectedCards.length > 1
                  ? selectedCards
                  : [source];

                source.draggable = false;
                const state = {{
                  source: source,
                  cards: draggedCards,
                  snapshot: Array.from(board.children),
                  startX: event.clientX,
                  startY: event.clientY,
                  clientX: event.clientX,
                  clientY: event.clientY,
                  active: false,
                  cancelled: false,
                  preview: null,
                  target: null,
                  anchorX: 0,
                  anchorY: 0,
                }};
                customGroupDrag = state;

                state.onMove = function (moveEvent) {{
                  if (state.cancelled) {{ return; }}
                  if ((moveEvent.buttons & 2) === 2) {{
                    cancelCustomGroupDrag(moveEvent);
                    return;
                  }}
                  state.clientX = moveEvent.clientX;
                  state.clientY = moveEvent.clientY;
                  if (!state.active) {{
                    const distance = Math.hypot(
                      moveEvent.clientX - state.startX,
                      moveEvent.clientY - state.startY
                    );
                    if (distance < 5) {{ return; }}
                    state.active = true;
                    state.source.classList.add('dragging');
                    if (state.cards.length > 1) {{
                      state.cards.forEach(function (card) {{
                        card.classList.add('group-dragging');
                      }});
                    }}
                    document.body.classList.add('ufa-custom-group-drag-active');
                    buildCustomGroupPreview(state, moveEvent);
                  }}
                  moveEvent.preventDefault();
                  positionCustomGroupPreview(
                    state,
                    moveEvent.clientX,
                    moveEvent.clientY
                  );
                  updateCustomGroupDropTarget(
                    moveEvent.clientX,
                    moveEvent.clientY
                  );
                  updateDragScroll(moveEvent.clientY);
                }};

                state.onMouseDown = function (downEvent) {{
                  if (downEvent.button === 2) {{ cancelCustomGroupDrag(downEvent); }}
                }};
                state.onContextMenu = function (contextEvent) {{
                  contextEvent.preventDefault();
                  contextEvent.stopImmediatePropagation();
                  cancelCustomGroupDrag(contextEvent);
                }};
                state.onKeyDown = function (keyEvent) {{
                  if (keyEvent.key === 'Escape') {{ cancelCustomGroupDrag(keyEvent); }}
                }};
                state.onBlur = function () {{
                  cancelCustomGroupDrag();
                  finishCustomGroupDrag(state, true);
                }};
                state.onMouseUp = function (upEvent) {{
                  if (upEvent.button !== 0) {{ return; }}
                  if (state.cancelled) {{
                    upEvent.preventDefault();
                    finishCustomGroupDrag(state, true);
                    return;
                  }}
                  if (!state.active) {{
                    finishCustomGroupDrag(state, false);
                    return;
                  }}
                  upEvent.preventDefault();
                  upEvent.stopImmediatePropagation();
                  const target = state.target;
                  if (target && !state.cards.includes(target)) {{
                    board._ufaPushUndo();
                    const items = Array.from(
                      board.querySelectorAll('.ufa-free-board-item')
                    );
                    const fromIndex = items.indexOf(state.source);
                    const toIndex = items.indexOf(target);
                    const fragment = document.createDocumentFragment();
                    state.cards.forEach(function (card) {{
                      fragment.appendChild(card);
                    }});
                    if (fromIndex < toIndex) {{ target.after(fragment); }}
                    else {{ target.before(fragment); }}
                  }} else {{
                    restoreCustomGroupSnapshot(state);
                  }}
                  finishCustomGroupDrag(state, true);
                }};

                document.addEventListener('mousemove', state.onMove, true);
                document.addEventListener('mouseup', state.onMouseUp, true);
                document.addEventListener('mousedown', state.onMouseDown, true);
                document.addEventListener('contextmenu', state.onContextMenu, true);
                document.addEventListener('keydown', state.onKeyDown, true);
                window.addEventListener('blur', state.onBlur, true);
              }}

              board.addEventListener('mousedown', beginCustomGroupDrag, true);

              function numericValue(input, fallback) {{
                const value = Number(input.value);
                return Number.isFinite(value) ? value : fallback;
              }}

              function selectedExactThrowCounts() {{
                return exactThrowCountInputs
                  .filter(function (input) {{ return input.checked; }})
                  .map(function (input) {{ return Number(input.value); }})
                  .filter(function (value) {{ return Number.isFinite(value); }});
              }}

              function updateExactThrowSummary() {{
                if (!throwCountSummary) {{ return; }}
                const selected = selectedExactThrowCounts();
                throwCountSummary.textContent = selected.length
                  ? 'Exact throws (' + selected.length + ')'
                  : 'Exact throws';
                throwCountSummary.title = selected.length
                  ? 'Only counts ' + selected.join(', ') + ' are included'
                  : 'Use the min/max range';
              }}

              updateExactThrowSummary();

              function readRecoverySnapshots() {{
                if (!recoveryStorageKey) {{ return []; }}
                try {{
                  const text = window.localStorage.getItem(recoveryStorageKey);
                  if (!text) {{ return []; }}
                  const parsed = JSON.parse(text);
                  const snapshots = Array.isArray(parsed)
                    ? parsed
                    : Array.isArray(parsed.snapshots)
                    ? parsed.snapshots
                    : [];
                  return snapshots
                    .filter(function (snapshot) {{
                      return snapshot && snapshot.payload && snapshot.saved_at;
                    }})
                    .slice(0, recoveryLimit);
                }} catch (error) {{
                  return [];
                }}
              }}

              function writeRecoverySnapshots(snapshots) {{
                if (!recoveryStorageKey) {{ return false; }}
                try {{
                  window.localStorage.setItem(
                    recoveryStorageKey,
                    JSON.stringify({{ version: 1, snapshots: snapshots.slice(0, recoveryLimit) }})
                  );
                  return true;
                }} catch (error) {{
                  if (recoveryStatus) {{
                    recoveryStatus.textContent = 'Save error';
                    recoveryStatus.title = 'Autosave unavailable - recovery disabled';
                    recoveryStatus.classList.add('save-error');
                  }}
                  return false;
                }}
              }}

              function recoveryTimestamp(value) {{
                const timestamp = new Date(value);
                if (Number.isNaN(timestamp.getTime())) {{ return 'Unknown time'; }}
                return timestamp.toLocaleString([], {{
                  month: 'short',
                  day: 'numeric',
                  hour: 'numeric',
                  minute: '2-digit',
                  second: '2-digit',
                }});
              }}

              function refreshRecoveryHistory(preferredId) {{
                if (!recoveryHistory || !recoveryRestore) {{ return; }}
                const snapshots = readRecoverySnapshots();
                const previousValue = preferredId || recoveryHistory.value;
                recoveryHistory.innerHTML = '';
                if (!snapshots.length) {{
                  const emptyOption = document.createElement('option');
                  emptyOption.value = '';
                  emptyOption.textContent = 'No recovery snapshots';
                  recoveryHistory.appendChild(emptyOption);
                  recoveryHistory.disabled = true;
                  recoveryRestore.disabled = true;
                  return;
                }}
                snapshots.forEach(function (snapshot, index) {{
                  const option = document.createElement('option');
                  option.value = snapshot.id || snapshot.saved_at;
                  option.textContent = (index === 0 ? 'Latest - ' : '')
                    + recoveryTimestamp(snapshot.saved_at);
                  recoveryHistory.appendChild(option);
                }});
                const hasPreviousValue = snapshots.some(function (snapshot) {{
                  return (snapshot.id || snapshot.saved_at) === previousValue;
                }});
                recoveryHistory.value = hasPreviousValue
                  ? previousValue
                  : (snapshots[0].id || snapshots[0].saved_at);
                recoveryHistory.disabled = false;
                recoveryRestore.disabled = false;
              }}

              function currentRecoveryUiState() {{
                return {{
                  line: lineFilter.value,
                  outcome: outcomeFilter.value,
                  hucks: huckFilter.value,
                  min_throws: minThrows.value,
                  max_throws: maxThrows.value,
                  throw_counts: selectedExactThrowCounts(),
                  cards_shown: cardLimit.value,
                  overlay_hidden: Boolean(overlayShell && overlayShell.hidden),
                }};
              }}

              function serializeRecoveryArrangement() {{
                const groups = [];
                let currentGroup = {{
                  title: 'Group 1',
                  possessions: [],
                  has_break: false,
                  auto_unsorted: false,
                }};
                Array.from(board.children).forEach(function (item) {{
                  if (item.classList.contains('ufa-free-row-break')) {{
                    if (currentGroup.has_break || currentGroup.possessions.length) {{
                      groups.push(currentGroup);
                    }}
                    const titleInput = item.querySelector('input');
                    const title = titleInput && titleInput.value.trim()
                      ? titleInput.value.trim()
                      : item.dataset.arrangeLabel;
                    currentGroup = {{
                      title: title || 'Pattern group ' + (groups.length + 2),
                      possessions: [],
                      has_break: true,
                      auto_unsorted: item.dataset.autoUnsorted === 'true',
                    }};
                    return;
                  }}
                  if (!item.classList.contains('ufa-free-card')) {{ return; }}
                  const checkbox = item.querySelector('.ufa-free-overlay-check');
                  currentGroup.possessions.push({{
                    possession_id: item.dataset.possessionId,
                    overlay_selected: Boolean(checkbox && checkbox.checked),
                  }});
                }});
                if (currentGroup.has_break || currentGroup.possessions.length) {{
                  groups.push(currentGroup);
                }}
                const savedAt = new Date().toISOString();
                return {{
                  version: 2,
                  saved_at: savedAt,
                  cards_saved: groups.reduce(function (total, group) {{
                    return total + group.possessions.length;
                  }}, 0),
                  groups: groups.map(function (group, index) {{
                    return {{
                      group_index: index + 1,
                      title: group.title,
                      auto_unsorted: Boolean(group.auto_unsorted),
                      possessions: group.possessions,
                    }};
                  }}),
                  ui_state: currentRecoveryUiState(),
                }};
              }}

              function recoverySignature(payload) {{
                return JSON.stringify({{
                  groups: payload.groups || [],
                  ui_state: payload.ui_state || {{}},
                }});
              }}

              function saveRecoverySnapshot() {{
                if (recoveryTimer !== null) {{
                  window.clearTimeout(recoveryTimer);
                  recoveryTimer = null;
                }}
                if (recoveryApplying || !recoveryStorageKey) {{ return; }}
                const payload = serializeRecoveryArrangement();
                const signature = recoverySignature(payload);
                const snapshots = readRecoverySnapshots();
                if (snapshots.length && snapshots[0].signature === signature) {{
                  if (recoveryStatus) {{
                    recoveryStatus.textContent = 'Saved';
                    recoveryStatus.title = 'Autosaved '
                      + recoveryTimestamp(snapshots[0].saved_at);
                    recoveryStatus.classList.remove('save-error');
                  }}
                  refreshRecoveryHistory(snapshots[0].id || snapshots[0].saved_at);
                  return;
                }}
                const snapshot = {{
                  id: payload.saved_at + '-' + Math.random().toString(36).slice(2, 8),
                  saved_at: payload.saved_at,
                  signature: signature,
                  payload: payload,
                }};
                snapshots.unshift(snapshot);
                if (!writeRecoverySnapshots(snapshots)) {{ return; }}
                if (recoveryStatus) {{
                  recoveryStatus.textContent = 'Saved';
                  recoveryStatus.title = 'Autosaved '
                    + recoveryTimestamp(snapshot.saved_at);
                  recoveryStatus.classList.remove('save-error');
                }}
                refreshRecoveryHistory(snapshot.id);
              }}

              function scheduleRecoverySave() {{
                if (recoveryApplying || !recoveryStorageKey) {{ return; }}
                if (recoveryTimer !== null) {{ window.clearTimeout(recoveryTimer); }}
                if (recoveryStatus) {{
                  recoveryStatus.textContent = 'Saving...';
                  recoveryStatus.title = 'Autosave pending';
                  recoveryStatus.classList.remove('save-error');
                }}
                recoveryTimer = window.setTimeout(saveRecoverySnapshot, 1000);
              }}

              function setControlValue(control, value) {{
                if (!control || value === undefined || value === null) {{ return; }}
                const nextValue = String(value);
                if (control.tagName === 'SELECT') {{
                  const valid = Array.from(control.options).some(function (option) {{
                    return option.value === nextValue;
                  }});
                  if (!valid) {{ return; }}
                }}
                control.value = nextValue;
              }}

              function applyRecoveryUiState(uiState) {{
                if (!uiState) {{ return; }}
                setControlValue(lineFilter, uiState.line);
                setControlValue(outcomeFilter, uiState.outcome);
                setControlValue(huckFilter, uiState.hucks);
                setControlValue(minThrows, uiState.min_throws);
                setControlValue(maxThrows, uiState.max_throws);
                if (Array.isArray(uiState.throw_counts)) {{
                  const selectedCounts = new Set(
                    uiState.throw_counts.map(function (value) {{ return String(value); }})
                  );
                  exactThrowCountInputs.forEach(function (input) {{
                    input.checked = selectedCounts.has(input.value);
                  }});
                  updateExactThrowSummary();
                }}
                setControlValue(cardLimit, uiState.cards_shown);
                if (typeof uiState.overlay_hidden === 'boolean') {{
                  setOverlayHidden(uiState.overlay_hidden);
                }}
              }}

              function applyRecoverySnapshot(snapshot, automatic) {{
                if (!snapshot || !snapshot.payload || !checkpointStorageKey
                    || !loadCheckpointButton) {{
                  return false;
                }}
                let checkpointText = null;
                let hadCheckpoint = false;
                recoveryApplying = true;
                if (automatic) {{ board._ufaSuppressUndo = true; }}
                try {{
                  checkpointText = window.localStorage.getItem(checkpointStorageKey);
                  hadCheckpoint = checkpointText !== null;
                  window.localStorage.setItem(
                    checkpointStorageKey,
                    JSON.stringify(snapshot.payload)
                  );
                  loadCheckpointButton.click();
                  }} catch (error) {{
                  if (recoveryStatus) {{
                    recoveryStatus.textContent = 'Restore error';
                    recoveryStatus.title = 'Recovery snapshot could not be restored';
                    recoveryStatus.classList.add('save-error');
                  }}
                  recoveryApplying = false;
                  return false;
                }} finally {{
                  try {{
                    if (hadCheckpoint) {{
                      window.localStorage.setItem(checkpointStorageKey, checkpointText);
                    }} else {{
                      window.localStorage.removeItem(checkpointStorageKey);
                    }}
                  }} catch (error) {{}}
                  if (automatic) {{ board._ufaSuppressUndo = false; }}
                }}
                if (recoveryOutput) {{
                  recoveryOutput.textContent = '';
                  recoveryOutput.style.display = 'none';
                }}
                applyRecoveryUiState(snapshot.payload.ui_state);
                previousThrowRange = null;
                if (!automatic) {{ applyFilters(); }}
                if (recoveryStatus) {{
                  recoveryStatus.textContent = automatic ? 'Recovered' : 'Restored';
                  recoveryStatus.title = (automatic ? 'Recovered ' : 'Restored ')
                    + recoveryTimestamp(snapshot.saved_at);
                  recoveryStatus.classList.remove('save-error');
                }}
                refreshRecoveryHistory(snapshot.id || snapshot.saved_at);
                window.setTimeout(function () {{
                  recoveryApplying = false;
                  if (!automatic) {{ scheduleRecoverySave(); }}
                }}, 0);
                return true;
              }}

              function startRecovery() {{
                if (!recoveryStorageKey || !recoveryHistory || !recoveryRestore) {{
                  return;
                }}
                const snapshots = readRecoverySnapshots();
                refreshRecoveryHistory();
                if (snapshots.length) {{
                  applyRecoverySnapshot(snapshots[0], true);
                }} else if (recoveryStatus) {{
                  recoveryStatus.textContent = 'Saved';
                  recoveryStatus.title = 'Autosave is ready';
                }}

                recoveryObserver = new MutationObserver(function () {{
                  if (board.dataset.suppressRecoveryMutation === 'true') {{
                    board.dataset.suppressRecoveryMutation = 'false';
                    return;
                  }}
                  scheduleRecoverySave();
                }});
                recoveryObserver.observe(board, {{ childList: true }});

                board.addEventListener('input', function (event) {{
                  if (event.target.matches('.ufa-free-row-break input')) {{
                    scheduleRecoverySave();
                  }}
                }});
                board.addEventListener('change', function (event) {{
                  if (event.target.matches('.ufa-free-overlay-check')) {{
                    scheduleRecoverySave();
                  }}
                }});
                browser.addEventListener('click', function (event) {{
                  if (event.target.closest(
                    '.ufa-free-row-overlay, .ufa-free-row-overlay-clear, .ufa-free-overlay-clear'
                  )) {{
                    window.setTimeout(scheduleRecoverySave, 0);
                  }}
                }});
                [lineFilter, outcomeFilter, huckFilter, minThrows, maxThrows, cardLimit]
                  .concat(exactThrowCountInputs)
                  .forEach(function (control) {{
                    control.addEventListener('change', scheduleRecoverySave);
                    control.addEventListener('input', scheduleRecoverySave);
                  }});
                recoveryRestore.addEventListener('click', function () {{
                  const selectedId = recoveryHistory.value;
                  const selectedSnapshot = readRecoverySnapshots().find(
                    function (snapshot) {{
                      return (snapshot.id || snapshot.saved_at) === selectedId;
                    }}
                  );
                  applyRecoverySnapshot(selectedSnapshot, false);
                  if (recoveryMenu) {{ recoveryMenu.removeAttribute('open'); }}
                }});
                document.addEventListener('visibilitychange', function () {{
                  if (document.visibilityState === 'hidden' && recoveryTimer !== null) {{
                    saveRecoverySnapshot();
                  }}
                }});
                window.addEventListener('beforeunload', function () {{
                  if (recoveryTimer !== null) {{ saveRecoverySnapshot(); }}
                }});

                window.setTimeout(function () {{
                  if (!snapshots.length) {{ return; }}
                  const currentSignature = recoverySignature(serializeRecoveryArrangement());
                  if (currentSignature !== snapshots[0].signature) {{
                    scheduleRecoverySave();
                  }}
                }}, 0);
              }}

              function refreshVisibleOverlay() {{
                const overlayInput = board.querySelector('.ufa-free-overlay-check');
                if (overlayInput) {{
                  overlayInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
              }}

              function syncRowBreakVisibility() {{
                board.querySelectorAll('.ufa-free-row-break').forEach(function (rowBreak) {{
                  rowBreak.hidden = false;
                }});
              }}

              function unsortedRowTitle(minimum, maximum, exactCounts) {{
                if (exactCounts && exactCounts.length) {{
                  return 'Unsorted: ' + exactCounts.join(', ') + ' throws';
                }}
                if (minimum === maximum) {{
                  return 'Unsorted: ' + minimum + ' throw' + (minimum === 1 ? '' : 's');
                }}
                return 'Unsorted: ' + minimum + '-' + maximum + ' throws';
              }}

              function updateRowSelectionCount() {{
                const selectionCount = boardActions && boardActions.querySelector(
                  '.ufa-free-arrange-selection-count'
                );
                if (!selectionCount) {{ return; }}
                const selectedCount = board.querySelectorAll(
                  '.ufa-free-card.row-selected'
                ).length;
                selectionCount.textContent = selectedCount
                  ? selectedCount + ' selected'
                  : '0 selected';
              }}

              function createFallbackUnsortedRow(title) {{
                const rowBreak = document.createElement('div');
                rowBreak.id = 'ufa-auto-unsorted-' + Date.now();
                rowBreak.className = 'ufa-free-board-item ufa-free-row-break';
                rowBreak.dataset.autoUnsorted = 'true';
                rowBreak.dataset.arrangeLabel = title;

                const titleInput = document.createElement('input');
                titleInput.type = 'text';
                titleInput.value = title;
                titleInput.setAttribute('aria-label', 'Row title');
                titleInput.addEventListener('click', function (event) {{
                  event.stopPropagation();
                }});

                const removeButton = document.createElement('button');
                removeButton.type = 'button';
                removeButton.className = 'ufa-free-row-remove';
                removeButton.textContent = 'Remove';
                removeButton.setAttribute('aria-label', 'Remove row break');
                removeButton.addEventListener('click', function (event) {{
                  event.stopPropagation();
                  board._ufaPushUndo();
                  rowBreak.remove();
                }});

                rowBreak.appendChild(titleInput);
                rowBreak.appendChild(removeButton);
                board.appendChild(rowBreak);
                return rowBreak;
              }}

              function moveNewThrowCardsToBottom(
                cards,
                minimum,
                maximum,
                exactCounts
              ) {{
                const newCards = cards.filter(function (card) {{
                  return card && card.isConnected
                    && card.classList.contains('ufa-free-card');
                }});
                if (!newCards.length) {{ return; }}

                // Always create a new staging row. Reusing an older automatic
                // row would append newly included cards to the user's last
                // organized row and change its contents unexpectedly.
                board.dataset.suppressFilterMutation = 'true';
                const stagingTitle = unsortedRowTitle(
                  minimum,
                  maximum,
                  exactCounts
                );
                createFallbackUnsortedRow(stagingTitle);
                newCards.forEach(function (card) {{ board.appendChild(card); }});
              }}

              let previousThrowRange = null;

              function applyFilters() {{
                if (board.dataset.suppressFilterRun === 'true') {{
                  board.dataset.suppressFilterRun = 'false';
                  return;
                }}
                const selectedLine = lineFilter.value;
                const selectedOutcome = outcomeFilter.value;
                const selectedHuck = huckFilter.value;
                const minimum = numericValue(minThrows, 0);
                const maximum = numericValue(maxThrows, Number.POSITIVE_INFINITY);
                const selectedExactCounts = selectedExactThrowCounts();
                const selectedExactCountSet = new Set(selectedExactCounts);
                const limit = Math.max(1, numericValue(cardLimit, Number.POSITIVE_INFINITY));
                const throwInputsComplete = minThrows.value.trim() !== ''
                  && maxThrows.value.trim() !== '';
                const previousExactCounts = previousThrowRange
                  && Array.isArray(previousThrowRange.exact_counts)
                  ? previousThrowRange.exact_counts
                  : [];
                const exactCountsExpanded = previousThrowRange !== null
                  && (
                    (selectedExactCounts.length === 0 && previousExactCounts.length > 0)
                    || selectedExactCounts.some(function (value) {{
                      return !previousExactCounts.includes(value);
                    }})
                  );
                const expandedThrowRange = throwInputsComplete
                  && previousThrowRange !== null
                  && (
                    minimum < previousThrowRange.minimum
                    || maximum > previousThrowRange.maximum
                    || exactCountsExpanded
                  );
                const newThrowCards = [];
                let visible = 0;
                let matched = 0;

                board.querySelectorAll('.ufa-free-card').forEach(function (card) {{
                  const throws = Number(card.dataset.throwCount || 0);
                  const hucks = Number(card.dataset.huckCount || 0);
                  const lineMatches = selectedLine === 'all' || card.dataset.lineType === selectedLine;
                  const outcomeMatches = selectedOutcome === 'all' || card.dataset.outcome === selectedOutcome;
                  const huckMatches = selectedHuck === 'all'
                    || (selectedHuck === 'non_huck' && hucks === 0)
                    || (selectedHuck === 'huck' && hucks > 0);
                  const throwMatches = throws >= minimum && throws <= maximum
                    && (
                      selectedExactCountSet.size === 0
                      || selectedExactCountSet.has(throws)
                    );
                  const matchedPreviousThrowRange = previousThrowRange !== null
                    && throws >= previousThrowRange.minimum
                    && throws <= previousThrowRange.maximum
                    && (
                      previousExactCounts.length === 0
                      || previousExactCounts.includes(throws)
                    );
                  if (expandedThrowRange && throwMatches && !matchedPreviousThrowRange
                      && lineMatches && outcomeMatches && huckMatches) {{
                    newThrowCards.push(card);
                  }}
                  const matches = lineMatches && outcomeMatches && huckMatches && throwMatches;
                  if (matches) {{ matched += 1; }}
                  const show = matches && visible < limit;
                  card.hidden = !show;
                  if (show) {{ visible += 1; }}
                  if (!show) {{ card.classList.remove('row-selected'); }}
                }});

                if (expandedThrowRange) {{
                  moveNewThrowCardsToBottom(
                    newThrowCards,
                    minimum,
                    maximum,
                    selectedExactCounts
                  );
                }}
                if (throwInputsComplete) {{
                  previousThrowRange = {{
                    minimum: minimum,
                    maximum: maximum,
                    exact_counts: selectedExactCounts,
                  }};
                }}
                updateRowSelectionCount();
                syncRowBreakVisibility();
                refreshVisibleOverlay();
                count.textContent = visible.toLocaleString() + ' shown of '
                  + matched.toLocaleString() + ' matching possessions';
              }}

              [lineFilter, outcomeFilter, huckFilter, minThrows, maxThrows, cardLimit]
                .concat(exactThrowCountInputs)
                .forEach(function (control) {{
                  control.addEventListener('change', function () {{
                    updateExactThrowSummary();
                    applyFilters();
                  }});
                  control.addEventListener('input', function () {{
                    updateExactThrowSummary();
                    applyFilters();
                  }});
                }});

              if (clearExactThrowCounts) {{
                clearExactThrowCounts.addEventListener('click', function () {{
                  exactThrowCountInputs.forEach(function (input) {{
                    input.checked = false;
                  }});
                  updateExactThrowSummary();
                  applyFilters();
                  scheduleRecoverySave();
                }});
              }}

              reset.addEventListener('click', function () {{
                lineFilter.value = 'all';
                outcomeFilter.value = 'all';
                huckFilter.value = 'all';
                minThrows.value = '{min_throw_count}';
                maxThrows.value = '{max_throw_count}';
                exactThrowCountInputs.forEach(function (input) {{
                  input.checked = false;
                }});
                updateExactThrowSummary();
                cardLimit.value = '{max_cards}';
                previousThrowRange = null;
                applyFilters();
                scheduleRecoverySave();
              }});

              function setOverlayHidden(hidden) {{
                overlayShell.hidden = hidden;
                browser.classList.toggle('overlay-hidden', hidden);
                const buttonText = hidden ? 'Show overlay' : 'Hide overlay';
                overlayToggle.textContent = buttonText;
                stickyOverlayToggle.textContent = buttonText;
                overlayToggle.setAttribute('aria-pressed', hidden ? 'true' : 'false');
                stickyOverlayToggle.setAttribute('aria-pressed', hidden ? 'true' : 'false');
              }}

              overlayToggle.addEventListener('click', function () {{
                setOverlayHidden(!overlayShell.hidden);
                scheduleRecoverySave();
              }});
              stickyOverlayToggle.addEventListener('click', function () {{
                setOverlayHidden(!overlayShell.hidden);
                scheduleRecoverySave();
              }});

              let filterQueued = false;
              new MutationObserver(function () {{
                if (board.dataset.suppressFilterMutation === 'true') {{
                  board.dataset.suppressFilterMutation = 'false';
                  return;
                }}
                if (filterQueued) {{ return; }}
                filterQueued = true;
                window.requestAnimationFrame(function () {{
                  filterQueued = false;
                  applyFilters();
                }});
              }}).observe(board, {{ childList: true }});

              startRecovery();
              applyFilters();
            }})();
          </script>
        </body>
        </html>
        """
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def create_scoring_possession_browser(
    possessions,
    paths,
    title="Possessions",
    n_shape_clusters=6,
):
    """Create an ipywidgets browser for Shown Space-style possession SVGs."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError(
            "ipywidgets is required for create_scoring_possession_browser. "
            "Install it with `pip install ipywidgets` and restart the notebook kernel."
        ) from exc

    if possessions.empty:
        return widgets.HTML("<b>No possessions available.</b>")

    lookup = _browser_path_lookup(paths)
    base_possessions = _sort_browser_possessions(possessions)
    base_possessions = base_possessions[
        base_possessions["possession_id"].isin(lookup)
    ].reset_index(drop=True)
    if base_possessions.empty:
        return widgets.HTML("<b>No matching possession paths available.</b>")

    if "line_type" not in base_possessions:
        base_possessions["line_type"] = base_possessions["possession_id"].map(
            lambda possession_id: _possession_line_type(lookup[possession_id])
        )
    if "path_cluster" not in base_possessions:
        try:
            base_possessions = cluster_scoring_possessions(
                base_possessions,
                paths,
                n_clusters=n_shape_clusters,
            )
            base_possessions = _sort_browser_possessions(base_possessions)
        except Exception:
            base_possessions = add_possession_style_labels(base_possessions)

    if "style" not in base_possessions:
        base_possessions = add_possession_style_labels(base_possessions)
    base_possessions = _add_browser_shape_cluster_labels(base_possessions)

    outcome_counts = (
        base_possessions.get("outcome", pd.Series("unknown", index=base_possessions.index))
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .value_counts()
    )
    total_count = len(base_possessions)
    goal_count = int(outcome_counts.get("goal", 0))
    turnover_count = int(outcome_counts.get("turnover", 0))
    header = widgets.HTML(
        f"""
        <div class="ufa-browser-title">
          <div class="kicker">Possession Browser</div>
          <h2>{escape(title)}</h2>
          <div class="subtitle">
            {total_count:,} possessions analyzed
            &middot; {goal_count:,} goals
            &middot; {turnover_count:,} turnovers
          </div>
        </div>
        """
    )
    view_filter = widgets.Dropdown(
        options=[
            ("Single possession", "single"),
            ("Shape gallery", "shape_gallery"),
            ("Free arrange board", "free_board"),
        ],
        value="single",
        description="View",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    line_filter = widgets.Dropdown(
        options=[
            ("All lines", "all"),
            ("O-line possessions", "o_line"),
            ("D-line possessions", "d_line"),
        ],
        value="all",
        description="Line",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    outcome_options = [("All outcomes", "all")]
    if "outcome" in base_possessions:
        outcome_counts = (
            base_possessions["outcome"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .value_counts()
        )
        for outcome_name in ["goal", "turnover", "unknown"]:
            if outcome_name in outcome_counts:
                outcome_options.append(
                    (
                        f"{outcome_name.title()} ({int(outcome_counts[outcome_name])})",
                        outcome_name,
                    )
                )
    outcome_filter = widgets.Dropdown(
        options=outcome_options,
        value="all",
        description="Outcome",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )

    def make_shape_options(frame):
        shape_options = [("All shapes", "all")]
        if "path_cluster" not in frame or frame.empty:
            return shape_options
        shape_counts = (
            frame
            .groupby(["path_cluster", "shape_cluster_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "path_cluster"], ascending=[False, True])
        )
        shape_options.extend(
            (
                f"{row['shape_cluster_label']} ({int(row['count'])})",
                int(row["path_cluster"]),
            )
            for _, row in shape_counts.iterrows()
        )
        return shape_options

    shape_options = make_shape_options(base_possessions)
    shape_filter = widgets.Dropdown(
        options=shape_options,
        value="all",
        description="Shape",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    min_throws = int(pd.to_numeric(base_possessions["throw_count"], errors="coerce").min())
    max_throws = int(pd.to_numeric(base_possessions["throw_count"], errors="coerce").max())
    throw_count_filter = widgets.IntRangeSlider(
        value=[min_throws, max_throws],
        min=min_throws,
        max=max_throws,
        step=1,
        description="Throws",
        continuous_update=False,
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    order_filter = widgets.Dropdown(
        options=[
            ("Game order", "game_order"),
            ("aEC / throw, high to low", "aec_per_throw_desc"),
            ("aEC / throw, low to high", "aec_per_throw_asc"),
        ],
        value="game_order",
        description="Order",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    free_board_card_filter = widgets.IntSlider(
        value=min(24, len(base_possessions)),
        min=1,
        max=max(1, len(base_possessions)),
        step=1,
        description="Cards shown",
        continuous_update=False,
        layout=widgets.Layout(width="430px", display="none"),
        style={"description_width": "85px"},
    )
    dropdown = widgets.Dropdown(
        options=[],
        value=None,
        description="Possession",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    previous_button = widgets.Button(
        description="Previous",
        icon="chevron-left",
        layout=widgets.Layout(width="112px"),
    )
    next_button = widgets.Button(
        description="Next",
        icon="chevron-right",
        layout=widgets.Layout(width="112px"),
    )
    menu_toggle_button = widgets.Button(
        description="Hide menu",
        icon="columns",
        layout=widgets.Layout(width="126px"),
        tooltip="Hide or show the browser controls",
    )
    count_label = widgets.HTML()
    count_label.add_class("ufa-browser-count")
    summary_html = widgets.HTML()
    shape_overview_html = widgets.HTML()
    field_html = widgets.HTML(
        layout=widgets.Layout(width="100%", overflow="visible")
    )
    overlay_html = widgets.HTML(
        layout=widgets.Layout(
            width="270px",
            min_width="270px",
            flex="0 0 270px",
            overflow="visible",
            display="none",
        )
    )
    overlay_html.add_class("ufa-browser-overlay-widget")
    overlay_toggle_button = widgets.Button(
        description="Hide overlay",
        icon="eye-slash",
        layout=widgets.Layout(width="145px", display="none"),
        tooltip="Hide or show the selected-possession overlay",
    )
    overlay_hidden = {"value": False}
    state = {"possessions": base_possessions}
    sticky_toolbar_bootstrap = (
        "var widget=this.closest('.widget-html');"
        "if(widget){"
        "widget.style.position='sticky';"
        "widget.style.top='8px';"
        "widget.style.zIndex='2000';"
        "widget.style.overflow='visible';"
        "}"
    )
    external_toolbar_html = f"""
    <div class="ufa-free-external-actions" aria-label="Free arrange controls"
      onmouseenter="{escape(sticky_toolbar_bootstrap, quote=True)}"
      onfocusin="{escape(sticky_toolbar_bootstrap, quote=True)}">
      <label>
        Placement
        <select class="ufa-free-external-position" aria-label="Placement"
          onchange="(function(value){{var boards=document.querySelectorAll('.ufa-free-board-wrap');var board=boards.length?boards[boards.length-1]:null;var select=board&&board.querySelector('.ufa-free-insert-position');if(select){{select.value=value;}}}})(this.value)">
          <option value="above">Above selected</option>
          <option value="below">Below selected</option>
          <option value="bottom">At bottom</option>
        </select>
      </label>
      <button type="button"
        onclick="(function(){{var boards=document.querySelectorAll('.ufa-free-board-wrap');var board=boards.length?boards[boards.length-1]:null;var position=document.querySelector('.ufa-free-external-position');var select=board&&board.querySelector('.ufa-free-insert-position');var button=board&&board.querySelector('.ufa-free-insert-row-break');if(select&&position){{select.value=position.value;}}if(button){{button.click();}}}})()">
        Insert row break
      </button>
      <button type="button"
        onclick="(function(){{var boards=document.querySelectorAll('.ufa-free-board-wrap');var board=boards.length?boards[boards.length-1]:null;var button=board&&board.querySelector('.ufa-free-clear-selection');if(button){{button.click();}}}})()">
        Clear selection
      </button>
      <button type="button" class="ufa-free-external-undo-arrangement"
        aria-label="Undo last arrangement change" title="Undo last arrangement change"
        onclick="(function(){{var boards=document.querySelectorAll('.ufa-free-board-wrap');var board=boards.length?boards[boards.length-1]:null;var button=board&&board.querySelector('.ufa-free-undo-arrangement');if(button&&!button.disabled){{button.click();}}}})()">
        Undo
      </button>
      <button class="primary ufa-free-external-save-arrangement" type="button"
        onclick="(function(){{var boards=document.querySelectorAll('.ufa-free-board-wrap');var board=boards.length?boards[boards.length-1]:null;var button=board&&board.querySelector('.ufa-free-save-arrangement');if(button){{button.click();}}}})()">
        Save arrangement
      </button>
      <button class="ufa-free-external-load-arrangement" type="button"
        onclick="(function(){{var boards=document.querySelectorAll('.ufa-free-board-wrap');var board=boards.length?boards[boards.length-1]:null;var button=board&&board.querySelector('.ufa-free-load-arrangement');if(button){{button.click();}}}})()">
        Load saved
      </button>
      <button type="button"
        onclick="(function(){{var boards=document.querySelectorAll('.ufa-free-board-wrap');var board=boards.length?boards[boards.length-1]:null;var button=board&&board.querySelector('.ufa-free-clear-row-breaks');if(button){{button.click();}}}})()">
        Clear row breaks
      </button>
    </div>
    """
    free_arrange_toolbar = widgets.HTML(
        value=external_toolbar_html,
        layout=widgets.Layout(width="100%", display="none", overflow="visible"),
    )
    free_arrange_toolbar.add_class("ufa-browser-sticky-toolbar")

    def set_free_arrange_toolbar_visibility():
        free_arrange_toolbar.layout.display = (
            None if view_filter.value == "free_board" else "none"
        )

    def set_overlay_visibility(hidden):
        overlay_hidden["value"] = bool(hidden)
        overlay_html.layout.display = "none" if hidden else None
        if hidden:
            overlay_toggle_button.description = "Show overlay"
            overlay_toggle_button.icon = "eye"
        else:
            overlay_toggle_button.description = "Hide overlay"
            overlay_toggle_button.icon = "eye-slash"

    def make_options(frame):
        options = []
        for index, row in frame.iterrows():
            aec_per_throw = _format_browser_number(row.get("aec_per_throw"), digits=3)
            label = (
                f"{index + 1}. {_line_type_label(row.get('line_type'))} | "
                f"{str(row.get('outcome', 'unknown')).title()} | "
                f"{_shape_cluster_label(row)} | "
                f"{row['GameID']} | Q{row['game_quarter']} "
                f"P{row['quarter_point']} | poss {row['possession_num']} | "
                f"{int(row['throw_count'])} throws | aEC/T {aec_per_throw}"
            )
            options.append((label, index))
        return options

    def order_possessions(frame):
        if frame.empty:
            return frame

        order_value = order_filter.value
        if order_value == "game_order":
            return _sort_browser_possessions(frame).reset_index(drop=True)

        ordered = frame.copy()
        ordered["_aec_per_throw_sort"] = pd.to_numeric(
            ordered.get("aec_per_throw"),
            errors="coerce",
        )
        ascending = order_value == "aec_per_throw_asc"
        tie_breakers = [
            column
            for column in [
                "GameID",
                "game_quarter",
                "quarter_point",
                "possession_num",
            ]
            if column in ordered
        ]
        sort_columns = ["_aec_per_throw_sort", *tie_breakers]
        ascending_values = [ascending, *([True] * len(tie_breakers))]
        return (
            ordered.sort_values(
                sort_columns,
                ascending=ascending_values,
                na_position="last",
            )
            .drop(columns=["_aec_per_throw_sort"])
            .reset_index(drop=True)
        )

    def update(index):
        browser_possessions = state["possessions"]
        set_free_arrange_toolbar_visibility()
        if index is None or browser_possessions.empty:
            count_label.value = "<b>0</b> of <b>0</b>"
            summary_html.value = "<b>No possessions match these filters.</b>"
            field_html.value = ""
            overlay_html.value = ""
            overlay_html.layout.display = "none"
            overlay_toggle_button.layout.display = "none"
            return

        if view_filter.value == "shape_gallery":
            count_label.value = f"<b>{len(browser_possessions)}</b> filtered"
            summary_html.value = ""
            summary_panel.layout.display = "none"
            overlay_html.value = ""
            overlay_html.layout.display = "none"
            overlay_toggle_button.layout.display = "none"
            field_panel.add_class("gallery")
            gallery_max_shapes = 1 if shape_filter.value != "all" else 6
            gallery_max_per_shape = 12 if shape_filter.value != "all" else 4
            field_html.value = render_possession_shape_gallery(
                browser_possessions,
                lookup,
                max_shapes=gallery_max_shapes,
                max_per_shape=gallery_max_per_shape,
            )
            return

        if view_filter.value == "free_board":
            count_label.value = f"<b>{len(browser_possessions)}</b> filtered"
            summary_html.value = ""
            summary_panel.layout.display = "none"
            overlay_toggle_button.layout.display = None
            overlay_html.layout.display = (
                "none" if overlay_hidden["value"] else None
            )
            field_panel.add_class("gallery")
            field_html.value, overlay_html.value = render_possession_free_board(
                browser_possessions,
                lookup,
                max_cards=int(free_board_card_filter.value),
                include_overlay=False,
                return_overlay=True,
                external_controls=True,
            )
            return

        summary_panel.layout.display = None
        overlay_html.value = ""
        overlay_html.layout.display = "none"
        overlay_toggle_button.layout.display = "none"
        field_panel.remove_class("gallery")

        row = browser_possessions.iloc[index]
        path = lookup[row["possession_id"]]
        count_label.value = f"<b>{index + 1}</b> of <b>{len(browser_possessions)}</b>"
        summary_html.value = render_possession_browser_summary(row, path)
        field_html.value = render_shownspace_possession_svg(path)

    def apply_filters(_=None):
        set_free_arrange_toolbar_visibility()
        free_board_card_filter.layout.display = (
            None if view_filter.value == "free_board" else "none"
        )
        selected_line = line_filter.value
        selected_outcome = outcome_filter.value
        min_throw_count, max_throw_count = throw_count_filter.value
        if selected_line == "all":
            filtered = base_possessions.copy()
        else:
            filtered = base_possessions[
                base_possessions["line_type"].eq(selected_line)
            ].reset_index(drop=True)

        if selected_outcome != "all" and "outcome" in filtered:
            filtered = filtered[
                filtered["outcome"]
                .fillna("unknown")
                .astype(str)
                .str.lower()
                .eq(selected_outcome)
            ].reset_index(drop=True)

        throw_counts = pd.to_numeric(filtered["throw_count"], errors="coerce")
        filtered = filtered[
            throw_counts.ge(min_throw_count) & throw_counts.le(max_throw_count)
        ].reset_index(drop=True)

        current_shape = shape_filter.value
        shape_options = make_shape_options(filtered)
        shape_values = [value for _, value in shape_options]
        if current_shape not in shape_values:
            current_shape = "all"
        if tuple(shape_filter.options) != tuple(shape_options):
            shape_filter.options = shape_options
        if shape_filter.value != current_shape:
            shape_filter.value = current_shape
        selected_shape = shape_filter.value

        shape_overview_html.value = render_shape_cluster_overview(
            filtered,
            selected_shape=selected_shape,
        )
        if selected_shape != "all" and "path_cluster" in filtered:
            filtered = filtered[
                filtered["path_cluster"].eq(selected_shape)
            ].reset_index(drop=True)

        filtered = order_possessions(filtered)

        board_card_limit = max(1, len(filtered))
        if free_board_card_filter.max != board_card_limit:
            free_board_card_filter.max = board_card_limit
        if free_board_card_filter.value > board_card_limit:
            free_board_card_filter.value = board_card_limit

        state["possessions"] = filtered
        if filtered.empty:
            dropdown.options = [("No possessions for this filter", None)]
            dropdown.value = None
            update(None)
            return

        dropdown.options = make_options(filtered)
        dropdown.value = 0
        update(0)

    def on_dropdown_change(change):
        if change["name"] == "value" and change["new"] is not None:
            update(change["new"])

    def on_previous(_):
        if dropdown.value is None:
            return
        dropdown.value = max(0, dropdown.value - 1)

    def on_next(_):
        if dropdown.value is None:
            return
        browser_possessions = state["possessions"]
        dropdown.value = min(len(browser_possessions) - 1, dropdown.value + 1)

    dropdown.observe(on_dropdown_change, names="value")
    line_filter.observe(apply_filters, names="value")
    outcome_filter.observe(apply_filters, names="value")
    shape_filter.observe(apply_filters, names="value")
    throw_count_filter.observe(apply_filters, names="value")
    order_filter.observe(apply_filters, names="value")
    free_board_card_filter.observe(apply_filters, names="value")
    view_filter.observe(apply_filters, names="value")
    previous_button.on_click(on_previous)
    next_button.on_click(on_next)

    def on_overlay_toggle(_):
        set_overlay_visibility(not overlay_hidden["value"])

    overlay_toggle_button.on_click(on_overlay_toggle)

    nav = widgets.HBox([previous_button, next_button, count_label])
    nav.add_class("ufa-browser-nav")
    controls = widgets.VBox(
        [
            header,
            view_filter,
            line_filter,
            outcome_filter,
            shape_filter,
            shape_overview_html,
            throw_count_filter,
            order_filter,
            free_board_card_filter,
            dropdown,
            nav,
        ],
        layout=widgets.Layout(width="455px"),
    )
    controls.add_class("ufa-browser-controls")
    field_panel = widgets.Box(
        [field_html],
        layout=widgets.Layout(
            min_width="560px",
            width="100%",
            flex="1 1 auto",
            overflow="visible",
        ),
    )
    field_panel.add_class("ufa-browser-field-panel")
    summary_panel = widgets.Box([summary_html])
    summary_panel.add_class("ufa-browser-info-panel")
    main_panel = widgets.HBox(
        [field_panel, overlay_html, summary_panel],
        layout=widgets.Layout(
            align_items="flex-start",
            gap="16px",
            width="100%",
            flex="1 1 auto",
            min_width="0",
            overflow="visible",
        ),
    )
    main_panel.add_class("ufa-browser-main-panel")
    shell = widgets.HBox(
        [controls, main_panel],
        layout=widgets.Layout(
            align_items="flex-start",
            gap="18px",
            width="100%",
            overflow="visible",
        ),
    )
    shell.add_class("ufa-possession-browser-shell")

    def on_menu_toggle(_):
        menu_is_hidden = controls.layout.display == "none"
        if menu_is_hidden:
            controls.layout.display = None
            menu_toggle_button.description = "Hide menu"
            menu_toggle_button.icon = "columns"
        else:
            controls.layout.display = "none"
            menu_toggle_button.description = "Show menu"
            menu_toggle_button.icon = "bars"

    menu_toggle_button.on_click(on_menu_toggle)
    topbar = widgets.HBox(
        [menu_toggle_button, overlay_toggle_button],
        layout=widgets.Layout(gap="8px"),
    )
    topbar.add_class("ufa-browser-topbar")
    apply_filters()
    browser_root = widgets.VBox(
        [
            widgets.HTML(_possession_browser_css()),
            topbar,
            free_arrange_toolbar,
            shell,
        ],
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    browser_root.add_class("ufa-possession-browser-root")
    return browser_root


def create_team_scoring_possession_browser(
    season=2026,
    default_team_id="glory",
    final_only=True,
    max_games=None,
    regular_season_only=True,
    outcomes=("goal", "turnover"),
    pull_receive_scores_only=False,
    long_field_only=False,
    max_start_y=45,
    min_field_progress=50,
    exclude_hucks=False,
    n_shape_clusters=8,
    delay=0.15,
):
    """Create a notebook browser with a team selector and cached team data."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "ipywidgets and IPython are required for create_team_scoring_possession_browser."
        ) from exc

    status = widgets.HTML("Loading game list...")
    output = widgets.Output()

    games = fetch_shownspace_games(
        season=season,
        final_only=final_only,
        regular_season_only=regular_season_only,
    )
    if games.empty:
        return widgets.VBox([widgets.HTML("<b>No Shown Space games found.</b>")])

    team_ids = sorted(
        set(games["HomeTeamID"].dropna().str.lower())
        | set(games["AwayTeamID"].dropna().str.lower())
    )
    if default_team_id.lower() not in team_ids and team_ids:
        default_team_id = team_ids[0]

    team_dropdown = widgets.Dropdown(
        options=[(team_id.title(), team_id) for team_id in team_ids],
        value=default_team_id.lower(),
        description="Team",
        layout=widgets.Layout(width="320px"),
        style={"description_width": "70px"},
    )
    load_button = widgets.Button(
        description="Load team",
        icon="search",
        button_style="primary",
        layout=widgets.Layout(width="110px"),
    )
    cache = {}

    def _team_games(team_id):
        selected = games[
            games["HomeTeamID"].str.lower().eq(team_id)
            | games["AwayTeamID"].str.lower().eq(team_id)
        ].sort_values("StartTimestamp")
        if max_games is not None:
            selected = selected.head(max_games)
        return selected.reset_index(drop=True)

    def _filtered_team_data(team_id):
        if team_id in cache:
            return cache[team_id]

        selected_games = _team_games(team_id)
        if selected_games.empty:
            possessions = pd.DataFrame()
            paths = []
        else:
            throws = fetch_shownspace_throws_for_games(
                selected_games["GameID"].tolist(),
                delay=delay,
            )
            possessions, paths = build_possessions(
                throws,
                team_id=team_id,
                outcomes=outcomes,
            )

        if not possessions.empty and pull_receive_scores_only:
            possessions = possessions[possessions["possession_num"].eq(1)].copy()
        if not possessions.empty and long_field_only:
            possessions = possessions[
                possessions["start_y"].le(max_start_y)
                & possessions["field_progress"].ge(min_field_progress)
            ].copy()
        if not possessions.empty and exclude_hucks:
            possessions = possessions[possessions["huck_count"].fillna(0).eq(0)].copy()

        if not possessions.empty:
            possession_ids = set(possessions["possession_id"])
            paths = [
                path
                for path in paths
                if not path.empty and path["possession_id"].iloc[0] in possession_ids
            ]

        cache[team_id] = (selected_games, possessions.reset_index(drop=True), paths)
        return cache[team_id]

    def _render_team(team_id):
        load_button.disabled = True
        status.value = f"Loading {escape(team_id.title())}..."
        with output:
            output.clear_output(wait=True)
        try:
            selected_games, possessions, paths = _filtered_team_data(team_id)
            if set(outcomes or []) == {"goal"}:
                title = f"{team_id.title()} scoring possessions, {season}"
                possession_word = "scoring possessions"
            else:
                title = f"{team_id.title()} offensive possessions, {season}"
                possession_word = "possessions"
            browser = create_scoring_possession_browser(
                possessions,
                paths,
                title=title,
                n_shape_clusters=n_shape_clusters,
            )
            status.value = (
                f"<b>{escape(team_id.title())}</b>: "
                f"{len(possessions):,} {possession_word} from "
                f"{len(selected_games):,} games"
            )
            with output:
                output.clear_output(wait=True)
                display(browser)
        finally:
            load_button.disabled = False

    def _on_load(_):
        _render_team(team_dropdown.value)

    load_button.on_click(_on_load)
    controls = widgets.HBox([team_dropdown, load_button])
    controls.add_class("ufa-team-browser-controls")
    status.add_class("ufa-browser-status")
    _render_team(team_dropdown.value)
    return widgets.VBox([widgets.HTML(_possession_browser_css()), controls, status, output])


def _add_path_arrows(fig, points, color, every=1, opacity=0.85):
    for index, (start, end) in enumerate(zip(points.iloc[:-1].itertuples(), points.iloc[1:].itertuples())):
        if index % every != 0:
            continue
        annotation = {
            "x": end.x,
            "y": end.y,
            "ax": start.x,
            "ay": start.y,
            "showarrow": True,
            "arrowhead": 3,
            "arrowsize": 1,
            "arrowwidth": 1.8,
            "arrowcolor": color,
            "opacity": opacity,
        }
        fig.add_annotation(
            **annotation,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
        )


def _add_path_arrows_to_subplot(fig, points, color, row, col, every=1, opacity=0.85):
    for index, (start, end) in enumerate(zip(points.iloc[:-1].itertuples(), points.iloc[1:].itertuples())):
        if index % every != 0:
            continue
        fig.add_annotation(
            x=end.x,
            y=end.y,
            ax=start.x,
            ay=start.y,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.8,
            arrowcolor=color,
            opacity=opacity,
            row=row,
            col=col,
        )


def plot_possession_path(path, title="Scoring possession path", color="#b74126"):
    """Plot one real scoring possession, preserving its actual zig-zag shape."""
    import plotly.graph_objects as go

    points = _path_points(path)
    fig = go.Figure()
    _add_field_shapes(fig)
    fig.add_trace(
        go.Scatter(
            x=points["x"],
            y=points["y"],
            mode="lines+markers",
            line={"color": color, "width": 4},
            marker={"size": 8, "color": color},
            text=["Start"] + _path_hover_text(path),
            hovertemplate="%{text}<extra></extra>",
            name="Real possession",
        )
    )
    _add_path_arrows(fig, points, color)
    _apply_field_layout(fig, title)
    return fig


def plot_representative_paths(
    representative_paths,
    title="Representative scoring path styles",
):
    """Overlay one real representative possession for each style or cluster."""
    import plotly.graph_objects as go

    colors = ["#b74126", "#164e87", "#7a3db8", "#2f7d32", "#d97706", "#0f766e"]
    fig = go.Figure()
    _add_field_shapes(fig)

    items = representative_paths.items()
    for index, (label, path) in enumerate(items):
        color = colors[index % len(colors)]
        points = _path_points(path)
        fig.add_trace(
            go.Scatter(
                x=points["x"],
                y=points["y"],
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 7, "color": color},
                text=["Start"] + _path_hover_text(path),
                hovertemplate=f"{label}<br>%{{text}}<extra></extra>",
                name=label,
            )
        )
        _add_path_arrows(fig, points, color, every=2, opacity=0.65)

    _apply_field_layout(fig, title, width=680)
    return fig


def _style_color(label, index):
    label_lower = str(label).lower()
    if "huck" in label_lower:
        return "#b74126"
    if "reset" in label_lower:
        return "#164e87"
    if "switch" in label_lower:
        return "#7a3db8"
    if "circuitous" in label_lower:
        return "#2f7d32"
    if "low-progress" in label_lower:
        return "#6b7280"
    if "balanced" in label_lower:
        return "#d97706"
    colors = ["#b74126", "#164e87", "#7a3db8", "#2f7d32", "#d97706", "#0f766e"]
    return colors[index % len(colors)]


def plot_team_representative_path_grid(
    team_representative_paths,
    title="Representative scoring path styles by team",
    style_filter=None,
    show_arrows=False,
):
    """Plot representative scoring paths for multiple teams side by side."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    teams = list(team_representative_paths.keys())
    fig = make_subplots(
        rows=1,
        cols=len(teams),
        subplot_titles=[team.title() for team in teams],
        horizontal_spacing=0.04,
    )

    for col, team in enumerate(teams, start=1):
        _add_field_shapes(fig, row=1, col=col)
        representatives = team_representative_paths[team]
        if style_filter is not None:
            representatives = {
                label: path
                for label, path in representatives.items()
                if style_filter.lower() in str(label).lower()
            }
        for index, (label, path) in enumerate(representatives.items()):
            color = _style_color(label, index)
            points = _path_points(path)
            fig.add_trace(
                go.Scatter(
                    x=points["x"],
                    y=points["y"],
                    mode="lines+markers",
                    line={"color": color, "width": 2.5},
                    marker={"size": 6, "color": color},
                    text=["Start"] + _path_hover_text(path),
                    hovertemplate=(
                        f"{team.title()}<br>{label}<br>%{{text}}<extra></extra>"
                    ),
                    name=f"{team.title()} - {label}",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            if show_arrows:
                _add_path_arrows_to_subplot(
                    fig,
                    points,
                    color,
                    row=1,
                    col=col,
                    every=3,
                    opacity=0.45,
                )

        if not representatives:
            fig.add_annotation(
                x=0,
                y=60,
                text=f"No {style_filter} path",
                showarrow=False,
                font={"size": 12, "color": "#526173"},
                row=1,
                col=col,
            )

        fig.update_xaxes(
            range=[FIELD_X_MIN - 5, FIELD_X_MAX + 5],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor=f"y{col if col > 1 else ''}",
            scaleratio=1,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            range=[FIELD_Y_MIN - 3, FIELD_Y_MAX + 3],
            showgrid=False,
            zeroline=False,
            visible=False,
            row=1,
            col=col,
        )

    fig.update_layout(
        title=title,
        width=max(380 * len(teams), 560),
        height=720,
        plot_bgcolor="#f6faf5",
        paper_bgcolor="white",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def _add_field_shapes(fig, row=None, col=None):
    line_color = "#1B1E26"
    fill_color = "#86d973"
    shape_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
    trace_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}

    fig.add_shape(
        type="rect",
        x0=FIELD_X_MIN,
        y0=FIELD_Y_MIN,
        x1=FIELD_X_MAX,
        y1=FIELD_Y_MAX,
        line={"color": line_color, "width": 2},
        fillcolor=fill_color,
        layer="below",
        **shape_kwargs,
    )
    for y in [ENDZONE_LOW_Y, ENDZONE_HIGH_Y]:
        fig.add_shape(
            type="line",
            x0=FIELD_X_MIN,
            y0=y,
            x1=FIELD_X_MAX,
            y1=y,
            line={"color": line_color, "width": 1.5},
            layer="below",
            **shape_kwargs,
        )
    for y in [40, 80]:
        fig.add_trace(
            go_scatter(
                x=[0],
                y=[y],
                mode="markers",
                marker={"size": 5, "color": line_color},
                hoverinfo="skip",
                showlegend=False,
            ),
            **trace_kwargs,
        )


def go_scatter(**kwargs):
    import plotly.graph_objects as go

    return go.Scatter(**kwargs)


def plot_average_scoring_path(
    average_path,
    paths=None,
    title="Average scoring path",
    show_individual_paths=True,
):
    import plotly.graph_objects as go

    fig = go.Figure()
    _add_field_shapes(fig)

    if show_individual_paths and paths:
        for path in paths:
            points = _path_points(path)
            fig.add_trace(
                go.Scatter(
                    x=points["x"],
                    y=points["y"],
                    mode="lines",
                    line={"color": "rgba(170, 61, 35, 0.18)", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=average_path["x"],
            y=average_path["y"],
            mode="lines+markers",
            line={"color": "#b74126", "width": 5},
            marker={"size": 9, "color": "#b74126"},
            customdata=np.stack(
                [
                    average_path["checkpoint"],
                    average_path["mean_cumulative_aec"],
                    average_path["mean_cp"],
                    average_path["possessions"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Progress: %{customdata[0]:.0%}<br>"
                "Avg cumulative aEC: %{customdata[1]:.3f}<br>"
                "Avg CP: %{customdata[2]:.1%}<br>"
                "Possessions: %{customdata[3]:.0f}<extra></extra>"
            ),
            name="Average path",
        )
    )

    for start, end in zip(average_path.iloc[:-1].itertuples(), average_path.iloc[1:].itertuples()):
        fig.add_annotation(
            x=end.x,
            y=end.y,
            ax=start.x,
            ay=start.y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=2,
            arrowcolor="#b74126",
            opacity=0.8,
        )

    _apply_field_layout(fig, title)
    return fig


def _apply_field_layout(fig, title, width=560, height=780):
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor="#f6faf5",
        paper_bgcolor="white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis={
            "range": [FIELD_X_MIN - 5, FIELD_X_MAX + 5],
            "showgrid": False,
            "zeroline": False,
            "visible": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "range": [FIELD_Y_MIN - 3, FIELD_Y_MAX + 3],
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
    )


def plot_scoring_heatmap(paths, title="Scoring possession catch-location heatmap"):
    import plotly.graph_objects as go

    catch_points = []
    for path in paths:
        catch_points.append(path[["ReceiverX", "ReceiverY"]].rename(
            columns={"ReceiverX": "x", "ReceiverY": "y"}
        ))
    catches = pd.concat(catch_points, ignore_index=True) if catch_points else pd.DataFrame()

    fig = go.Figure()
    _add_field_shapes(fig)
    if not catches.empty:
        fig.add_trace(
            go.Histogram2dContour(
                x=catches["x"],
                y=catches["y"],
                colorscale="Hot",
                contours={"coloring": "heatmap"},
                opacity=0.65,
                showscale=True,
                name="Catch density",
            )
        )
    _apply_field_layout(fig, title)
    return fig
