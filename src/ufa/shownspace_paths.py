import time
import hashlib
import json
from html import escape

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


def fetch_shownspace_games(season=2026, final_only=True, limit=500, timeout=30):
    response = requests.get(
        f"{BASE_URL}/api/games",
        params={"year": season, "limit": limit},
        timeout=timeout,
        headers={"User-Agent": "Ultimate-Frisbee-Analytics educational analysis"},
    )
    response.raise_for_status()
    games = pd.DataFrame(response.json().get("games") or [])
    if games.empty or not final_only:
        return games

    status = games.get("Status", pd.Series("", index=games.index)).fillna("")
    is_final = games.get("is_final", pd.Series(False, index=games.index)).fillna(False)
    final_mask = is_final.astype(bool) | status.str.strip().str.lower().str.startswith(
        "final"
    )
    return games[final_mask].reset_index(drop=True)


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


def fetch_shownspace_season_throws(season=2026, team_id=None, max_games=None, delay=0.15):
    games = fetch_shownspace_games(season=season, final_only=True)
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


def build_scoring_possessions(throws, team_id=None):
    if throws.empty:
        return pd.DataFrame(), []

    throws = throws.copy()
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
        final_throw = path.iloc[-1]
        if not bool(final_throw.get("ReceiverY", 0) > ENDZONE_HIGH_Y):
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
        paths.append(path)

    possessions = pd.DataFrame(possession_rows)
    return possessions, paths


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
    avg_throws = _cluster_average(group, "throw_count")
    cluster_size = len(group)

    if pd.notna(avg_hucks) and avg_hucks >= 0.5:
        return "huck"
    if pd.notna(avg_resets) and avg_resets >= 3:
        return "reset-heavy"
    if pd.notna(avg_throws) and avg_throws <= 3:
        return "quick strike"
    if pd.notna(avg_throws) and avg_throws >= 10:
        return "methodical"
    if cluster_size <= 5:
        return "outlier"
    return "mixed"


def _cluster_geometry_descriptors(group):
    width = _cluster_average(group, "shape_width")
    directness = _cluster_average(group, "shape_directness")
    side_switches = _cluster_average(group, "shape_side_switches")
    middle_usage = _cluster_average(group, "shape_middle_third_share")
    sideline_usage = _cluster_average(group, "shape_sideline_share")

    lane_descriptor = None
    if pd.notna(middle_usage) and middle_usage >= 0.55:
        lane_descriptor = "middle-lane"
    elif pd.notna(sideline_usage) and sideline_usage >= 0.25:
        lane_descriptor = "sideline"

    movement_descriptor = None
    if pd.notna(width):
        if width <= 18:
            movement_descriptor = "narrow"
        elif width >= 34:
            movement_descriptor = "wide"

    if pd.notna(side_switches) and side_switches >= 3:
        movement_descriptor = (
            f"{movement_descriptor}/switch-heavy"
            if movement_descriptor
            else "switch-heavy"
        )

    directness_descriptor = None
    if pd.notna(directness):
        if directness >= 0.75:
            directness_descriptor = "direct"
        elif directness <= 0.50:
            directness_descriptor = "winding"

    descriptors = []
    if lane_descriptor is not None:
        descriptors.append(lane_descriptor)
    if movement_descriptor is not None:
        descriptors.append(movement_descriptor)
    if directness_descriptor is not None:
        descriptors.append(directness_descriptor)
    return descriptors[:2]


def _describe_shape_cluster(cluster, group):
    cluster_id = int(cluster)
    primary_style = _cluster_primary_style(group)
    descriptors = _cluster_geometry_descriptors(group)
    descriptor_text = f", {'/'.join(descriptors)}" if descriptors else ""
    return f"Shape {cluster_id}: {primary_style}{descriptor_text}"


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

    tags = []
    if pd.notna(hucks) and hucks >= 1:
        tags.append("huck")
    elif pd.notna(resets) and resets >= 3:
        tags.append("reset-heavy")
    elif pd.notna(throws) and throws <= 3:
        tags.append("quick strike")
    elif pd.notna(throws) and throws >= 10:
        tags.append("methodical")

    if pd.notna(width):
        if width <= 18:
            tags.append("narrow")
        elif width >= 34:
            tags.append("wide")
    if pd.notna(side_switches) and side_switches >= 3:
        tags.append("switch-heavy")
    if pd.notna(directness):
        if directness >= 0.75:
            tags.append("direct")
        elif directness <= 0.50:
            tags.append("winding")

    return ", ".join(tags) if tags else "mixed"


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
        .ufa-shape-overview .scroll {{
          max-height: 180px;
          overflow: auto;
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
          group can still look wider, narrower, cleaner, or messier than the label.
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
          <b>Shape names:</b>
          huck = average at least 0.5 hucks per possession;
          reset-heavy = average at least 3 resets;
          quick strike = average 3 or fewer throws;
          methodical = average 10 or more throws;
          mixed = no single simple style dominates;
          outlier = a small mixed group that does not fit a common style cleanly.
          narrow = average width used is 18 yards or less;
          wide = 34 yards or more;
          switch-heavy = average at least 3 side switches;
          direct = directness is 75% or higher;
          winding = directness is 50% or lower;
          middle-lane = at least 55% of touch points are in the middle third;
          sideline = at least 25% are near either sideline.
        </div>
        <div class="guide">
          <b>Columns:</b>
          N = possessions;
          Thr = average throws;
          Width = average field width used;
          Sw = average side switches;
          Dir = directness;
          Mid = middle-third usage;
          Side = sideline usage;
          Hu = average hucks;
          Re = average resets;
          aEC/T = average aEC per throw.
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
      .ufa-browser-svg { background: #f7faf7; display: block; }
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
        gap: 12px;
      }
      .ufa-browser-field-wrap:focus {
        outline: 2px solid #91a7c2;
        outline-offset: 4px;
      }
      .ufa-throw-detail {
        box-sizing: border-box;
        width: 205px;
        min-height: 150px;
        border-left: 1px solid #d8e0e8;
        padding: 8px 0 8px 12px;
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
        margin-bottom: 8px;
      }
      .ufa-detail-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        border-bottom: 1px solid #edf1f5;
        padding: 3px 0;
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


def render_possession_browser_summary(possession, path):
    game_id = escape(str(possession.get("GameID", "-")))
    team_id = escape(str(possession.get("team_id", "-")).title())
    side = "Home" if bool(possession.get("is_home_team", False)) else "Away"
    line_type = escape(_line_type_label(possession.get("line_type")))
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
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
          font-size: 13px;
          line-height: 1.45;
          width: 315px;
        }}
        .ufa-browser-summary h3 {{
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 18px;
          margin: 0 0 8px;
          color: #0b1a33;
        }}
        .ufa-browser-summary .meta {{
          border-top: 1px solid #d9e1ea;
          margin-top: 10px;
          padding-top: 10px;
        }}
        .ufa-browser-summary .row {{
          display: flex;
          justify-content: space-between;
          gap: 16px;
          border-bottom: 1px solid #edf1f5;
          padding: 3px 0;
        }}
        .ufa-browser-summary .label {{ color: #637188; }}
        .ufa-browser-summary .value {{ font-weight: 700; text-align: right; }}
      </style>
      <h3>{team_id}</h3>
      <div>{game_id}</div>
      <div>Q{quarter} - point {quarter_point} - possession {possession_num} - {side} - {line_type}</div>
      <div>Group: {shape_label}</div>
      <div>This possession: {possession_shape_label}</div>
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


def create_scoring_possession_browser(
    possessions,
    paths,
    title="Scoring possessions",
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
        return widgets.HTML("<b>No scoring possessions available.</b>")

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

    header = widgets.HTML(
        f"<h2 style='margin:0 0 8px;color:#223a5e;font-family:system-ui'>{escape(title)}</h2>"
    )
    line_filter = widgets.Dropdown(
        options=[
            ("All lines", "all"),
            ("O-line scores", "o_line"),
            ("D-line scores", "d_line"),
        ],
        value="all",
        description="Line",
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
    dropdown = widgets.Dropdown(
        options=[],
        value=None,
        description="Possession",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    previous_button = widgets.Button(description="Previous", layout=widgets.Layout(width="95px"))
    next_button = widgets.Button(description="Next", layout=widgets.Layout(width="95px"))
    count_label = widgets.HTML()
    summary_html = widgets.HTML()
    shape_overview_html = widgets.HTML()
    field_html = widgets.HTML()
    state = {"possessions": base_possessions}

    def make_options(frame):
        options = []
        for index, row in frame.iterrows():
            label = (
                f"{index + 1}. {_line_type_label(row.get('line_type'))} | "
                f"{_shape_cluster_label(row)} | "
                f"{row['GameID']} | Q{row['game_quarter']} "
                f"P{row['quarter_point']} | poss {row['possession_num']} | "
                f"{int(row['throw_count'])} throws"
            )
            options.append((label, index))
        return options

    def update(index):
        browser_possessions = state["possessions"]
        if index is None or browser_possessions.empty:
            count_label.value = "<b>0</b> of <b>0</b>"
            summary_html.value = "<b>No scoring possessions match these filters.</b>"
            field_html.value = ""
            return

        row = browser_possessions.iloc[index]
        path = lookup[row["possession_id"]]
        count_label.value = f"<b>{index + 1}</b> of <b>{len(browser_possessions)}</b>"
        summary_html.value = render_possession_browser_summary(row, path)
        field_html.value = render_shownspace_possession_svg(path)

    def apply_filters(_=None):
        selected_line = line_filter.value
        min_throw_count, max_throw_count = throw_count_filter.value
        if selected_line == "all":
            filtered = base_possessions.copy()
        else:
            filtered = base_possessions[
                base_possessions["line_type"].eq(selected_line)
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
    shape_filter.observe(apply_filters, names="value")
    throw_count_filter.observe(apply_filters, names="value")
    previous_button.on_click(on_previous)
    next_button.on_click(on_next)
    apply_filters()

    controls = widgets.VBox(
        [
            header,
            line_filter,
            shape_filter,
            shape_overview_html,
            throw_count_filter,
            dropdown,
            widgets.HBox([previous_button, next_button, count_label]),
            summary_html,
        ],
        layout=widgets.Layout(width="450px"),
    )
    return widgets.HBox(
        [controls, field_html],
        layout=widgets.Layout(align_items="flex-start", gap="18px"),
    )


def create_team_scoring_possession_browser(
    season=2026,
    default_team_id="glory",
    final_only=True,
    max_games=None,
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

    games = fetch_shownspace_games(season=season, final_only=final_only)
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
            possessions, paths = build_scoring_possessions(throws, team_id=team_id)

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
            title = f"{team_id.title()} scoring possessions, {season}"
            browser = create_scoring_possession_browser(
                possessions,
                paths,
                title=title,
                n_shape_clusters=n_shape_clusters,
            )
            status.value = (
                f"<b>{escape(team_id.title())}</b>: "
                f"{len(possessions):,} scoring possessions from "
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
    _render_team(team_dropdown.value)
    return widgets.VBox([controls, status, output])


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
    if "quick" in label_lower:
        return "#7a3db8"
    if "methodical" in label_lower:
        return "#2f7d32"
    if "outlier" in label_lower:
        return "#6b7280"
    if "mixed" in label_lower:
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
