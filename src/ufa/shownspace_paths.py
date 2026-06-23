import time
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
    labeled["style"] = np.select(conditions, choices, default="balanced")
    return labeled


def cluster_scoring_possessions(possessions, n_clusters=4, random_state=0):
    """Cluster scoring possessions by shape and efficiency features."""
    if possessions.empty:
        return possessions.copy()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    clustered = add_possession_style_labels(possessions)
    features = _style_features(clustered)
    cluster_count = min(n_clusters, len(clustered))
    if cluster_count <= 1:
        clustered["path_cluster"] = 0
        return clustered

    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init="auto")
    clustered["path_cluster"] = model.fit_predict(scaled)
    return clustered


def summarize_path_clusters(possessions):
    if possessions.empty:
        return pd.DataFrame()

    summary = (
        possessions.groupby(["path_cluster", "style"], dropna=False)
        .agg(
            possessions=("possession_id", "count"),
            avg_throws=("throw_count", "mean"),
            avg_aec_per_throw=("aec_per_throw", "mean"),
            avg_cp=("mean_cp", "mean"),
            avg_yards_per_throw=("yards_per_throw", "mean"),
            avg_max_throw_distance=("max_throw_distance", "mean"),
            avg_resets=("reset_count", "mean"),
            avg_lateral_yards=("lateral_yards", "mean"),
        )
        .reset_index()
        .sort_values(["possessions", "avg_aec_per_throw"], ascending=[False, False])
    )
    return summary


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
    features = _style_features(possessions)
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


def render_shownspace_possession_svg(path, width=260, height=560):
    """Return a Shown Space-style SVG field for one scoring possession."""
    path = path.sort_values("possession_throw").copy()
    possession_id = str(path["possession_id"].iloc[0]) if "possession_id" in path else "path"
    detail_id = f"ufa-throw-detail-{abs(hash(possession_id))}"
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
    for index, (_, throw) in enumerate(path.iterrows(), start=1):
        x1 = sx(throw["ThrowerX"])
        y1 = sy(throw["ThrowerY"])
        x2 = sx(throw["ReceiverX"])
        y2 = sy(throw["ReceiverY"])
        goal_class = " goal-throw" if bool(float(throw["ReceiverY"]) > ENDZONE_HIGH_Y) else ""
        detail_html = json.dumps(throw_detail_html(index, throw))
        update_detail = (
            f"document.getElementById({json.dumps(detail_id)}).innerHTML = "
            f"{detail_html};"
        )
        throw_shapes.append(
            f'<g class="ufa-throw{goal_class}" '
            f'onmouseover="{escape(update_detail, quote=True)}" '
            f'onclick="{escape(update_detail, quote=True)}">'
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" />'
            f'<circle class="throw-start" cx="{x1:.2f}" cy="{y1:.2f}" r="2.8" />'
            f'<circle class="throw-end" cx="{x2:.2f}" cy="{y2:.2f}" r="3.1" />'
            "</g>"
        )

    css = """
    <style>
      .ufa-browser-svg { background: #f7faf7; display: block; }
      .ufa-field { fill: #86d973; stroke: #071019; stroke-width: 2.2; }
      .ufa-yard-line { stroke: #071019; stroke-width: 1.6; }
      .ufa-center-dot { fill: #071019; stroke: none; }
      .ufa-throw line { stroke: #071019; stroke-width: 2.2; stroke-linecap: round; }
      .ufa-throw circle { fill: #071019; stroke: #071019; stroke-width: 1.2; }
      .ufa-throw.goal-throw line { stroke: #c3482b; }
      .ufa-throw.goal-throw .throw-end { fill: #c3482b; stroke: #071019; stroke-width: 1.8; }
      .ufa-throw:hover line { stroke: #c3482b; stroke-width: 3.2; }
      .ufa-throw:hover circle { fill: #c3482b; stroke: #071019; stroke-width: 1.8; }
      .ufa-throw { cursor: pointer; }
      .ufa-browser-field-wrap {
        display: flex;
        align-items: flex-start;
        gap: 12px;
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
    return (
        f'<div class="ufa-browser-field-wrap">'
        f"{css}"
        f'<svg class="ufa-browser-svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f"{''.join(shapes)}{''.join(throw_shapes)}</svg>"
        f'<div id="{detail_id}" class="ufa-throw-detail">'
        f'<div class="ufa-detail-placeholder">Hover or click a throw on the field.</div>'
        f"</div></div>"
    )


def render_possession_browser_summary(possession, path):
    game_id = escape(str(possession.get("GameID", "-")))
    team_id = escape(str(possession.get("team_id", "-")).title())
    side = "Home" if bool(possession.get("is_home_team", False)) else "Away"
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
      <div>Q{quarter} - point {quarter_point} - possession {possession_num} - {side}</div>
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
      </div>
    </div>
    """


def create_scoring_possession_browser(possessions, paths, title="Scoring possessions"):
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

    browser_possessions = _sort_browser_possessions(possessions)
    lookup = _browser_path_lookup(paths)
    browser_possessions = browser_possessions[
        browser_possessions["possession_id"].isin(lookup)
    ].reset_index(drop=True)
    if browser_possessions.empty:
        return widgets.HTML("<b>No matching possession paths available.</b>")

    options = []
    for index, row in browser_possessions.iterrows():
        label = (
            f"{index + 1}. {row['GameID']} | Q{row['game_quarter']} "
            f"P{row['quarter_point']} | poss {row['possession_num']} | "
            f"{int(row['throw_count'])} throws"
        )
        options.append((label, index))

    header = widgets.HTML(
        f"<h2 style='margin:0 0 8px;color:#223a5e;font-family:system-ui'>{escape(title)}</h2>"
    )
    dropdown = widgets.Dropdown(
        options=options,
        value=0,
        description="Possession",
        layout=widgets.Layout(width="430px"),
        style={"description_width": "85px"},
    )
    previous_button = widgets.Button(description="Previous", layout=widgets.Layout(width="95px"))
    next_button = widgets.Button(description="Next", layout=widgets.Layout(width="95px"))
    count_label = widgets.HTML()
    summary_html = widgets.HTML()
    field_html = widgets.HTML()

    def update(index):
        row = browser_possessions.iloc[index]
        path = lookup[row["possession_id"]]
        count_label.value = f"<b>{index + 1}</b> of <b>{len(browser_possessions)}</b>"
        summary_html.value = render_possession_browser_summary(row, path)
        field_html.value = render_shownspace_possession_svg(path)

    def on_dropdown_change(change):
        if change["name"] == "value" and change["new"] is not None:
            update(change["new"])

    def on_previous(_):
        dropdown.value = max(0, dropdown.value - 1)

    def on_next(_):
        dropdown.value = min(len(browser_possessions) - 1, dropdown.value + 1)

    dropdown.observe(on_dropdown_change, names="value")
    previous_button.on_click(on_previous)
    next_button.on_click(on_next)
    update(0)

    controls = widgets.VBox(
        [
            header,
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
