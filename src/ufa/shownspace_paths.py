import time

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


def _add_path_arrows(fig, points, color, every=1, opacity=0.85):
    for index, (start, end) in enumerate(zip(points.iloc[:-1].itertuples(), points.iloc[1:].itertuples())):
        if index % every != 0:
            continue
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
            arrowsize=1,
            arrowwidth=1.8,
            arrowcolor=color,
            opacity=opacity,
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


def _add_field_shapes(fig):
    line_color = "#1B1E26"
    fill_color = "#86d973"
    fig.add_shape(
        type="rect",
        x0=FIELD_X_MIN,
        y0=FIELD_Y_MIN,
        x1=FIELD_X_MAX,
        y1=FIELD_Y_MAX,
        line={"color": line_color, "width": 2},
        fillcolor=fill_color,
        layer="below",
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
            )
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
