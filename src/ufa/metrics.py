import numpy as np


ADVANCED_VALUE_COLUMNS = [
    "xcp",
    "t_ec",
    "r_ec",
    "total_ec",
    "t_aec",
    "r_aec",
    "total_aec",
    "wpa",
    "etv",
]


def _safe_divide(numerator, denominator):
    return np.where(denominator == 0, np.nan, numerator / denominator)


def add_throw_metric_columns(throws, huck_distance=40):
    throws = throws.copy()

    throws["field_x_delta"] = throws["endX"] - throws["throwerX"]
    throws["field_y_delta"] = throws["endY"] - throws["throwerY"]
    throws["abs_field_y_delta"] = throws["field_y_delta"].abs()
    throws["huck_attempt"] = throws["throw_distance"] >= huck_distance
    throws["huck_completion"] = throws["huck_attempt"] & throws["completion"].astype(bool)
    throws["is_goal"] = throws["completion"].astype(bool) & (throws["receiverY"] > 100)
    throws["throwing_yards"] = np.where(
        throws["completion"].astype(bool),
        throws["receiverY"].clip(0, 100) - throws["throwerY"].clip(0, 100),
        0,
    )
    throws["receiving_yards"] = throws["throwing_yards"]

    if "xcp" in throws.columns:
        throws["completion_over_expected"] = throws["completion"] - throws["xcp"]

    return throws


def _advanced_aggregations(throws):
    aggregations = {}

    for column in ADVANCED_VALUE_COLUMNS:
        if column in throws.columns:
            aggregations[f"total_{column}"] = (column, "sum")
            aggregations[f"avg_{column}"] = (column, "mean")

    if "xcp" in throws.columns:
        aggregations["expected_completions"] = ("xcp", "sum")

    if "completion_over_expected" in throws.columns:
        aggregations["completions_over_expected"] = (
            "completion_over_expected",
            "sum",
        )

    return aggregations


def calculate_thrower_stats(throws, huck_distance=40):
    throws = add_throw_metric_columns(throws, huck_distance=huck_distance)

    player_stats = (
        throws
        .groupby("thrower")
        .agg(
            attempts=("completion", "count"),
            completions=("completion", "sum"),
            turnovers=("turnover", "sum"),
            huck_attempts=("huck_attempt", "sum"),
            huck_completions=("huck_completion", "sum"),
            avg_throw_distance=("throw_distance", "mean"),
            avg_field_y_delta=("field_y_delta", "mean"),
            total_field_y_delta=("field_y_delta", "sum"),
            **_advanced_aggregations(throws),
        )
        .reset_index()
    )

    player_stats["completion_pct"] = _safe_divide(
        player_stats["completions"],
        player_stats["attempts"],
    )
    player_stats["huck_rate"] = _safe_divide(
        player_stats["huck_attempts"],
        player_stats["attempts"],
    )
    player_stats["huck_completion_pct"] = _safe_divide(
        player_stats["huck_completions"],
        player_stats["huck_attempts"],
    )

    if "expected_completions" in player_stats.columns:
        player_stats["cpoe"] = _safe_divide(
            player_stats["completions"] - player_stats["expected_completions"],
            player_stats["attempts"],
        )

    return player_stats


def calculate_receiver_stats(throws, huck_distance=40):
    throws = add_throw_metric_columns(throws, huck_distance=huck_distance)
    completions = throws[throws["receiver"].notna()].copy()

    receiver_stats = (
        completions
        .groupby("receiver")
        .agg(
            catches=("completion", "sum"),
            huck_catches=("huck_completion", "sum"),
            avg_received_distance=("throw_distance", "mean"),
            avg_received_field_y_delta=("field_y_delta", "mean"),
            total_received_field_y_delta=("field_y_delta", "sum"),
            **_advanced_aggregations(completions),
        )
        .reset_index()
    )

    return receiver_stats


def _calculate_hockey_assists(throws):
    completed = throws[throws["completion"].astype(bool)].copy()
    completed["previous_thrower"] = completed["thrower"].shift(1)
    completed["previous_receiver"] = completed["receiver"].shift(1)
    completed["previous_team_side"] = completed["team_side"].shift(1)

    hockey_assists = completed[
        completed["is_goal"]
        & completed["previous_thrower"].notna()
        & completed["previous_receiver"].eq(completed["thrower"])
        & completed["previous_team_side"].eq(completed["team_side"])
    ]

    return (
        hockey_assists
        .groupby("previous_thrower")
        .size()
        .rename("HA")
        .rename_axis("player")
        .reset_index()
    )


def _player_metric(frame, player_column, metric_column, metric_name, agg_func="sum"):
    return (
        frame
        .dropna(subset=[player_column])
        .groupby(player_column)[metric_column]
        .agg(agg_func)
        .rename(metric_name)
        .rename_axis("player")
        .reset_index()
    )


def calculate_box_score_stats(throws, huck_distance=40):
    throws = add_throw_metric_columns(throws, huck_distance=huck_distance)

    thrower_stats = (
        throws
        .groupby("thrower")
        .agg(
            attempts=("completion", "count"),
            T=("turnover", "sum"),
            C=("completion", "sum"),
            A=("is_goal", "sum"),
            huck_attempts=("huck_attempt", "sum"),
            huck_completions=("huck_completion", "sum"),
            throwing_yards=("throwing_yards", "sum"),
            **_advanced_aggregations(throws),
        )
        .rename_axis("player")
        .reset_index()
    )

    goals = _player_metric(throws[throws["is_goal"]], "receiver", "is_goal", "G")
    receiving_yards = _player_metric(
        throws[throws["completion"].astype(bool)],
        "receiver",
        "receiving_yards",
        "receiving_yards",
    )
    hockey_assists = _calculate_hockey_assists(throws)

    if "defender" in throws.columns:
        blocks = _player_metric(
            throws[throws["defender"].notna()],
            "defender",
            "turnover",
            "B",
        )
    else:
        blocks = None

    box_score = thrower_stats
    for metric in [goals, receiving_yards, hockey_assists, blocks]:
        if metric is not None:
            box_score = box_score.merge(metric, on="player", how="outer")

    fill_zero_columns = [
        "attempts",
        "T",
        "C",
        "A",
        "G",
        "HA",
        "B",
        "huck_attempts",
        "huck_completions",
        "throwing_yards",
        "receiving_yards",
    ]
    for column in fill_zero_columns:
        if column not in box_score.columns:
            box_score[column] = 0
        box_score[column] = box_score[column].fillna(0)

    box_score["CP%"] = _safe_divide(box_score["C"], box_score["attempts"])
    box_score["HuR"] = _safe_divide(box_score["huck_attempts"], box_score["attempts"])
    box_score["HuCP%"] = _safe_divide(
        box_score["huck_completions"],
        box_score["huck_attempts"],
    )
    box_score["total_yards"] = box_score["throwing_yards"] + box_score["receiving_yards"]
    box_score["plus_minus"] = box_score["G"] + box_score["A"] - box_score["T"]

    if "expected_completions" in box_score.columns:
        box_score["CPOE"] = _safe_divide(
            box_score["C"] - box_score["expected_completions"],
            box_score["attempts"],
        )

    ordered_columns = [
        "player",
        "G",
        "A",
        "HA",
        "T",
        "B",
        "C",
        "CP%",
        "HuR",
        "HuCP%",
        "attempts",
        "huck_attempts",
        "huck_completions",
        "throwing_yards",
        "receiving_yards",
        "total_yards",
        "plus_minus",
    ]
    if "CPOE" in box_score.columns:
        ordered_columns.append("CPOE")

    remaining_columns = [
        column for column in box_score.columns if column not in ordered_columns
    ]

    return box_score[ordered_columns + remaining_columns].sort_values(
        ["plus_minus", "C", "G"],
        ascending=False,
    ).reset_index(drop=True)


def calculate_team_stats(throws, huck_distance=40):
    throws = add_throw_metric_columns(throws, huck_distance=huck_distance)

    team_stats = (
        throws
        .groupby("team_side")
        .agg(
            attempts=("completion", "count"),
            completions=("completion", "sum"),
            turnovers=("turnover", "sum"),
            huck_attempts=("huck_attempt", "sum"),
            huck_completions=("huck_completion", "sum"),
            avg_throw_distance=("throw_distance", "mean"),
            avg_field_y_delta=("field_y_delta", "mean"),
            total_field_y_delta=("field_y_delta", "sum"),
            **_advanced_aggregations(throws),
        )
        .reset_index()
    )

    team_stats["completion_pct"] = _safe_divide(
        team_stats["completions"],
        team_stats["attempts"],
    )
    team_stats["huck_rate"] = _safe_divide(
        team_stats["huck_attempts"],
        team_stats["attempts"],
    )
    team_stats["huck_completion_pct"] = _safe_divide(
        team_stats["huck_completions"],
        team_stats["huck_attempts"],
    )

    if "expected_completions" in team_stats.columns:
        team_stats["cpoe"] = _safe_divide(
            team_stats["completions"] - team_stats["expected_completions"],
            team_stats["attempts"],
        )

    return team_stats
