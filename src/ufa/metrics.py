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
