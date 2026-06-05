def calculate_thrower_stats(throws):
    player_stats = (
        throws
        .groupby("thrower")
        .agg(
            attempts=("completion", "count"),
            completions=("completion", "sum"),
            turnovers=("turnover", "sum"),
            avg_throw_distance=("throw_distance", "mean")
        )
        .reset_index()
    )

    player_stats["completion_pct"] = (
        player_stats["completions"] / player_stats["attempts"]
    )

    return player_stats