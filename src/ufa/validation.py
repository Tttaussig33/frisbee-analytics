import pandas as pd


DEFAULT_METRIC_COLUMN_MAP = {
    "Tot-aEC": "total_aec",
    "T-aEC": "t_aec",
    "R-aEC": "r_aec",
    "WPA": "wpa",
    "G": "G",
    "A": "A",
    "HA": "HA",
    "T": "T",
    "B": "B",
    "C": "C",
    "CP%": "CP%",
    "HuR": "HuR",
    "HuCP%": "HuCP%",
    "xCP": "avg_xcp",
    "CPOE": "CPOE",
    "OE": "OE",
    "DE": "DE",
    "OPts": "OPts",
    "DPts": "DPts",
    "OPoss": "OPoss",
    "DPoss": "DPoss",
    "OI%": "OI%",
}


def normalize_player_key(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def load_reference_stats(path):
    return pd.read_csv(path)


def compare_metric_tables(
    generated,
    reference,
    player_column="player",
    reference_player_column=None,
    metric_column_map=None,
):
    reference_player_column = reference_player_column or player_column
    metric_column_map = metric_column_map or DEFAULT_METRIC_COLUMN_MAP

    generated = generated.copy()
    reference = reference.copy()

    generated["_player_key"] = normalize_player_key(generated[player_column])
    reference["_player_key"] = normalize_player_key(reference[reference_player_column])

    comparisons = []
    for reference_metric, generated_metric in metric_column_map.items():
        if reference_metric not in reference.columns:
            continue
        if generated_metric not in generated.columns:
            continue

        reference_values = reference[
            ["_player_key", reference_player_column, reference_metric]
        ].rename(
            columns={
                reference_player_column: "reference_player",
                reference_metric: "reference_value",
            }
        )
        generated_values = generated[
            ["_player_key", player_column, generated_metric]
        ].rename(
            columns={
                player_column: "generated_player",
                generated_metric: "generated_value",
            }
        )

        merged = reference_values.merge(
            generated_values,
            on="_player_key",
            how="inner",
        )
        if merged.empty:
            continue

        merged["metric"] = reference_metric
        merged["difference"] = merged["generated_value"] - merged["reference_value"]
        merged["absolute_difference"] = merged["difference"].abs()
        merged["percent_difference"] = merged["difference"] / merged[
            "reference_value"
        ].replace(0, pd.NA)

        comparisons.append(
            merged[
                [
                    "metric",
                    "reference_player",
                    "generated_player",
                    "reference_value",
                    "generated_value",
                    "difference",
                    "absolute_difference",
                    "percent_difference",
                ]
            ]
        )

    if not comparisons:
        return pd.DataFrame(
            columns=[
                "metric",
                "reference_player",
                "generated_player",
                "reference_value",
                "generated_value",
                "difference",
                "absolute_difference",
                "percent_difference",
            ]
        )

    return pd.concat(comparisons, ignore_index=True)


def summarize_metric_comparison(comparison):
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "matched_players",
                "mean_absolute_difference",
                "median_absolute_difference",
                "max_absolute_difference",
            ]
        )

    return (
        comparison
        .groupby("metric")
        .agg(
            matched_players=("generated_player", "nunique"),
            mean_absolute_difference=("absolute_difference", "mean"),
            median_absolute_difference=("absolute_difference", "median"),
            max_absolute_difference=("absolute_difference", "max"),
        )
        .reset_index()
        .sort_values("mean_absolute_difference", ascending=False)
    )
