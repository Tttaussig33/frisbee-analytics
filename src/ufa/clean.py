import numpy as np


THROW_COLUMNS = [
    "event_index",
    "type",
    "line",
    "time",
    "thrower",
    "throwerX",
    "throwerY",
    "receiver",
    "receiverX",
    "receiverY",
    "defender",
    "turnoverX",
    "turnoverY",
    "team_side",
    "game_quarter",
    "times",
    "current_line",
    "is_home_team",
    "home_team_score",
    "away_team_score",
    "total_points",
    "possession_num",
    "possession_id",
    "possession_throw",
    "completion",
    "turnover",
    "endX",
    "endY",
    "throw_distance",
]


def _attach_block_defenders(events, throws):
    throws = throws.copy()

    for defender_event in events[
        events["defender"].notna()
        & events["thrower"].isna()
    ].itertuples():
        previous_turnovers = throws[
            (throws["event_index"] < defender_event.event_index)
            & (throws["turnover"] == 1)
            & (throws["defender"].isna())
        ]
        if previous_turnovers.empty:
            continue

        previous_index = previous_turnovers.index[-1]
        if throws.loc[previous_index, "thrower"] == defender_event.defender:
            continue

        throws.loc[previous_index, "defender"] = defender_event.defender

    return throws


def clean_game_events(events):
    events = events.copy()
    events["event_index"] = np.arange(len(events))

    if "time" in events.columns:
        events["event_time"] = events["time"].astype(float)
        events["times"] = events["event_time"].ffill().fillna(0)
    else:
        events["times"] = 0

    events["game_quarter"] = 1

    if "line" in events.columns:
        events["current_line"] = events["line"].replace("", np.nan).ffill()

    coordinate_columns = [
        "throwerX",
        "throwerY",
        "receiverX",
        "receiverY",
        "turnoverX",
        "turnoverY",
    ]
    for column in coordinate_columns:
        events[column] = events[column].astype(float)

    events["endX"] = events["receiverX"].fillna(events["turnoverX"])
    events["endY"] = events["receiverY"].fillna(events["turnoverY"])

    throws = events[
        events["thrower"].notna()
        & events["throwerX"].notna()
        & events["throwerY"].notna()
        & events["endX"].notna()
        & events["endY"].notna()
    ].copy()

    throws["completion"] = throws["receiver"].notna().astype(int)
    throws["turnover"] = (throws["completion"] == 0).astype(int)
    throws = _attach_block_defenders(events, throws)

    throws["is_home_team"] = throws["team_side"].eq("home")

    throws["throw_distance"] = np.sqrt(
        (throws["endX"] - throws["throwerX"]) ** 2
        + (throws["endY"] - throws["throwerY"]) ** 2
    )
    is_goal = throws["completion"].astype(bool) & (
        throws["type"].eq(19) | (throws["receiverY"] > 100)
    )

    throws["home_team_score"] = (
        (is_goal & throws["team_side"].eq("home"))
        .shift(fill_value=False)
        .cumsum()
        .astype(int)
    )
    throws["away_team_score"] = (
        (is_goal & throws["team_side"].eq("away"))
        .shift(fill_value=False)
        .cumsum()
        .astype(int)
    )
    throws["total_points"] = throws["home_team_score"] + throws["away_team_score"]

    point_id = is_goal.shift(fill_value=False).cumsum()
    turnovers_before_throw = (
        throws["turnover"]
        .astype(bool)
        .shift(fill_value=False)
        .groupby(point_id)
        .cumsum()
    )
    throws["possession_num"] = turnovers_before_throw + 1

    terminal_throw = throws["turnover"].astype(bool) | is_goal
    throws["possession_id"] = terminal_throw.shift(fill_value=False).cumsum() + 1
    throws["possession_throw"] = throws.groupby("possession_id").cumcount() + 1

    return throws[[column for column in THROW_COLUMNS if column in throws.columns]]
