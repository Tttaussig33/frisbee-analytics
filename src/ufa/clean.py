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

    throws["throw_distance"] = np.sqrt(
        (throws["endX"] - throws["throwerX"]) ** 2
        + (throws["endY"] - throws["throwerY"]) ** 2
    )

    return throws[[column for column in THROW_COLUMNS if column in throws.columns]]
