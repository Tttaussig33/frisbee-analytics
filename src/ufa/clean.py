import numpy as np


THROW_COLUMNS = [
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


def clean_game_events(events):
    events = events.copy()

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

    throws["throw_distance"] = np.sqrt(
        (throws["endX"] - throws["throwerX"]) ** 2
        + (throws["endY"] - throws["throwerY"]) ** 2
    )

    return throws[[column for column in THROW_COLUMNS if column in throws.columns]]
