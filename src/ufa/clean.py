import numpy as np


def clean_game_events(events):
    throws = events[events["thrower"].notna()].copy()

    throws["completion"] = throws["receiver"].notna().astype(int)
    throws["turnover"] = throws["turnoverX"].notna().astype(int)

    throws["endX"] = throws["receiverX"].fillna(throws["turnoverX"])
    throws["endY"] = throws["receiverY"].fillna(throws["turnoverY"])

    throws["throw_distance"] = np.sqrt(
        (throws["endX"] - throws["throwerX"]) ** 2
        + (throws["endY"] - throws["throwerY"]) ** 2
    )

    return throws