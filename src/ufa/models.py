import numpy as np
import pandas as pd

from .etv import DEFAULT_FV_FEATURES, ExpectedThrowingValueModel, prepare_etv_features


DEFAULT_CP_FEATURES = [
    "thrower_x",
    "thrower_y",
    "receiver_x",
    "receiver_y",
    "throw_distance",
    "throw_angle",
    "y_diff",
    "x_diff",
    "times",
]

DEFAULT_FV_TRAINING_FEATURES = [
    "thrower_x",
    "thrower_y",
    "times",
]


def _fit_classifier_model(frame, features, target, model=None, scaler=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    model = model or LogisticRegression(max_iter=1000, random_state=0)
    scaler = scaler or StandardScaler()

    x = frame[features].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    y = frame[target].astype(int)

    x_scaled = scaler.fit_transform(x)
    model.fit(x_scaled, y)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
    }


def prepare_model_training_frame(throws):
    frame = prepare_etv_features(throws)
    if "completion" not in frame.columns:
        frame["completion"] = 1 - frame["turnover"].astype(int)
    if "point_outcome" not in frame.columns:
        frame = add_point_outcome(frame)

    frame["completion_target"] = frame["completion"].astype(int)
    frame["eventual_score_target"] = frame["point_outcome"].astype(int)
    return frame


def add_point_outcome(frame):
    frame = frame.copy()

    if "completion" not in frame.columns:
        frame["completion"] = 1 - frame["turnover"].astype(int)

    if "receiver_y" not in frame.columns:
        frame = prepare_etv_features(frame)

    scoring_throw = frame["completion"].astype(bool) & (frame["receiver_y"] > 100)

    point_keys = [
        column
        for column in ["gameID", "home_team_score", "away_team_score", "game_quarter"]
        if column in frame.columns
    ]
    if len(point_keys) == 4 and "is_home_team" in frame.columns:
        scoring_teams = (
            frame.loc[scoring_throw, point_keys + ["is_home_team"]]
            .drop_duplicates(point_keys, keep="last")
            .rename(columns={"is_home_team": "scoring_is_home_team"})
        )
        frame = frame.merge(scoring_teams, on=point_keys, how="left")
        frame["point_outcome"] = frame["is_home_team"].eq(
            frame["scoring_is_home_team"]
        ).astype(int)
        frame = frame.drop(columns=["scoring_is_home_team"])
        return frame

    possession_keys = [
        column
        for column in [
            "gameID",
            "home_team_score",
            "away_team_score",
            "possession_num",
            "game_quarter",
        ]
        if column in frame.columns
    ]
    if len(possession_keys) == 5:
        frame["point_outcome"] = scoring_throw.groupby(
            [frame[column] for column in possession_keys]
        ).transform("max").astype(int)
        return frame

    fallback_keys = [
        column
        for column in ["gameID", "total_points", "possession_num"]
        if column in frame.columns
    ]
    if not fallback_keys:
        frame["point_outcome"] = scoring_throw.astype(int)
        return frame

    frame["point_outcome"] = scoring_throw.groupby(
        [frame[column] for column in fallback_keys]
    ).transform("max").astype(int)

    return frame


def prepare_all_games_training_data(data):
    if isinstance(data, (str, bytes)):
        frame = pd.read_csv(data)
    else:
        frame = data.copy()

    if "completion" not in frame.columns:
        frame["completion"] = 1 - frame["turnover"].astype(int)

    frame = prepare_etv_features(frame)
    frame = add_point_outcome(frame)

    frame["completion_target"] = frame["completion"].astype(int)
    frame["eventual_score_target"] = frame["point_outcome"].astype(int)

    return frame


def train_completion_probability_model(
    throws,
    features=None,
    model=None,
    scaler=None,
    min_throw_distance=1.5,
):
    frame = prepare_model_training_frame(throws)
    frame = frame[frame["throw_distance"] >= min_throw_distance].copy()
    features = features or DEFAULT_CP_FEATURES
    return _fit_classifier_model(
        frame,
        features=features,
        target="completion_target",
        model=model,
        scaler=scaler,
    )


def train_field_value_model(
    throws,
    features=None,
    model=None,
    scaler=None,
):
    frame = prepare_model_training_frame(throws)
    features = features or DEFAULT_FV_TRAINING_FEATURES
    return _fit_classifier_model(
        frame,
        features=features,
        target="eventual_score_target",
        model=model,
        scaler=scaler,
    )


def train_etv_models(throws, cp_features=None, fv_features=None):
    cp_model = train_completion_probability_model(throws, features=cp_features)
    fv_model = train_field_value_model(throws, features=fv_features)

    return {
        "cp_model": cp_model,
        "fv_model": fv_model,
    }


def train_etv_models_from_all_games(data, cp_features=None, fv_features=None):
    training_frame = prepare_all_games_training_data(data)
    return train_etv_models(
        training_frame,
        cp_features=cp_features,
        fv_features=fv_features,
    )


def build_etv_model(model_bundle):
    return ExpectedThrowingValueModel(
        cp_model=model_bundle["cp_model"],
        fv_model=model_bundle["fv_model"],
    )


def save_model_bundle(model_bundle, path):
    import joblib

    joblib.dump(model_bundle, path)
    return path


def load_model_bundle(path):
    import joblib

    return joblib.load(path)
