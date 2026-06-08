import numpy as np

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
    frame["completion_target"] = frame["completion"].astype(int)
    frame["eventual_score_target"] = (
        frame["receiver_y"].fillna(frame["thrower_y"]) > 100
    ).astype(int)
    return frame


def train_completion_probability_model(
    throws,
    features=None,
    model=None,
    scaler=None,
):
    frame = prepare_model_training_frame(throws)
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
    features = features or DEFAULT_FV_FEATURES
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
