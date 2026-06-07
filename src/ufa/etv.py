import numpy as np


DEFAULT_FV_FEATURES = [
    "thrower_x",
    "thrower_y",
    "times",
    "thrower_x_squared",
    "thrower_y_squared",
    "thrower_interaction_squared",
]


def prepare_etv_features(throws):
    features = throws.copy()

    features["thrower_x"] = features["throwerX"]
    features["thrower_y"] = features["throwerY"]
    features["receiver_x"] = features["receiverX"].fillna(features["endX"])
    features["receiver_y"] = features["receiverY"].fillna(features["endY"])

    features["x_diff"] = features["receiver_x"] - features["thrower_x"]
    features["y_diff"] = features["receiver_y"] - features["thrower_y"]
    features["throw_distance"] = np.sqrt(
        features["x_diff"] ** 2 + features["y_diff"] ** 2
    )
    features["throw_angle"] = np.arctan2(features["y_diff"], features["x_diff"])

    if "times" not in features.columns:
        features["times"] = features["time"].fillna(0) if "time" in features else 0

    features["thrower_x_squared"] = features["thrower_x"] ** 2
    features["thrower_y_squared"] = features["thrower_y"] ** 2
    features["thrower_interaction_squared"] = (
        features["thrower_x"] * features["thrower_y"]
    )
    features["receiver_x_squared"] = features["receiver_x"] ** 2
    features["receiver_y_squared"] = features["receiver_y"] ** 2
    features["receiver_interaction_squared"] = (
        features["receiver_x"] * features["receiver_y"]
    )

    return features


def _model_proba(model, scaler, frame, features):
    x = frame[features]
    if scaler is not None:
        x = scaler.transform(x)
    return model.predict_proba(x)[:, 1]


class ExpectedThrowingValueModel:
    def __init__(self, cp_model, fv_model):
        self.cp_model = cp_model["model"]
        self.fv_model = fv_model["model"]
        self.cp_scaler = cp_model.get("scaler")
        self.fv_scaler = fv_model.get("scaler")
        self.cp_features = cp_model["features"]
        self.fv_features = fv_model.get("features", DEFAULT_FV_FEATURES)

    def _opponent_field_value_frame(self, end_location_frame):
        opponent = end_location_frame[self.fv_features].copy()
        opponent["thrower_x"] = -opponent["thrower_x"]
        opponent["thrower_y"] = (120 - opponent["thrower_y"]).clip(lower=20, upper=100)

        if "possession_num" in opponent.columns:
            opponent["possession_num"] += 1
        if "possession_throw" in opponent.columns:
            opponent["possession_throw"] = 1
        if "score_diff" in opponent.columns:
            opponent["score_diff"] = -opponent["score_diff"]

        return opponent

    def predict_components(self, throws):
        features = prepare_etv_features(throws)

        xcp = _model_proba(
            self.cp_model,
            self.cp_scaler,
            features,
            self.cp_features,
        )
        fv_start = _model_proba(
            self.fv_model,
            self.fv_scaler,
            features,
            self.fv_features,
        )

        receiver_fv_features = [
            feature.replace("thrower", "receiver")
            for feature in self.fv_features
        ]
        end_location = features[receiver_fv_features].rename(
            columns={
                feature.replace("thrower", "receiver"): feature
                for feature in self.fv_features
            }
        )
        fv_end = _model_proba(
            self.fv_model,
            self.fv_scaler,
            end_location,
            self.fv_features,
        )
        fv_end = np.where(features["receiver_y"] > 100, 1, fv_end)

        opponent = self._opponent_field_value_frame(end_location)
        fv_opponent = _model_proba(
            self.fv_model,
            self.fv_scaler,
            opponent,
            self.fv_features,
        )

        etv = (xcp * fv_end) - ((1 - xcp) * fv_opponent)

        return {
            "xcp": xcp,
            "fv_start": fv_start,
            "fv_end": fv_end,
            "fv_opponent": fv_opponent,
            "etv": etv,
        }

    def predict(self, throws):
        return self.predict_components(throws)["etv"]


def add_expected_throwing_value(throws, model):
    throws = throws.copy()
    predictions = model.predict_components(throws)

    for column, values in predictions.items():
        throws[column] = values

    throws["cpoe"] = throws["completion"] - throws["xcp"]
    throws["t_ec"] = throws["etv"] - throws["fv_start"]
    throws["r_ec"] = np.where(throws["completion"].astype(bool), throws["fv_end"], 0)
    throws["total_ec"] = throws["t_ec"] + throws["r_ec"]

    return throws
