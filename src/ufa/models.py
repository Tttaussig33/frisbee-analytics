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


def _clean_feature_frame(frame, features):
    x = frame[features].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True))


def _fit_classifier_model(frame, features, target, model=None, scaler=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    model = model or LogisticRegression(max_iter=1000, random_state=0)
    scaler = scaler or StandardScaler()

    x = _clean_feature_frame(frame, features)
    y = frame[target].astype(int)

    x_scaled = scaler.fit_transform(x)
    model.fit(x_scaled, y)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
    }


def _predict_classifier_probability(model_bundle, frame):
    features = model_bundle["features"]
    x = _clean_feature_frame(frame, features)
    scaler = model_bundle.get("scaler")
    if scaler is not None:
        x = scaler.transform(x)
    return model_bundle["model"].predict_proba(x)[:, 1]


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


def split_training_data(
    throws,
    random_test_size=0.2,
    temporal_game_count=75,
    player_holdout_count=50,
    player_min_throws=200,
    random_state=0,
):
    frame = prepare_all_games_training_data(throws)

    if "gameDate" not in frame.columns and "gameID" in frame.columns:
        frame["gameDate"] = pd.to_datetime(
            frame["gameID"].astype(str).str[:10],
            errors="coerce",
        )
    if "year" not in frame.columns and "gameDate" in frame.columns:
        frame["year"] = frame["gameDate"].dt.year

    rng = np.random.default_rng(random_state)

    player_test = frame.iloc[0:0].copy()
    model_pool = frame
    if "thrower" in frame.columns:
        thrower_counts = frame.groupby("thrower").size()
        candidates = thrower_counts[thrower_counts > player_min_throws].index.to_numpy()
        if len(candidates) > 0 and player_holdout_count > 0:
            holdout_count = min(player_holdout_count, len(candidates))
            holdout_throwers = rng.choice(candidates, size=holdout_count, replace=False)
            player_test = frame[frame["thrower"].isin(holdout_throwers)].copy()
            model_pool = frame[~frame["thrower"].isin(holdout_throwers)].copy()

    temporal_test = model_pool.iloc[0:0].copy()
    split_pool = model_pool
    if "gameID" in model_pool.columns and "gameDate" in model_pool.columns:
        game_dates = (
            model_pool[["gameID", "gameDate"]]
            .drop_duplicates("gameID")
            .sort_values("gameDate")
        )
        if temporal_game_count > 0 and len(game_dates) > temporal_game_count:
            temporal_ids = game_dates["gameID"].tail(temporal_game_count)
            temporal_test = model_pool[model_pool["gameID"].isin(temporal_ids)].copy()
            split_pool = model_pool[~model_pool["gameID"].isin(temporal_ids)].copy()

    validation = split_pool.iloc[0:0].copy()
    train = split_pool
    if "gameID" in split_pool.columns and random_test_size > 0:
        if "year" in split_pool.columns and split_pool["year"].notna().any():
            validation_ids = []
            for _, year_games in split_pool.groupby("year")["gameID"]:
                game_ids = pd.Series(year_games.unique())
                sample_size = int(round(len(game_ids) * random_test_size))
                if sample_size > 0 and len(game_ids) > sample_size:
                    validation_ids.extend(
                        rng.choice(game_ids.to_numpy(), size=sample_size, replace=False)
                    )
        else:
            game_ids = pd.Series(split_pool["gameID"].unique())
            sample_size = int(round(len(game_ids) * random_test_size))
            validation_ids = (
                rng.choice(game_ids.to_numpy(), size=sample_size, replace=False)
                if sample_size > 0 and len(game_ids) > sample_size
                else []
            )

        validation = split_pool[split_pool["gameID"].isin(validation_ids)].copy()
        train = split_pool[~split_pool["gameID"].isin(validation_ids)].copy()

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "temporal_test": temporal_test.reset_index(drop=True),
        "player_test": player_test.reset_index(drop=True),
    }


def calculate_classifier_metrics(y_true, y_probability, threshold=0.5):
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_probability = np.asarray(y_probability)
    y_pred = (y_probability >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probability)
    else:
        auc = np.nan

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    return {
        "n": len(y_true),
        "positive_rate": y_true.mean() if len(y_true) else np.nan,
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
        "auc": auc,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
    }


def evaluate_classifier_model(model_bundle, frame, target, threshold=0.5):
    if frame.empty:
        return {
            "n": 0,
            "positive_rate": np.nan,
            "accuracy": np.nan,
            "auc": np.nan,
            "ppv": np.nan,
            "npv": np.nan,
        }

    y_probability = _predict_classifier_probability(model_bundle, frame)
    return calculate_classifier_metrics(frame[target], y_probability, threshold=threshold)


def evaluate_etv_model_bundle(model_bundle, datasets):
    rows = []
    for dataset_name, frame in datasets.items():
        prepared = prepare_model_training_frame(frame)

        cp_frame = prepared[prepared["throw_distance"] >= 1.5].copy()
        cp_metrics = evaluate_classifier_model(
            model_bundle["cp_model"],
            cp_frame,
            target="completion_target",
        )
        rows.append({"dataset": dataset_name, "model": "cp", **cp_metrics})

        fv_metrics = evaluate_classifier_model(
            model_bundle["fv_model"],
            prepared,
            target="eventual_score_target",
        )
        rows.append({"dataset": dataset_name, "model": "fv", **fv_metrics})

    return pd.DataFrame(rows)


def train_etv_models_from_split(
    splits,
    cp_features=None,
    fv_features=None,
    cp_model=None,
    fv_model=None,
):
    train = splits["train"]
    cp_model_bundle = train_completion_probability_model(
        train,
        features=cp_features,
        model=cp_model,
    )
    fv_model_bundle = train_field_value_model(
        train,
        features=fv_features,
        model=fv_model,
    )

    return {
        "cp_model": cp_model_bundle,
        "fv_model": fv_model_bundle,
    }


def train_xgboost_etv_models_from_split(
    splits,
    cp_features=None,
    fv_features=None,
    cp_params=None,
    fv_params=None,
):
    from xgboost import XGBClassifier

    default_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": 0,
    }

    cp_settings = {**default_params, **(cp_params or {})}
    fv_settings = {**default_params, **(fv_params or {})}

    return train_etv_models_from_split(
        splits,
        cp_features=cp_features,
        fv_features=fv_features,
        cp_model=XGBClassifier(**cp_settings),
        fv_model=XGBClassifier(**fv_settings),
    )


def compare_model_bundles(model_bundles, datasets):
    tables = []
    for model_name, model_bundle in model_bundles.items():
        table = evaluate_etv_model_bundle(model_bundle, datasets)
        table.insert(0, "bundle", model_name)
        tables.append(table)

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def tune_xgboost_classifier(
    frame,
    features,
    target,
    n_trials=15,
    cv=5,
    random_state=0,
):
    import optuna
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    x = _clean_feature_frame(frame, features)
    y = frame[target].astype(int)

    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 25, 200),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            eval_metric="logloss",
            random_state=random_state,
        )
        pipeline = make_pipeline(StandardScaler(), model)
        return cross_val_score(pipeline, x, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_model = XGBClassifier(
        **study.best_params,
        eval_metric="logloss",
        random_state=random_state,
    )
    return _fit_classifier_model(
        frame,
        features=features,
        target=target,
        model=best_model,
    ), study


def train_tuned_xgboost_etv_models_from_split(
    splits,
    cp_features=None,
    fv_features=None,
    n_trials=15,
    cv=5,
    random_state=0,
):
    train = prepare_model_training_frame(splits["train"])
    cp_features = cp_features or DEFAULT_CP_FEATURES
    fv_features = fv_features or DEFAULT_FV_TRAINING_FEATURES

    cp_train = train[train["throw_distance"] >= 1.5].copy()
    cp_model, cp_study = tune_xgboost_classifier(
        cp_train,
        cp_features,
        "completion_target",
        n_trials=n_trials,
        cv=cv,
        random_state=random_state,
    )
    fv_model, fv_study = tune_xgboost_classifier(
        train,
        fv_features,
        "eventual_score_target",
        n_trials=n_trials,
        cv=cv,
        random_state=random_state,
    )

    return {
        "cp_model": cp_model,
        "fv_model": fv_model,
    }, {
        "cp_study": cp_study,
        "fv_study": fv_study,
    }


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
