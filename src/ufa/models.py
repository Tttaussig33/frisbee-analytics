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


def normalize_processed_game_times(times):
    normalized = pd.to_numeric(times, errors="coerce") / 60
    normalized = normalized.mask(normalized < 0, normalized + 5)
    while normalized.gt(12).any():
        normalized = normalized.mask(normalized > 12, normalized - 12)
    return normalized


def _sort_like_processed_games(frame):
    time_column = "_sort_times" if "_sort_times" in frame.columns else "times"
    sort_columns = ["gameID", "game_quarter", time_column]
    if all(column in frame.columns for column in sort_columns):
        return frame.sort_values(sort_columns, ascending=[True, True, False])
    return frame


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


def _augment_training_frame(frame, features, mirror=True, noise_factor=0.0, random_state=0):
    augmented = frame.copy()

    if mirror:
        mirrored = augmented.copy()
        mirror_columns = [
            column
            for column in features
            if "_x" in column or "x_diff" in column or "throw_angle" in column
        ]
        for column in mirror_columns:
            mirrored[column] = -mirrored[column]
        augmented = pd.concat([augmented, mirrored], ignore_index=True)

    if noise_factor > 0:
        jitter_columns = [
            column
            for column in features
            if pd.api.types.is_numeric_dtype(augmented[column])
            and augmented[column].nunique(dropna=True) > 2
        ]
        if jitter_columns:
            rng = np.random.default_rng(random_state)
            noisy = augmented.copy()
            feature_ranges = (
                augmented[jitter_columns].max() - augmented[jitter_columns].min()
            )
            jitter = rng.uniform(
                -noise_factor,
                noise_factor,
                size=noisy[jitter_columns].shape,
            ) * feature_ranges.to_numpy()
            noisy[jitter_columns] = noisy[jitter_columns] + jitter
            for column in jitter_columns:
                noisy[column] = noisy[column].clip(
                    lower=augmented[column].min(),
                    upper=augmented[column].max(),
                )
            augmented = pd.concat([augmented, noisy], ignore_index=True)

    return augmented


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

    sorted_frame = _sort_like_processed_games(frame)
    scoring_throw = sorted_frame["completion"].astype(bool) & (
        sorted_frame["receiver_y"] > 100
    )

    point_keys = [
        column
        for column in ["gameID", "home_team_score", "away_team_score", "game_quarter"]
        if column in frame.columns
    ]
    if len(point_keys) == 4 and "is_home_team" in frame.columns:
        final_throws = (
            sorted_frame
            .groupby(point_keys, sort=False)
            .tail(1)
        )
        scoring_teams = (
            final_throws.loc[
                final_throws["completion"].astype(bool)
                & (final_throws["receiver_y"] > 100),
                point_keys + ["is_home_team"],
            ]
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
        final_throws = (
            sorted_frame
            .groupby(possession_keys, sort=False)
            .tail(1)
        )
        scoring_possessions = final_throws.loc[
            final_throws["completion"].astype(bool)
            & (final_throws["receiver_y"] > 100),
            possession_keys,
        ].drop_duplicates()
        frame = frame.merge(
            scoring_possessions.assign(point_outcome=1),
            on=possession_keys,
            how="left",
        )
        frame["point_outcome"] = frame["point_outcome"].fillna(0).astype(int)
        return frame

    fallback_keys = [
        column
        for column in ["gameID", "total_points", "possession_num"]
        if column in frame.columns
    ]
    if not fallback_keys:
        frame["point_outcome"] = scoring_throw.astype(int)
        return frame

    frame["point_outcome"] = (
        frame["completion"].astype(bool)
        & (frame["receiver_y"] > 100)
    ).groupby(
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

    if "times" in frame.columns:
        frame["_sort_times"] = pd.to_numeric(frame["times"], errors="coerce")
        frame["times"] = normalize_processed_game_times(frame["times"])
        if "game_quarter" in frame.columns:
            frame.loc[frame["game_quarter"].eq(6), "times"] = 12

    frame = prepare_etv_features(frame)
    frame = add_point_outcome(frame)
    frame = _sort_like_processed_games(frame)

    frame["completion_target"] = frame["completion"].astype(int)
    frame["eventual_score_target"] = frame["point_outcome"].astype(int)

    return frame


def split_training_data(
    throws,
    random_test_size=0.4,
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

    player_test = frame.iloc[0:0].copy()
    model_pool = frame
    if "thrower" in frame.columns:
        thrower_counts = frame.groupby("thrower").size()
        candidates = thrower_counts[thrower_counts > player_min_throws].index.to_numpy()
        if len(candidates) > 0 and player_holdout_count > 0:
            holdout_count = min(player_holdout_count, len(candidates))
            rng = np.random.RandomState(random_state)
            holdout_throwers = rng.choice(candidates, size=holdout_count, replace=False)
            player_test = frame[frame["thrower"].isin(holdout_throwers)].copy()
            model_pool = frame[~frame["thrower"].isin(holdout_throwers)].copy()

    temporal_test = model_pool.iloc[0:0].copy()
    split_pool = model_pool
    validation_pool = split_pool
    if "gameID" in model_pool.columns and "gameDate" in model_pool.columns:
        sorted_pool = model_pool.sort_values("gameDate")
        game_ids = pd.Series(sorted_pool["gameID"].unique())
        if temporal_game_count > 0 and len(game_ids) > temporal_game_count:
            temporal_ids = game_ids.tail(temporal_game_count)
            remaining_ids = game_ids.iloc[:-temporal_game_count]
            temporal_test = model_pool[model_pool["gameID"].isin(temporal_ids)].copy()
            split_pool = model_pool[~model_pool["gameID"].isin(temporal_ids)].copy()
            validation_pool = sorted_pool[
                sorted_pool["gameID"].isin(remaining_ids)
            ].copy()

    validation = split_pool.iloc[0:0].copy()
    train = split_pool
    if "gameID" in split_pool.columns and random_test_size > 0:
        if "year" in split_pool.columns and split_pool["year"].notna().any():
            validation_ids = []
            game_count = validation_pool["gameID"].nunique()
            sample_size = int(game_count * random_test_size / 4)
            for _, year_group in validation_pool.groupby("year"):
                if sample_size > 0 and len(year_group) >= sample_size:
                    sampled = year_group.sample(
                        n=sample_size,
                        random_state=random_state,
                    )
                    validation_ids.extend(sampled["gameID"].unique())
        else:
            game_ids = pd.Series(split_pool["gameID"].unique())
            sample_size = int(round(len(game_ids) * random_test_size))
            validation_ids = (
                np.random.RandomState(random_state).choice(
                    game_ids.to_numpy(),
                    size=sample_size,
                    replace=False,
                )
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


def format_model_performance_table(
    results,
    bundle_labels=None,
    dataset_labels=None,
    dataset_order=None,
    decimals=3,
):
    table_source = results.copy()
    if "bundle" not in table_source.columns:
        table_source["bundle"] = "model"

    bundle_labels = {
        "model": "Model",
        "logistic": "Baseline",
        "baseline": "Baseline",
        "xgboost": "Model",
        **(bundle_labels or {}),
    }
    dataset_labels = {
        "validation": "Random",
        "temporal_test": "Temporal",
        "player_test": "Player",
        "train": "Train",
        **(dataset_labels or {}),
    }
    dataset_order = dataset_order or [
        "validation",
        "temporal_test",
        "player_test",
        "train",
    ]

    metric_order = ["accuracy", "auc", "ppv", "npv"]
    metric_labels = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "ppv": "PPV",
        "npv": "NPV",
    }
    model_labels = {
        "fv": "FV",
        "cp": "CP",
    }

    rows = []
    for _, result in table_source.iterrows():
        dataset = result["dataset"]
        model = result["model"]
        bundle = result["bundle"]
        column_name = (
            f"{model_labels.get(model, str(model).upper())} "
            f"{bundle_labels.get(bundle, str(bundle).title())}"
        )

        for metric in metric_order:
            rows.append(
                {
                    "Dataset": dataset_labels.get(dataset, dataset),
                    "Metric": metric_labels[metric],
                    "Column": column_name,
                    "Value": result[metric],
                    "_dataset_order": (
                        dataset_order.index(dataset)
                        if dataset in dataset_order
                        else len(dataset_order)
                    ),
                    "_metric_order": metric_order.index(metric),
                }
            )

    long_table = pd.DataFrame(rows)
    if long_table.empty:
        return pd.DataFrame()

    column_order = [
        "FV Baseline",
        "FV Model",
        "CP Baseline",
        "CP Model",
    ]
    available_columns = list(long_table["Column"].drop_duplicates())
    column_order = [
        column for column in column_order if column in available_columns
    ] + [column for column in available_columns if column not in column_order]

    long_table = long_table.sort_values(["_dataset_order", "_metric_order"])
    index_order = pd.MultiIndex.from_frame(
        long_table[["Dataset", "Metric"]].drop_duplicates()
    )
    formatted = long_table.pivot_table(
        index=["Dataset", "Metric"],
        columns="Column",
        values="Value",
        aggfunc="first",
    )
    formatted = formatted.reindex(index_order)
    formatted = formatted.reindex(columns=column_order)
    return formatted.round(decimals)


def model_performance_table_to_latex(
    table,
    caption="CP and FV Model performance compared to baseline.",
    label=None,
    use_multirow=True,
):
    def latex_escape(value):
        return str(value).replace("_", "\\_").replace("%", "\\%")

    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
    ]
    if label:
        lines.append(f"\\label{{{latex_escape(label)}}}")

    lines.extend(
        [
            f"\\begin{{tabular}}{{ll{'r' * len(table.columns)}}}",
            "\\toprule",
            " & ".join(["", "Metric", *[latex_escape(column) for column in table.columns]])
            + " \\\\",
            "\\midrule",
        ]
    )

    for dataset, dataset_table in table.groupby(level=0, sort=False):
        row_count = len(dataset_table)
        for row_index, ((_, metric), row) in enumerate(dataset_table.iterrows()):
            if row_index == 0:
                dataset_text = (
                    f"\\multirow{{{row_count}}}{{*}}{{{latex_escape(dataset)}}}"
                    if use_multirow
                    else latex_escape(dataset)
                )
            else:
                dataset_text = ""

            values = [
                "" if pd.isna(value) else f"{value:.3f}"
                for value in row.to_list()
            ]
            lines.append(
                " & ".join([dataset_text, latex_escape(metric), *values]) + " \\\\"
            )
        if dataset != table.index.get_level_values(0)[-1]:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def write_model_performance_latex_document(
    table,
    path,
    caption="CP and FV Model performance compared to baseline.",
    label="tab:cp-fv-performance",
):
    table_latex = model_performance_table_to_latex(
        table,
        caption=caption,
        label=label,
    )
    document = "\n".join(
        [
            "\\documentclass[11pt]{article}",
            "",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{multirow}",
            "\\usepackage{caption}",
            "",
            "\\captionsetup{",
            "  font=small,",
            "  labelfont=bf",
            "}",
            "",
            "\\begin{document}",
            "",
            table_latex,
            "",
            "\\end{document}",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(document)

    return path


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
    mirror=True,
    apply_noise=False,
):
    import optuna
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    def objective(trial):
        noise_factor = trial.suggest_float("noise_factor", 0.0, 0.05)
        trial_frame = _augment_training_frame(
            frame,
            features,
            mirror=mirror,
            noise_factor=noise_factor if apply_noise else 0.0,
            random_state=random_state,
        )
        trial_x = _clean_feature_frame(trial_frame, features)
        trial_y = trial_frame[target].astype(int)
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 10, 200),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
            eval_metric="logloss",
            random_state=random_state,
        )
        pipeline = make_pipeline(StandardScaler(), model)
        return cross_val_score(pipeline, trial_x, trial_y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_frame = _augment_training_frame(
        frame,
        features,
        mirror=mirror,
        noise_factor=study.best_params.get("noise_factor", 0.0) if apply_noise else 0.0,
        random_state=random_state,
    )
    best_params = {
        key: value
        for key, value in study.best_params.items()
        if key != "noise_factor"
    }
    best_model = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=random_state,
    )
    return _fit_classifier_model(
        best_frame,
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
    apply_noise=False,
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
        apply_noise=apply_noise,
    )
    fv_model, fv_study = tune_xgboost_classifier(
        train,
        fv_features,
        "eventual_score_target",
        n_trials=n_trials,
        cv=cv,
        random_state=random_state,
        apply_noise=apply_noise,
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
