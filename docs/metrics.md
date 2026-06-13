# Shown Space-Style Metrics Roadmap

This project uses Shown Space and the Expected Throwing Value work as a model
for advanced ultimate frisbee analytics. The goal is not to copy private outputs
from another site, but to build reproducible metrics from UFA event data and use
public Shown Space tables as validation references.

## Implementation Status

| Group | Metric | Meaning | Status | Notes |
| --- | --- | --- | --- | --- |
| Contribution | Tot-aEC | Total adjusted expected contribution | Model-ready | Computed as `total_aec` when CP/FV model outputs are supplied. |
| Contribution | T-aEC | Throwing adjusted expected contribution | Model-ready | Computed as `t_aec` when CP/FV model outputs are supplied. Current allocation is based on expected throw value contribution. |
| Contribution | R-aEC | Receiving adjusted expected contribution | Model-ready | Computed as `r_aec` when CP/FV model outputs are supplied. Current allocation is the remaining observed aEC after thrower credit. |
| Contribution | LC | Lag contribution | Unknown | Needs exact Shown Space definition/tooltip. |
| Contribution | WPA | Win probability added | Needs model | Requires a win probability model by game state. |
| Box Score | G | Goals | Implemented | Completed catch with receiver end Y beyond the goal line. |
| Box Score | A | Assists | Implemented | Thrower on a goal. |
| Box Score | HA | Hockey assists | Implemented as proxy | Previous completed thrower before a goal on same team. |
| Box Score | T | Turnovers | Implemented | Throw attempts without a receiver. |
| Box Score | B | Blocks | Partial | Counts `defender` values on throw rows when present. Raw defensive event reconstruction may improve this. |
| Box Score | C | Completions | Implemented | Completed throws by thrower. |
| Box Score | CP% | Completion percentage | Implemented | Current version is completions / attempts. |
| Throwing | HuR | Huck rate | Implemented | Huck attempts / attempts, with configurable huck distance. |
| Throwing | HuCP% | Huck completion percentage | Implemented | Huck completions / huck attempts. |
| Throwing | HuAP% | Huck assist percentage | Needs definition | Needs exact Shown Space tooltip. |
| Throwing | xCP | Expected completion probability | Model-ready | Completion probability model output. Added by `add_expected_throwing_value` when a fitted model is supplied. |
| Throwing | CPOE | Completion percentage over expected | Model-ready | Computed when `xcp` exists. |
| Efficiency | PI-T | Player impact, throwing | Needs definition/model | Needs exact Shown Space tooltip. |
| Efficiency | PI-P | Player impact, possession | Needs definition/model | Needs exact Shown Space tooltip. |
| Efficiency | OE | Offensive efficiency | Needs possessions | Requires offensive possession reconstruction. |
| Efficiency | DE | Defensive efficiency | Needs possessions | Requires defensive possession reconstruction. |
| Usage | OPts | Offensive points played | Needs line/point context | Requires point starts and line context. |
| Usage | DPts | Defensive points played | Needs line/point context | Requires point starts and line context. |
| Usage | OPoss | Offensive possessions | Needs possessions | Requires possession reconstruction. |
| Usage | DPoss | Defensive possessions | Needs possessions | Requires possession reconstruction. |
| Usage | OI% | Offensive involvement percentage | Needs possessions | Requires possession reconstruction plus touches/throws/catches. |

## Model-Derived Metrics

The Expected Throwing Value flow has three pieces:

1. CP model: predicts completion probability for a throw.
2. FV model: predicts field value/scoring probability from a field location.
3. ETV calculation: combines CP and FV into expected throwing value.

The reference training notebook in Braden Eberhard's Expected Throwing Value
project trains:

- CP on `thrower_x`, `thrower_y`, `receiver_x`, `receiver_y`,
  `throw_distance`, `throw_angle`, `y_diff`, `x_diff`, and `times`, with very
  short throws filtered out at `throw_distance >= 1.5`.
- FV on `thrower_x`, `thrower_y`, and `times`.
- Logistic regression and XGBoost candidates through Optuna, saving the selected
  CP/FV models into an ETV wrapper.

The local `data/raw/all_games_1024.csv` file is already close to this processed
training table. It has the field, time, score, possession, and player columns.
The project derives:

- `completion = 1 - turnover`
- `point_outcome` from the scoring team within a game/score/quarter point when
  those columns are available, with a possession-level fallback for smaller
  cleaned datasets.
- `throw_angle` in degrees, matching the Expected Throwing Value processing
  notebook.

The adapter in `src/ufa/etv.py` expects fitted model dictionaries with:

```python
{
    "model": fitted_model,
    "scaler": fitted_scaler_or_none,
    "features": feature_names,
}
```

Once those are available, `add_expected_throwing_value` adds:

- `xcp`
- `fv_start`
- `fv_end`
- `fv_opponent`
- `etv`
- `cpoe`
- `ec`
- `aec`
- `t_ec`
- `r_ec`
- `total_ec`
- `t_aec`
- `r_aec`
- `total_aec`

The current EC/aEC formulas follow the paper's FV metric definitions:

```text
EC = fv_end - fv_start       if the throw is completed
EC = -fv_opponent            if the throw is a turnover

aEC = (fv_end - fv_start) / (1 - fv_possession_start)  if completed
aEC = -fv_opponent                                      if turnover
```

The project also keeps separate thrower, receiver, and total columns so the box
score layer can expose Shown Space-style `T-aEC`, `R-aEC`, and `Tot-aEC`
columns once a model is attached.

## Model Training and Evaluation

The project now has reusable helpers for Braden-style model validation:

- `split_training_data`: creates train, random validation, temporal holdout, and
  player holdout datasets.
- `train_etv_models_from_split`: trains the current logistic CP/FV baseline on
  only the train split.
- `train_xgboost_etv_models_from_split`: trains an untuned XGBoost CP/FV model.
- `train_tuned_xgboost_etv_models_from_split`: runs Optuna cross-validation for
  XGBoost model tuning.
- `evaluate_etv_model_bundle`: reports accuracy, AUC, PPV, and NPV for CP and
  FV models across each split.
- `compare_model_bundles`: compares multiple model bundles, such as logistic
  baseline versus XGBoost.

Example:

```python
from ufa import (
    compare_model_bundles,
    format_model_performance_table,
    split_training_data,
    train_etv_models_from_split,
    train_xgboost_etv_models_from_split,
)

splits = split_training_data("../data/raw/all_games_1024.csv")

baseline_bundle = train_etv_models_from_split(splits)
xgb_bundle = train_xgboost_etv_models_from_split(splits)

comparison = compare_model_bundles(
    {"logistic": baseline_bundle, "xgboost": xgb_bundle},
    splits,
)
paper_table = format_model_performance_table(comparison)
paper_table
```

The paper-style table can also be written to LaTeX:

```python
from ufa import write_model_performance_latex_document

write_model_performance_latex_document(
    paper_table,
    "../docs/model_performance_table.tex",
)
```

The table format is designed to resemble the paper's CP/FV performance table,
but the values come from the locally generated model results.

## Validation Plan

For games or players that overlap with Shown Space public tables:

1. Export or manually save the public reference stats as CSV.
2. Generate this project's stats for the same games/date range.
3. Compare matching metrics by player.
4. Use differences to identify definition mismatches or model calibration issues.

For model validation, compare CP/FV performance across the split datasets:

1. Train a logistic baseline bundle.
2. Train or tune an XGBoost bundle.
3. Evaluate both bundles on random, temporal, and player holdout sets.
4. Export the formatted results table to LaTeX for paper-style reporting.
