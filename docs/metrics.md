# Shown Space-Style Metrics Roadmap

This project uses Shown Space and the Expected Throwing Value work as a model
for advanced ultimate frisbee analytics. The goal is not to copy private outputs
from another site, but to build reproducible metrics from UFA event data and use
public Shown Space tables as validation references.

## Implementation Status

| Group | Metric | Meaning | Status | Notes |
| --- | --- | --- | --- | --- |
| Contribution | Tot-aEC | Total adjusted expected contribution | Needs model | Sum of throwing and receiving adjusted expected contribution. |
| Contribution | T-aEC | Throwing adjusted expected contribution | Needs model | Requires CP/FV/ETV outputs and adjustment layer. |
| Contribution | R-aEC | Receiving adjusted expected contribution | Needs model | Requires CP/FV/ETV outputs and receiver allocation. |
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
| Throwing | xCP | Expected completion probability | Needs model | Completion probability model output. |
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
- `t_ec`
- `r_ec`
- `total_ec`

## Validation Plan

For games or players that overlap with Shown Space public tables:

1. Export or manually save the public reference stats as CSV.
2. Generate this project's stats for the same games/date range.
3. Compare matching metrics by player.
4. Use differences to identify definition mismatches or model calibration issues.
