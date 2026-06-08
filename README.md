# Ultimate Frisbee Analytics

A small Python toolkit for pulling UFA game events, cleaning them into throw-level
data, and calculating basic player/team throwing metrics.

## Current Milestone

Given a game ID or date, the package can:

- fetch raw game events from the UFA Stats API
- clean event rows into a throws dataframe
- calculate attempts, completions, turnovers, completion percentage, and average
  throw distance
- calculate first-pass Shown Space-style summaries such as huck rate, huck
  completion percentage, receiver yardage proxies, and optional xCP/CPOE or
  expected contribution aggregates when model-output columns are available
- save raw events, cleaned throws, thrower stats, and team stats as CSVs

## Quick Start

```python
import sys

sys.path.insert(0, "src")

from ufa import build_game_throws, save_game_pipeline_outputs

result = build_game_throws(game_id="2024-06-08-LA-POR")

throws = result.throws
box_score_stats = result.box_score_stats
thrower_stats = result.thrower_stats
team_stats = result.team_stats

# Receiver summaries are also available.
from ufa import calculate_receiver_stats

receiver_stats = calculate_receiver_stats(throws)

save_game_pipeline_outputs(result, "data/processed")
```

Process the first game on a date:

```python
from ufa import build_game_throws

result = build_game_throws(date="2024-06-08", game_index=0)
```

Process every game on a date:

```python
from ufa import build_date_throws

results = build_date_throws("2024-06-08")
```

Fetch the current UFA data window from 2024 through today:

```python
from ufa import get_games_since_2024

games = get_games_since_2024()
```

Search for games before choosing one to process:

```python
from ufa import search_games

games = search_games("2024-06-01", "2024-06-30", team="breeze")
games[["gameID", "awayTeamID", "homeTeamID", "awayScore", "homeScore"]]
```

Then use one of the returned IDs:

```python
from ufa import build_game_throws

game_id = games.loc[0, "gameID"]
result = build_game_throws(game_id=game_id)
```

## Expected Throwing Value

The package has an adapter for CP/FV models trained from Braden Eberhard's
Expected Throwing Value project. Pass fitted model dictionaries with `model`,
`scaler`, and `features` keys into `ExpectedThrowingValueModel`, then pass the
model to the pipeline.

```python
from ufa import ExpectedThrowingValueModel, build_game_throws

etv_model = ExpectedThrowingValueModel(cp_model=cp_model, fv_model=fv_model)
result = build_game_throws(game_id="2024-06-08-LA-POR", etv_model=etv_model)
```

You can also train simple baseline CP/FV models from cleaned throws:

```python
from ufa import build_etv_model, train_etv_models

model_bundle = train_etv_models(throws)
etv_model = build_etv_model(model_bundle)
```

## Validation

Compare generated stats to a Shown Space-style reference CSV:

```python
from ufa import compare_metric_tables, load_reference_stats, summarize_metric_comparison

reference = load_reference_stats("data/reference/shown_space_box_score.csv")
comparison = compare_metric_tables(result.box_score_stats, reference)
summary = summarize_metric_comparison(comparison)
```

## Core Modules

- `src/ufa/client.py`: API calls for games and game events
- `src/ufa/clean.py`: raw events to throw-level rows
- `src/ufa/metrics.py`: player and team throwing summaries
- `src/ufa/pipeline.py`: end-to-end orchestration helpers
- `src/ufa/models.py`: baseline CP/FV model training and loading helpers
- `src/ufa/validation.py`: compare generated metrics to reference tables
