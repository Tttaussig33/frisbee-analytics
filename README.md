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

## Core Modules

- `src/ufa/client.py`: API calls for games and game events
- `src/ufa/clean.py`: raw events to throw-level rows
- `src/ufa/metrics.py`: player and team throwing summaries
- `src/ufa/pipeline.py`: end-to-end orchestration helpers
