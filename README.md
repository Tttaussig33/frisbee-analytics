# Ultimate Frisbee Analytics

A project exploring how teams use space to move the disc and create scoring opportunities in the Ultimate Frisbee Association (UFA).

## Explore the project

- **[Open the possession browser](https://tttaussig33.github.io/frisbee-analytics/)** - Browse possessions for every team, filter by line and outcome, overlay field paths, compare teams, and organize recurring patterns.
- **[Read the research paper](paper/ufa_team_possession_patterns.pdf)** - *Deep Hucks Or Small Ball? Spatial Patterns and Possession Value Among Four 2026 Semifinalists in the Ultimate Frisbee Association.*

## What it studies

The project combines public Shown Space play-by-play data with field coordinates, possession outcomes, and throw-level adjusted expected contribution (aEC) values. The browser makes it possible to inspect individual possessions and compare the recurring spatial patterns of different teams.

The paper focuses on regular-season O-line possessions from four 2026 UFA semifinalists:

- New York Empire
- Austin Sol
- Minnesota Wind Chill
- Oakland Spiders

Goals and turnovers are included so the selected patterns can be compared using observed offensive efficiency (OOE), total aEC per possession, throw count, and field-path overlays.

## Local development

The browser is generated from the cached play-by-play data with:

```powershell
python scripts/export_possession_pattern_browser.py --all-teams --team glory --season 2026
```

The generated team pages are written to `outputs/possession_browsers/`. The GitHub Pages workflow publishes the browser at the link above.
