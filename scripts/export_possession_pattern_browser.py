import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ufa.shownspace_paths import (  # noqa: E402
    build_possessions,
    write_possession_pattern_browser_html,
)


def _team_game_files(source_dir, team_id, regular_season_games):
    candidates = []
    for csv_path in sorted(source_dir.glob("*.csv")):
        sample = pd.read_csv(csv_path, nrows=1, low_memory=False)
        if sample.empty:
            continue
        home = str(sample.get("home_team_id", pd.Series([""])).iloc[0]).lower()
        away = str(sample.get("away_team_id", pd.Series([""])).iloc[0]).lower()
        if team_id not in {home, away}:
            continue
        start = pd.to_datetime(
            sample.get("start_timestamp", pd.Series([pd.NaT])).iloc[0],
            errors="coerce",
        )
        candidates.append((start, csv_path))

    candidates.sort(
        key=lambda item: (
            pd.Timestamp.max if pd.isna(item[0]) else item[0],
            item[1].name,
        )
    )
    if regular_season_games > 0:
        candidates = candidates[:regular_season_games]
    return [csv_path for _, csv_path in candidates]


def _cached_team_ids(source_dir):
    team_ids = set()
    for csv_path in sorted(source_dir.glob("*.csv")):
        sample = pd.read_csv(csv_path, nrows=1, low_memory=False)
        if sample.empty:
            continue
        for column in ("home_team_id", "away_team_id"):
            value = str(sample.get(column, pd.Series([""])).iloc[0]).strip().lower()
            if value:
                team_ids.add(value)
    return sorted(team_ids)


def _team_options(team_ids, season):
    return [
        {
            "id": team_id,
            "label": team_id.title(),
            "href": f"{season}-{team_id}.html",
        }
        for team_id in team_ids
    ]


def _write_index(output_dir, season, default_team, team_ids):
    default_page = f"{season}-{default_team}.html"
    links = "\n".join(
        f'<li><a href="{season}-{team_id}.html">{team_id.title()}</a></li>'
        for team_id in team_ids
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="0; url={default_page}" />
  <title>{season} UFA possession pattern browsers</title>
</head>
<body>
  <h1>{season} UFA possession pattern browsers</h1>
  <p><a href="{default_page}">Open {default_team.title()}</a></p>
  <ul>{links}</ul>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Export a standalone UFA possession pattern browser."
    )
    parser.add_argument("--team", default="glory", help="Shown Space team ID")
    parser.add_argument(
        "--all-teams",
        action="store_true",
        help="Export linked browser pages for every cached team",
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing one Shown Space throws CSV per game",
    )
    parser.add_argument(
        "--regular-season-games",
        type=int,
        default=0,
        help="Use the team's first N cached games; set 0 to include every cached game",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=["goal", "turnover"],
        choices=["goal", "turnover"],
    )
    parser.add_argument("--max-cards", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for linked team pages (used with --all-teams)",
    )
    args = parser.parse_args()

    team_id = args.team.strip().lower()
    source_dir = args.source_dir or (
        REPO_ROOT / "data" / "raw" / f"shownspace_throws_{args.season}_by_game"
    )
    cached_team_ids = _cached_team_ids(source_dir)
    if team_id not in cached_team_ids:
        raise SystemExit(f"No cached {team_id} games found in {source_dir}")

    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "possession_browsers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    teams_to_export = cached_team_ids if args.all_teams else [team_id]
    navigation_options = _team_options(cached_team_ids, args.season)

    if args.all_teams and args.output is not None:
        raise SystemExit("--output cannot be combined with --all-teams; use --output-dir")

    for export_team_id in teams_to_export:
        game_files = _team_game_files(
            source_dir,
            export_team_id,
            args.regular_season_games,
        )
        if not game_files:
            print(f"Skipped {export_team_id}: no cached games")
            continue

        throws = pd.concat(
            [pd.read_csv(csv_path, low_memory=False) for csv_path in game_files],
            ignore_index=True,
        )
        possessions, paths = build_possessions(
            throws,
            team_id=export_team_id,
            outcomes=tuple(args.outcomes),
        )
        project_arrangement_text = None
        arrangement_path = (
            REPO_ROOT
            / "data"
            / "arrangements"
            / str(args.season)
            / f"{export_team_id}.json"
        )
        if arrangement_path.exists():
            try:
                project_arrangement_text = arrangement_path.read_text(encoding="utf-8")
                json.loads(project_arrangement_text)
            except (OSError, json.JSONDecodeError) as error:
                print(
                    f"Warning: skipped invalid project arrangement "
                    f"{arrangement_path}: {error}"
                )
                project_arrangement_text = None
        output_path = (
            args.output
            if not args.all_teams and args.output is not None
            else output_dir / f"{args.season}-{export_team_id}.html"
        )
        output_path = write_possession_pattern_browser_html(
            possessions,
            paths,
            output_path=output_path,
            title=f"{export_team_id.title()} {args.season} possession pattern browser",
            max_cards=args.max_cards,
            team_id=export_team_id,
            team_options=navigation_options,
            persistence_key=f"{args.season}:{export_team_id}",
            project_arrangement_text=project_arrangement_text,
        )
        print(
            f"{export_team_id.title()}: {len(game_files):,} games, "
            f"{len(possessions):,} possessions -> {output_path.resolve()}"
        )

    if args.all_teams:
        index_path = _write_index(
            output_dir,
            args.season,
            team_id,
            cached_team_ids,
        )
        print(f"Index: {index_path.resolve()}")


if __name__ == "__main__":
    main()
