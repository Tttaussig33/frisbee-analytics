"""Build paper figures and tables from the saved possession arrangements.

The manuscript uses the first three groups in each hand-organized checkpoint as
the selected patterns.  The checkpoint itself can contain later playoff cards;
the analysis below keeps only regular-season game files through the final
regular-season weekend, so the paper's estimates do not silently mix seasons.

Outputs are intentionally simple SVG and text/CSV artifacts so they remain
portable and inspectable without a plotting backend such as Kaleido.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ufa.shownspace_paths import (  # noqa: E402
    ENDZONE_HIGH_Y,
    ENDZONE_LOW_Y,
    FIELD_X_MAX,
    FIELD_X_MIN,
    FIELD_Y_MAX,
    FIELD_Y_MIN,
    build_possessions,
)


FOCUS_TEAMS = ("empire", "sol", "windchill", "spiders")
FOCUS_TEAM_NAMES = {
    "empire": "New York Empire",
    "sol": "Austin Sol",
    "windchill": "Minnesota Wind Chill",
    "spiders": "Oakland Spiders",
}
PAPER_ARRANGEMENT_FILES = {
    "empire": "empire-paper.json",
    "sol": "sol-paper.json",
    "windchill": "windchill-paper.json",
    "spiders": "spiders-paper.json",
}
REGULAR_SEASON_END = date(2026, 7, 19)
HEATMAP_DISPLAY_THROWS = 20
PAPER_COLORS = {
    "navy": "#102a43",
    "muted": "#526173",
    "field": "#86d973",
    "field_bg": "#f4f8f3",
    "goal": "#155e9e",
    "turnover": "#b43f35",
    "rule": "#243b53",
    "accent": "#c98b2e",
}


def _number(value, digits=3):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _percent(value, digits=1):
    if value is None or pd.isna(value):
        return "-"
    return f"{100 * float(value):.{digits}f}%"


def _latex_text(value):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _latex_number(value, digits=3):
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _latex_percent(value, digits=1):
    if value is None or pd.isna(value):
        return "--"
    return f"{100 * float(value):.{digits}f}\\%"


def _game_date(csv_path: Path, sample: pd.DataFrame) -> date | None:
    name_date = csv_path.name[:10]
    parsed = pd.to_datetime(name_date, errors="coerce")
    if pd.notna(parsed):
        return parsed.date()
    if "start_timestamp" in sample and not sample.empty:
        parsed = pd.to_datetime(sample["start_timestamp"].iloc[0], errors="coerce")
        if pd.notna(parsed):
            return parsed.date()
    return None


def team_game_files(source_dir: Path, team_id: str, regular_season_end: date):
    candidates = []
    for csv_path in sorted(source_dir.glob("*.csv")):
        sample = pd.read_csv(csv_path, nrows=1, low_memory=False)
        if sample.empty:
            continue
        home = str(sample.get("home_team_id", pd.Series([""])).iloc[0]).lower()
        away = str(sample.get("away_team_id", pd.Series([""])).iloc[0]).lower()
        if team_id not in {home, away}:
            continue
        game_date = _game_date(csv_path, sample)
        if game_date is not None and game_date > regular_season_end:
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
    return [csv_path for _, csv_path in candidates]


def cached_team_ids(source_dir: Path, regular_season_end: date):
    team_ids = set()
    for csv_path in sorted(source_dir.glob("*.csv")):
        sample = pd.read_csv(csv_path, nrows=1, low_memory=False)
        if sample.empty:
            continue
        game_date = _game_date(csv_path, sample)
        if game_date is not None and game_date > regular_season_end:
            continue
        for column in ("home_team_id", "away_team_id"):
            value = str(sample.get(column, pd.Series([""])).iloc[0]).strip().lower()
            if value:
                team_ids.add(value)
    return sorted(team_ids)


def load_team(source_dir: Path, team_id: str, regular_season_end: date):
    files = team_game_files(source_dir, team_id, regular_season_end)
    if not files:
        raise ValueError(f"No regular-season files found for {team_id}")
    throws = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in files],
        ignore_index=True,
    )
    possessions, paths = build_possessions(
        throws,
        team_id=team_id,
        outcomes=("goal", "turnover"),
    )
    return files, possessions, paths


def _path_lookup(paths):
    return {
        str(path["possession_id"].iloc[0]): path
        for path in paths
        if not path.empty and "possession_id" in path
    }


def _group_stats(group, possession_by_id, total_o_line):
    ids = [str(card.get("possession_id", "")) for card in group.get("possessions", [])]
    ids = [possession_id for possession_id in ids if possession_id in possession_by_id.index]
    frame = possession_by_id.reindex(ids).dropna(subset=["possession_id"])
    frame = frame.loc[frame["line_type"].eq("o_line")].copy()
    if frame.empty:
        return {
            "group_index": int(group.get("group_index", 0)),
            "title": str(group.get("title") or "Pattern"),
            "ids": [],
            "n": 0,
            "share": np.nan,
            "goals": 0,
            "turnovers": 0,
            "goal_ending_share": np.nan,
            "total_aec_per_possession": np.nan,
            "median_throws": np.nan,
            "huck_rate": np.nan,
            "median_start_y": np.nan,
            "median_field_progress": np.nan,
        }

    goals = int(frame["outcome"].eq("goal").sum())
    turnovers = int(frame["outcome"].eq("turnover").sum())
    return {
        "group_index": int(group.get("group_index", 0)),
        "title": str(group.get("title") or "Pattern"),
        "ids": frame["possession_id"].astype(str).tolist(),
        "n": int(len(frame)),
        "share": float(len(frame) / total_o_line) if total_o_line else np.nan,
        "goals": goals,
        "turnovers": turnovers,
        "goal_ending_share": float(goals / len(frame)) if len(frame) else np.nan,
        "total_aec_per_possession": float(frame["total_aec"].mean()),
        "median_throws": float(frame["throw_count"].median()),
        "huck_rate": float(frame["huck_count"].gt(0).mean()),
        "median_start_y": float(frame["start_y"].median()),
        "median_field_progress": float(frame["field_progress"].median()),
    }


def _team_summary(team_id, files, possessions):
    frame = possessions.loc[possessions["line_type"].eq("o_line")].copy()
    goals = int(frame["outcome"].eq("goal").sum())
    turnovers = int(frame["outcome"].eq("turnover").sum())
    return {
        "team_id": team_id,
        "team_name": FOCUS_TEAM_NAMES.get(team_id, team_id.title()),
        "regular_season_games": len(files),
        "o_line_possessions": int(len(frame)),
        "goals": goals,
        "turnovers": turnovers,
        "goal_ending_share": float(goals / len(frame)) if len(frame) else np.nan,
        "total_aec_per_possession": float(frame["total_aec"].mean()) if len(frame) else np.nan,
        "median_throws": float(frame["throw_count"].median()) if len(frame) else np.nan,
    }


def collect_data(source_dir: Path, arrangement_dir: Path, regular_season_end: date):
    team_ids = cached_team_ids(source_dir, regular_season_end)
    loaded = {}
    for team_id in team_ids:
        files, possessions, paths = load_team(source_dir, team_id, regular_season_end)
        loaded[team_id] = {
            "files": files,
            "possessions": possessions,
            "paths": paths,
            "path_by_id": _path_lookup(paths),
        }

    team_summaries = []
    pattern_rows = []
    focus_patterns = {}
    for team_id in FOCUS_TEAMS:
        if team_id not in loaded:
            raise ValueError(f"Missing loaded data for focus team {team_id}")
        data = loaded[team_id]
        possessions = data["possessions"]
        o_line_count = int(possessions["line_type"].eq("o_line").sum())
        team_summaries.append(_team_summary(team_id, data["files"], possessions))
        payload_path = arrangement_dir / PAPER_ARRANGEMENT_FILES[team_id]
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        possession_by_id = possessions.set_index("possession_id", drop=False)
        groups = payload.get("groups", [])[:3]
        grouped_stats = [
            (group, _group_stats(group, possession_by_id, o_line_count))
            for group in groups
        ]
        if team_id == "windchill":
            grouped_stats.sort(
                key=lambda item: (
                    -item[1]["n"],
                    item[1]["group_index"],
                )
            )
        stats = []
        for pattern_number, (group, row) in enumerate(grouped_stats, start=1):
            stats.append(row)
            row_for_table = {
                "team_id": team_id,
                "team_name": FOCUS_TEAM_NAMES[team_id],
                "pattern": pattern_number,
                **{key: value for key, value in row.items() if key not in {"ids"}},
            }
            pattern_rows.append(row_for_table)
        focus_patterns[team_id] = {
            "team_id": team_id,
            "team_name": FOCUS_TEAM_NAMES[team_id],
            "arrangement_saved_at": payload.get("saved_at"),
            "arrangement_path": str(payload_path.relative_to(REPO_ROOT)),
            "patterns": stats,
        }

    heatmap_rows = []
    for team_id, data in loaded.items():
        goals = data["possessions"].loc[
            data["possessions"]["line_type"].eq("o_line")
            & data["possessions"]["outcome"].eq("goal")
        ].copy()
        counts = goals["throw_count"].value_counts().to_dict()
        total = int(len(goals))
        mode = int(goals["throw_count"].mode().iloc[0]) if total else None
        heatmap_rows.append(
            {
                "team_id": team_id,
                "team_name": FOCUS_TEAM_NAMES.get(team_id, team_id.title()),
                "o_line_goal_possessions": total,
                "mode_throws": mode,
                "counts": {str(int(key)): int(value) for key, value in counts.items()},
            }
        )
    heatmap_rows.sort(key=lambda row: (row["mode_throws"] is None, row["mode_throws"] or 0, row["team_name"]))

    league_counts = {}
    for row in heatmap_rows:
        for throw_count, count in row["counts"].items():
            league_counts[int(throw_count)] = league_counts.get(int(throw_count), 0) + count
    league_mode = max(league_counts, key=lambda key: (league_counts[key], -key)) if league_counts else None

    observed_dates = []
    for data in loaded.values():
        for path in data["files"]:
            sample = pd.read_csv(path, nrows=1, low_memory=False)
            observed = _game_date(path, sample)
            if observed is not None:
                observed_dates.append(observed)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regular_season_end": regular_season_end.isoformat(),
        "regular_season_start": min(observed_dates).isoformat(),
        "team_ids": team_ids,
        "focus_teams": list(FOCUS_TEAMS),
        "team_summaries": team_summaries,
        "pattern_rows": pattern_rows,
        "focus_patterns": focus_patterns,
        "heatmap_rows": heatmap_rows,
        "league_counts": {str(key): value for key, value in sorted(league_counts.items())},
        "league_mode_throws": league_mode,
        "league_goal_possessions": int(sum(league_counts.values())),
        "loaded": loaded,
    }


def _svg_text(x, y, text, *, size=14, color=None, weight="400", anchor="start", family="Arial, sans-serif"):
    color = color or PAPER_COLORS["navy"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}px" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(text))}</text>'
    )


def _svg_field(x, y, width, height, paths, stats):
    parts = []
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="8" fill="{PAPER_COLORS["field_bg"]}" stroke="#d7e2eb" stroke-width="1"/>'
    )
    margin_y = 12
    field_y = y + margin_y
    field_h = height - 90
    field_w = field_h * (FIELD_X_MAX - FIELD_X_MIN) / (FIELD_Y_MAX - FIELD_Y_MIN)
    field_x = x + (width - field_w) / 2

    def sx(value):
        return field_x + (float(value) - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN) * field_w

    def sy(value):
        return field_y + (FIELD_Y_MAX - float(value)) / (FIELD_Y_MAX - FIELD_Y_MIN) * field_h

    parts.append(
        f'<rect x="{field_x:.1f}" y="{field_y:.1f}" width="{field_w:.1f}" height="{field_h:.1f}" '
        f'fill="{PAPER_COLORS["field"]}" stroke="{PAPER_COLORS["rule"]}" stroke-width="1.8"/>'
    )
    for y_value in (ENDZONE_LOW_Y, ENDZONE_HIGH_Y):
        parts.append(
            f'<line x1="{field_x:.1f}" y1="{sy(y_value):.1f}" x2="{field_x + field_w:.1f}" '
            f'y2="{sy(y_value):.1f}" stroke="{PAPER_COLORS["rule"]}" stroke-width="1.2"/>'
        )
    for y_value in (40, 80):
        parts.append(
            f'<circle cx="{sx(0):.1f}" cy="{sy(y_value):.1f}" r="2.0" fill="{PAPER_COLORS["rule"]}"/>'
        )

    opacity = 0.18 if stats["n"] <= 25 else 0.10
    stroke_width = 1.5 if stats["n"] <= 25 else 1.05
    for path in paths:
        if path.empty:
            continue
        outcome = str(path.get("outcome", pd.Series(["unknown"])).iloc[0]).lower()
        color = PAPER_COLORS["goal"] if outcome == "goal" else PAPER_COLORS["turnover"]
        path = path.sort_values("possession_throw")
        points = []
        for _, throw in path.iterrows():
            try:
                start = (sx(throw["ThrowerX"]), sy(throw["ThrowerY"]))
                end = (sx(throw["ReceiverX"]), sy(throw["ReceiverY"]))
            except (TypeError, ValueError):
                continue
            points.extend([start, end])
            parts.append(
                f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.1f}" stroke-linecap="round" '
                f'opacity="{opacity:.2f}"/>'
            )
        for px, py in points[::2]:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.7" fill="{color}" opacity="{min(0.8, opacity + 0.28):.2f}"/>'
            )

    metric_line_1 = f"n={stats['n']} | G={stats['goals']} | TOs={stats['turnovers']}"
    metric_line_2 = (
        f"OOE {_percent(stats['goal_ending_share'])} | "
        f"aEC/pos {_number(stats['total_aec_per_possession'])}"
    )
    metric_line_3 = (
        f"Median {_number(stats['median_throws'], 1)} throws | "
        f"Hucks {_percent(stats['huck_rate'])}"
    )
    metric_lines = [metric_line_1, metric_line_2, metric_line_3]
    metric_gap = 22
    metric_last_y = y + height - 14
    metric_first_y = metric_last_y - metric_gap * (len(metric_lines) - 1)
    for index, line in enumerate(metric_lines):
        parts.append(
            _svg_text(
                x + width / 2,
                metric_first_y + index * metric_gap,
                line,
                size=17 if index == 0 else 16,
                color=PAPER_COLORS["navy"],
                weight="700" if index == 0 else "400",
                anchor="middle",
            )
        )
    return "".join(parts)


def write_team_pattern_svg(team_id, data, out_path: Path):
    patterns = data["patterns"]
    path_by_id = data["path_by_id"]
    width = 1120
    height = 650
    panel_width = 340
    panel_height = 510
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        _svg_text(26, 32, f"{FOCUS_TEAM_NAMES[team_id]}: three common O-line patterns", size=21, weight="700"),
        _svg_text(26, 55, "Regular season; overlays show every O-line goal and turnover in the selected row.", size=12, color=PAPER_COLORS["muted"]),
        f'<line x1="26" y1="72" x2="1094" y2="72" stroke="#d7e2eb" stroke-width="1"/>',
    ]
    for index, pattern in enumerate(patterns):
        paths = [path_by_id[possession_id] for possession_id in pattern["ids"] if possession_id in path_by_id]
        pieces.append(
            _svg_field(
                16 + index * panel_width,
                84,
                panel_width - 18,
                panel_height,
                paths,
                pattern,
            )
        )
    legend_y = 632
    pieces.append(f'<line x1="26" y1="{legend_y - 5}" x2="50" y2="{legend_y - 5}" stroke="{PAPER_COLORS["goal"]}" stroke-width="2.5"/>')
    pieces.append(_svg_text(57, legend_y, "goal", size=11, color=PAPER_COLORS["muted"]))
    pieces.append(f'<line x1="105" y1="{legend_y - 5}" x2="129" y2="{legend_y - 5}" stroke="{PAPER_COLORS["turnover"]}" stroke-width="2.5"/>')
    pieces.append(_svg_text(136, legend_y, "turnover", size=11, color=PAPER_COLORS["muted"]))
    pieces.append(_svg_text(1092, legend_y, "O-line only; paths are drawn in actual field coordinates.", size=11, color=PAPER_COLORS["muted"], anchor="end"))
    pieces.append("</svg>")
    out_path.write_text("".join(pieces), encoding="utf-8")


def _heat_color(value, maximum):
    if maximum <= 0:
        fraction = 0.0
    else:
        fraction = max(0.0, min(1.0, float(value) / maximum))
    light = (244, 250, 242)
    dark = (8, 91, 48)
    channels = [round(light[i] + fraction * (dark[i] - light[i])) for i in range(3)]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def write_heatmap_svg(heatmap_rows, league_mode, out_path: Path):
    cell_w = 30
    cell_h = 32
    left = 230
    top = 124
    right = 260
    bottom = 108
    columns = list(range(1, HEATMAP_DISPLAY_THROWS + 1)) + [f">{HEATMAP_DISPLAY_THROWS}"]
    height = top + len(heatmap_rows) * cell_h + bottom
    numeric_columns = len(columns)
    width = left + numeric_columns * cell_w + right
    maximum = max(
        [
            count / row["o_line_goal_possessions"]
            for row in heatmap_rows
            for throw_count, count in row["counts"].items()
            if row["o_line_goal_possessions"]
        ]
        or [0.0]
    )
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        _svg_text(28, 34, "Regular-season O-line goal throw-count distribution", size=24, weight="700"),
        _svg_text(28, 61, f"Each row is normalized within team; league-wide mode = {league_mode} throws.", size=15, color=PAPER_COLORS["muted"]),
        _svg_text(28, 86, f"Dark outline marks each team's mode; >{HEATMAP_DISPLAY_THROWS} combines longer possessions.", size=13, color=PAPER_COLORS["muted"]),
        _svg_text(left - 16, top - 18, "Team", size=14, weight="700", anchor="end"),
        _svg_text(left + numeric_columns * cell_w + 32, top - 18, "n goals", size=14, weight="700"),
    ]
    for row_index, row in enumerate(heatmap_rows):
        y = top + row_index * cell_h
        name = row["team_name"].replace("New York ", "").replace("Austin ", "").replace("Minnesota ", "").replace("Oakland ", "")
        label = f"{name} (mode {row['mode_throws']})"
        pieces.append(_svg_text(left - 16, y + 21, label, size=14, weight="600", anchor="end"))
        pieces.append(_svg_text(left + numeric_columns * cell_w + 32, y + 21, f"n={row['o_line_goal_possessions']}", size=14, color=PAPER_COLORS["muted"], weight="600"))
        mode_throws = row["mode_throws"]
        mode_column = None
        if mode_throws is not None and not pd.isna(mode_throws):
            mode_column = mode_throws if mode_throws <= HEATMAP_DISPLAY_THROWS else f">{HEATMAP_DISPLAY_THROWS}"
        for column in columns:
            if isinstance(column, int):
                count = row["counts"].get(str(column), 0)
            else:
                count = sum(
                    value for throw_count, value in row["counts"].items()
                    if int(throw_count) > HEATMAP_DISPLAY_THROWS
                )
            share = count / row["o_line_goal_possessions"] if row["o_line_goal_possessions"] else 0.0
            stroke = PAPER_COLORS["rule"] if column == mode_column else "#ffffff"
            stroke_width = 2.0 if column == mode_column else 0.7
            pieces.append(
                f'<rect x="{left + columns.index(column) * cell_w}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{_heat_color(share, maximum)}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )
    x_label_y = top + len(heatmap_rows) * cell_h + 25
    for column in columns:
        pieces.append(_svg_text(left + columns.index(column) * cell_w + cell_w / 2, x_label_y, str(column), size=12, color=PAPER_COLORS["muted"], anchor="middle"))
    pieces.append(_svg_text(left + numeric_columns * cell_w / 2, x_label_y + 25, "Number of throws in possession", size=13, color=PAPER_COLORS["muted"], anchor="middle"))
    legend_x = left + numeric_columns * cell_w + 145
    legend_y = top + 25
    pieces.append(_svg_text(legend_x, legend_y - 12, "Share of goals", size=13, weight="700", anchor="middle"))
    for index in range(6):
        fraction = index / 5
        y = legend_y + index * 31
        pieces.append(f'<rect x="{legend_x - 15}" y="{y}" width="30" height="31" fill="{_heat_color(maximum * (1 - fraction), maximum)}"/>')
        pieces.append(_svg_text(legend_x + 22, y + 20, f"{maximum * (1 - fraction):.2f}", size=12, color=PAPER_COLORS["muted"]))
    pieces.append(_svg_text(28, height - 20, "Rows are regular-season O-line goals; postseason game files are excluded.", size=13, color=PAPER_COLORS["muted"]))
    pieces.append("</svg>")
    out_path.write_text("".join(pieces), encoding="utf-8")


def write_tables(results, output_dir: Path):
    pattern_frame = pd.DataFrame(results["pattern_rows"])
    pattern_frame = (
        pattern_frame.sort_values(
            ["goal_ending_share", "n", "team_name", "pattern"],
            ascending=[False, False, True, True],
            na_position="last",
            kind="stable",
        )
        .reset_index(drop=True)
    )
    pattern_frame.insert(0, "rank", range(1, len(pattern_frame) + 1))
    team_frame = pd.DataFrame(results["team_summaries"])
    pattern_frame.to_csv(output_dir / "pattern_metrics.csv", index=False)
    team_frame.to_csv(output_dir / "team_metrics.csv", index=False)

    team_rows = []
    for row in results["team_summaries"]:
        team_rows.append(
            " & ".join(
                [
                    _latex_text(row["team_name"]),
                    str(row["regular_season_games"]),
                    str(row["o_line_possessions"]),
                    str(row["goals"]),
                    str(row["turnovers"]),
                    _latex_percent(row["goal_ending_share"]),
                    _latex_number(row["total_aec_per_possession"]),
                ]
            )
            + r" \\"
        )
    team_rows.append(r"\bottomrule")
    (output_dir / "generated_team_table.tex").write_text(
        "\n".join(team_rows) + "\n", encoding="utf-8"
    )

    pattern_rows = []
    for row in pattern_frame.to_dict("records"):
        pattern_rows.append(
            " & ".join(
                [
                    str(row["rank"]),
                    _latex_text(row["team_name"]),
                    str(row["pattern"]),
                    str(row["n"]),
                    _latex_percent(row["share"]),
                    str(row["goals"]),
                    str(row["turnovers"]),
                    _latex_percent(row["goal_ending_share"]),
                    _latex_number(row["total_aec_per_possession"]),
                    _latex_number(row["median_throws"], 1),
                ]
            )
            + r" \\"
        )
    pattern_rows.append(r"\bottomrule")
    (output_dir / "generated_pattern_table.tex").write_text(
        "\n".join(pattern_rows) + "\n", encoding="utf-8"
    )

    heatmap_rows = []
    for row in results["heatmap_rows"]:
        heatmap_rows.append(
            " & ".join(
                [
                    _latex_text(row["team_name"]),
                    str(row["o_line_goal_possessions"]),
                    str(row["mode_throws"]),
                ]
            )
            + r" \\"
        )
    (output_dir / "generated_heatmap_table.tex").write_text(
        "\n".join(heatmap_rows) + "\n", encoding="utf-8"
    )


def _json_safe(results):
    safe = {key: value for key, value in results.items() if key != "loaded"}
    return safe


def main():
    parser = argparse.ArgumentParser(description="Build LaTeX-ready UFA team possession pattern figures.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--regular-season-end", type=date.fromisoformat, default=REGULAR_SEASON_END)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--arrangement-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    source_dir = args.source_dir or (REPO_ROOT / "data" / "raw" / f"shownspace_throws_{args.season}_by_game")
    arrangement_dir = args.arrangement_dir or (REPO_ROOT / "data" / "arrangements" / str(args.season))
    output_dir = args.output_dir or (REPO_ROOT / "paper" / "generated")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = collect_data(source_dir, arrangement_dir, args.regular_season_end)
    (output_dir / "metrics.json").write_text(json.dumps(_json_safe(results), indent=2) + "\n", encoding="utf-8")

    for team_id in FOCUS_TEAMS:
        write_team_pattern_svg(team_id, results["focus_patterns"][team_id] | {"path_by_id": results["loaded"][team_id]["path_by_id"]}, figure_dir / f"{team_id}_top3_patterns.svg")
    write_heatmap_svg(results["heatmap_rows"], results["league_mode_throws"], figure_dir / "league_throw_count_heatmap.svg")
    write_tables(results, output_dir)

    print(f"Regular-season window: {results['regular_season_start']} through {results['regular_season_end']}")
    print(f"Teams in heatmap: {len(results['team_ids'])}; league O-line goals: {results['league_goal_possessions']:,}; mode: {results['league_mode_throws']} throws")
    for row in results["team_summaries"]:
        print(f"{row['team_name']}: {row['regular_season_games']} games, {row['o_line_possessions']} O-line possessions, {row['goals']} goals, {row['turnovers']} turnovers")
    print(f"Wrote paper data and figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
