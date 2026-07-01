import json
import re
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from ufa.client import get_games_for_date_range
from ufa.shownspace_paths import (
    ENDZONE_HIGH_Y,
    ENDZONE_LOW_Y,
    FIELD_X_MAX,
    FIELD_X_MIN,
    FIELD_Y_MAX,
    FIELD_Y_MIN,
    build_scoring_possessions,
    fetch_shownspace_throws_for_games,
)


ANCHOR_COLUMNS = [
    "game_id",
    "youtube_url",
    "game_quarter",
    "quarter_point",
    "video_seconds",
    "note",
]
POSSESSION_ANCHOR_COLUMNS = [
    "game_id",
    "game_quarter",
    "quarter_point",
    "possession_num",
    "is_home_team",
    "team_id",
    "video_seconds",
    "note",
]
CLOCK_ANCHOR_COLUMNS = [
    "game_id",
    "source_timestamp",
    "video_seconds",
    "note",
]
FNF_GAME_COLUMNS = [
    "season",
    "week",
    "game_id",
    "youtube_url",
    "source",
    "note",
]


def is_youtube_url(url):
    """Return True when a URL points to YouTube."""
    if not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().replace("www.", "")
    return host == "youtu.be" or host.endswith("youtube.com")


def extract_youtube_video_id(url):
    """Extract a YouTube video id from watch, short, embed, or shorts URLs."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty YouTube URL string.")

    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.strip("/")

    if host in {"youtu.be"} and path:
        return path.split("/")[0]

    if host.endswith("youtube.com"):
        query_video_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_video_id:
            return query_video_id

        parts = path.split("/")
        for marker in ("embed", "shorts", "live"):
            if marker in parts:
                index = parts.index(marker)
                if index + 1 < len(parts):
                    return parts[index + 1]

    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([A-Za-z0-9_-]{6,})", url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract a YouTube video id from: {url}")


def find_friday_night_frisbee_games(
    start_date,
    end_date=None,
    team_id=None,
    youtube_only=True,
):
    """Find Friday games from UFA metadata, prioritizing free YouTube streams."""
    games = get_games_for_date_range(start_date, end_date=end_date)
    if games.empty:
        return games

    games = games.copy()
    start_timestamp = games.get("startTimestamp", pd.Series("", index=games.index))
    local_start_date = start_timestamp.astype(str).str.slice(0, 10)
    games["start_datetime"] = pd.to_datetime(local_start_date, errors="coerce")
    games["is_friday"] = games["start_datetime"].dt.dayofweek.eq(4)
    streaming_url = games.get("streamingURL", pd.Series("", index=games.index)).fillna("")
    games["is_youtube_stream"] = streaming_url.apply(is_youtube_url)

    filtered = games[games["is_friday"]].copy()
    if youtube_only:
        filtered = filtered[filtered["is_youtube_stream"]].copy()

    if team_id is not None and not filtered.empty:
        team_id = team_id.lower()
        filtered = filtered[
            filtered["awayTeamID"].str.lower().eq(team_id)
            | filtered["homeTeamID"].str.lower().eq(team_id)
        ].copy()

    columns = [
        "gameID",
        "awayTeamID",
        "homeTeamID",
        "awayScore",
        "homeScore",
        "status",
        "startTimestamp",
        "week",
        "location",
        "streamingURL",
        "is_youtube_stream",
    ]
    return (
        filtered[[column for column in columns if column in filtered.columns]]
        .sort_values("startTimestamp")
        .reset_index(drop=True)
    )


def find_youtube_url_for_game(game_id, start_date, end_date=None):
    """Return the YouTube stream URL for a game when UFA metadata exposes one."""
    games = get_games_for_date_range(start_date, end_date=end_date)
    if games.empty or "gameID" not in games:
        return None

    matched = games[games["gameID"].astype(str).eq(str(game_id))]
    if matched.empty or "streamingURL" not in matched:
        return None

    url = matched["streamingURL"].dropna().astype(str).iloc[0]
    return url if is_youtube_url(url) else None


def load_fnf_game_schedule(path, season=None):
    """Load the manually verified Friday Night Frisbee YouTube schedule."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=FNF_GAME_COLUMNS)

    schedule = pd.read_csv(path)
    for column in FNF_GAME_COLUMNS:
        if column not in schedule:
            schedule[column] = np.nan

    schedule = schedule[FNF_GAME_COLUMNS].copy()
    if season is not None and not schedule.empty:
        schedule = schedule[
            pd.to_numeric(schedule["season"], errors="coerce").eq(int(season))
        ].copy()
    return schedule.reset_index(drop=True)


def build_fnf_game_table(
    start_date,
    end_date=None,
    schedule_path=None,
    season=None,
    team_id=None,
):
    """Combine Friday UFA games with a manually verified YouTube FNF schedule."""
    friday_games = find_friday_night_frisbee_games(
        start_date,
        end_date=end_date,
        team_id=team_id,
        youtube_only=False,
    )
    if friday_games.empty:
        return friday_games

    friday_games = friday_games.copy()
    streaming_url = friday_games.get("streamingURL", pd.Series("", index=friday_games.index))
    friday_games["youtube_url"] = streaming_url.where(streaming_url.apply(is_youtube_url), "")
    friday_games["fnf_source"] = np.where(friday_games["youtube_url"].ne(""), "streamingURL", "")
    friday_games["is_fnf_youtube"] = friday_games["youtube_url"].ne("")

    if schedule_path is not None:
        schedule = load_fnf_game_schedule(schedule_path, season=season)
        if not schedule.empty:
            schedule = schedule.rename(columns={"game_id": "gameID"})
            friday_games = friday_games.merge(
                schedule[["gameID", "youtube_url", "source", "note"]],
                on="gameID",
                how="left",
                suffixes=("", "_manual"),
            )
            manual_url = friday_games["youtube_url_manual"].fillna("")
            friday_games["youtube_url"] = np.where(
                manual_url.ne(""),
                manual_url,
                friday_games["youtube_url"],
            )
            friday_games["fnf_source"] = np.where(
                manual_url.ne(""),
                friday_games["source"].fillna("manual"),
                friday_games["fnf_source"],
            )
            friday_games["is_fnf_youtube"] = friday_games["youtube_url"].fillna("").ne("")
            friday_games = friday_games.drop(
                columns=[
                    column
                    for column in ["youtube_url_manual", "source"]
                    if column in friday_games
                ]
            )

    preferred_columns = [
        "gameID",
        "awayTeamID",
        "homeTeamID",
        "startTimestamp",
        "week",
        "status",
        "location",
        "youtube_url",
        "is_fnf_youtube",
        "fnf_source",
        "streamingURL",
    ]
    return friday_games[
        [column for column in preferred_columns if column in friday_games.columns]
    ].reset_index(drop=True)


def load_fnf_point_anchors(path, game_id):
    """Load manual point-start video anchors for one game."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ANCHOR_COLUMNS)

    anchors = pd.read_csv(path)
    for column in ANCHOR_COLUMNS:
        if column not in anchors:
            anchors[column] = np.nan

    anchors = anchors[ANCHOR_COLUMNS].copy()
    anchors = anchors[anchors["game_id"].astype(str).eq(str(game_id))].reset_index(drop=True)
    anchors["game_quarter"] = pd.to_numeric(anchors["game_quarter"], errors="coerce")
    anchors["quarter_point"] = pd.to_numeric(anchors["quarter_point"], errors="coerce")
    anchors["video_seconds"] = pd.to_numeric(anchors["video_seconds"], errors="coerce")
    return anchors


def _coerce_anchor_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "home"}:
        return True
    if text in {"false", "0", "no", "n", "away"}:
        return False
    return np.nan


def load_fnf_possession_anchors(path, game_id):
    """Load manual possession-start video anchors for one game."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=POSSESSION_ANCHOR_COLUMNS)

    anchors = pd.read_csv(path)
    for column in POSSESSION_ANCHOR_COLUMNS:
        if column not in anchors:
            anchors[column] = np.nan

    anchors = anchors[POSSESSION_ANCHOR_COLUMNS].copy()
    anchors = anchors[anchors["game_id"].astype(str).eq(str(game_id))].reset_index(drop=True)
    for column in ["game_quarter", "quarter_point", "possession_num", "video_seconds"]:
        anchors[column] = pd.to_numeric(anchors[column], errors="coerce")
    anchors["is_home_team"] = anchors["is_home_team"].apply(_coerce_anchor_bool)
    anchors["team_id"] = anchors["team_id"].fillna("").astype(str).str.lower()
    return anchors


def load_fnf_clock_anchors(path, game_id):
    """Load game-level source-clock to YouTube-clock calibration anchors."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CLOCK_ANCHOR_COLUMNS)

    anchors = pd.read_csv(path)
    for column in CLOCK_ANCHOR_COLUMNS:
        if column not in anchors:
            anchors[column] = np.nan

    anchors = anchors[CLOCK_ANCHOR_COLUMNS].copy()
    anchors = anchors[anchors["game_id"].astype(str).eq(str(game_id))].reset_index(drop=True)
    anchors["source_timestamp"] = pd.to_numeric(
        anchors["source_timestamp"],
        errors="coerce",
    )
    anchors["video_seconds"] = pd.to_numeric(anchors["video_seconds"], errors="coerce")
    return anchors


def build_fnf_browser_data(game_id, team_id=None):
    """Fetch Shown Space throws and build scoring possessions for one game."""
    throws = fetch_shownspace_throws_for_games([game_id], delay=0)
    possessions, paths = build_scoring_possessions(throws, team_id=team_id)
    return possessions, paths


def add_estimated_video_seconds(paths, anchors, seconds_per_throw=3.0):
    """Attach estimated YouTube timestamps to each throw from point-start anchors."""
    if anchors is None or anchors.empty:
        anchor_lookup = {}
    else:
        clean_anchors = anchors.dropna(
            subset=["game_quarter", "quarter_point", "video_seconds"]
        ).copy()
        anchor_lookup = {
            (int(row.game_quarter), int(row.quarter_point)): float(row.video_seconds)
            for row in clean_anchors.itertuples(index=False)
        }

    estimated_paths = []
    for path in paths:
        estimated = path.sort_values("possession_throw").copy()
        if estimated.empty:
            estimated_paths.append(estimated)
            continue

        quarter = pd.to_numeric(estimated["game_quarter"].iloc[0], errors="coerce")
        point = pd.to_numeric(estimated["quarter_point"].iloc[0], errors="coerce")
        if pd.isna(quarter) or pd.isna(point):
            point_start = np.nan
        else:
            point_start = anchor_lookup.get((int(quarter), int(point)), np.nan)
        possession_throw = pd.to_numeric(
            estimated["possession_throw"],
            errors="coerce",
        ).fillna(1)
        if pd.isna(point_start):
            estimated["video_seconds"] = np.nan
            estimated["point_anchor_seconds"] = np.nan
        else:
            estimated["video_seconds"] = point_start + (possession_throw - 1) * seconds_per_throw
            estimated["point_anchor_seconds"] = point_start
        estimated_paths.append(estimated)

    return estimated_paths


def _possession_anchor_lookup(anchors):
    if anchors is None or anchors.empty:
        return {}

    clean = anchors.dropna(
        subset=[
            "game_quarter",
            "quarter_point",
            "possession_num",
            "is_home_team",
            "video_seconds",
        ]
    ).copy()
    lookup = {}
    for row in clean.itertuples(index=False):
        base_key = (
            int(row.game_quarter),
            int(row.quarter_point),
            int(row.possession_num),
            bool(row.is_home_team),
        )
        team_id = str(row.team_id).lower() if not pd.isna(row.team_id) else ""
        payload = {
            "video_seconds": float(row.video_seconds),
            "note": row.note,
        }
        lookup[base_key + (team_id,)] = payload
        if not team_id:
            lookup[base_key + ("",)] = payload
    return lookup


def _point_anchor_lookup(anchors):
    if anchors is None or anchors.empty:
        return {}

    clean = anchors.dropna(
        subset=["game_quarter", "quarter_point", "video_seconds"]
    ).copy()
    return {
        (int(row.game_quarter), int(row.quarter_point)): float(row.video_seconds)
        for row in clean.itertuples(index=False)
    }


def _clock_anchor_offset(clock_anchors):
    if clock_anchors is None or clock_anchors.empty:
        return np.nan

    clean = clock_anchors.dropna(subset=["source_timestamp", "video_seconds"]).copy()
    if clean.empty:
        return np.nan

    first_anchor = clean.sort_values("source_timestamp").iloc[0]
    return float(first_anchor["source_timestamp"]) - float(first_anchor["video_seconds"])


def add_possession_video_seconds(
    possessions,
    paths,
    anchors,
    point_anchors=None,
    clock_anchors=None,
    seconds_per_throw=3.0,
):
    """Attach possession-start video timestamps, preferring exact possession anchors."""
    if possessions is None or possessions.empty:
        return possessions.copy(), paths

    synced_possessions = _sort_possessions(possessions).copy()
    possession_lookup = _possession_anchor_lookup(anchors)
    point_lookup = _point_anchor_lookup(point_anchors)
    clock_offset = _clock_anchor_offset(clock_anchors)

    path_lookup = {
        str(path["possession_id"].iloc[0]): path.sort_values("possession_throw").copy()
        for path in paths
        if not path.empty and "possession_id" in path
    }
    fallback_offsets = {}
    running_throw_counts = {}
    rows = []

    for row in synced_possessions.itertuples(index=False):
        quarter = pd.to_numeric(getattr(row, "game_quarter", np.nan), errors="coerce")
        point = pd.to_numeric(getattr(row, "quarter_point", np.nan), errors="coerce")
        possession_num = pd.to_numeric(getattr(row, "possession_num", np.nan), errors="coerce")
        is_home_team = bool(getattr(row, "is_home_team", False))
        team_id = str(getattr(row, "team_id", "")).lower()
        possession_id = str(getattr(row, "possession_id"))
        path = path_lookup.get(possession_id, pd.DataFrame())
        throw_count = len(path) if not path.empty else int(getattr(row, "throw_count", 0) or 0)
        first_source_timestamp = np.nan
        if not path.empty and "source_timestamp" in path:
            source_timestamp = pd.to_numeric(path["source_timestamp"], errors="coerce")
            if source_timestamp.notna().any():
                first_source_timestamp = source_timestamp.min()

        video_seconds = np.nan
        sync_status = "missing"
        sync_note = ""
        if not pd.isna(quarter) and not pd.isna(point) and not pd.isna(possession_num):
            point_key = (int(quarter), int(point))
            base_key = (int(quarter), int(point), int(possession_num), is_home_team)
            exact = possession_lookup.get(base_key + (team_id,)) or possession_lookup.get(
                base_key + ("",)
            )
            if exact is not None:
                video_seconds = exact["video_seconds"]
                sync_status = "possession_anchor"
                sync_note = exact.get("note") or ""
            elif not pd.isna(clock_offset) and not pd.isna(first_source_timestamp):
                video_seconds = first_source_timestamp - clock_offset
                sync_status = "source_timestamp_estimate"
                sync_note = "Estimated from game-level source timestamp calibration."
            else:
                if point_key in point_lookup:
                    offset = running_throw_counts.get(point_key, 0) * seconds_per_throw
                    video_seconds = point_lookup[point_key] + offset
                    sync_status = "point_estimate"
                    sync_note = "Estimated from point-start anchor."
            running_throw_counts[point_key] = (
                running_throw_counts.get(point_key, 0) + max(throw_count, 1)
            )

        fallback_offsets[possession_id] = (video_seconds, sync_status, sync_note)
        rows.append((video_seconds, sync_status, sync_note))

    synced_possessions["video_seconds"] = [row[0] for row in rows]
    synced_possessions["video_sync_status"] = [row[1] for row in rows]
    synced_possessions["video_sync_note"] = [row[2] for row in rows]

    synced_paths = []
    for path in paths:
        if path.empty or "possession_id" not in path:
            synced_paths.append(path)
            continue

        possession_id = str(path["possession_id"].iloc[0])
        video_seconds, sync_status, sync_note = fallback_offsets.get(
            possession_id,
            (np.nan, "missing", ""),
        )
        synced = path.sort_values("possession_throw").copy()
        possession_throw = pd.to_numeric(
            synced["possession_throw"],
            errors="coerce",
        ).fillna(1)
        if pd.isna(video_seconds):
            synced["video_seconds"] = np.nan
        else:
            synced["video_seconds"] = video_seconds + (possession_throw - 1) * seconds_per_throw
        synced["possession_video_seconds"] = video_seconds
        synced["video_sync_status"] = sync_status
        synced["video_sync_note"] = sync_note
        synced_paths.append(synced)

    return synced_possessions, synced_paths


def write_fnf_video_browser_html(game_id, youtube_url, possessions, paths, output_path):
    """Write a standalone Friday Night Frisbee video + field browser HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_id = extract_youtube_video_id(youtube_url)

    browser_possessions = _sort_possessions(possessions)
    path_lookup = {
        str(path["possession_id"].iloc[0]): path
        for path in paths
        if not path.empty and "possession_id" in path
    }
    browser_possessions = browser_possessions[
        browser_possessions["possession_id"].astype(str).isin(path_lookup)
    ].reset_index(drop=True)

    payload = {
        "gameId": str(game_id),
        "youtubeUrl": youtube_url,
        "youtubeVideoId": video_id,
        "field": {
            "xMin": FIELD_X_MIN,
            "xMax": FIELD_X_MAX,
            "yMin": FIELD_Y_MIN,
            "yMax": FIELD_Y_MAX,
            "endzoneLowY": ENDZONE_LOW_Y,
            "endzoneHighY": ENDZONE_HIGH_Y,
        },
        "possessions": _json_records(browser_possessions),
        "paths": {
            possession_id: _json_records(path)
            for possession_id, path in path_lookup.items()
        },
    }

    html = _render_html_document(payload)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_fnf_index_html(fnf_games, output_path):
    """Write a local index page linking to generated Friday Night Frisbee browsers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    games = pd.DataFrame(fnf_games).copy()
    if games.empty:
        games = pd.DataFrame(columns=["gameID", "youtube_url", "is_fnf_youtube"])

    rows = []
    for _, game in games.iterrows():
        game_id = game.get("gameID", game.get("game_id", ""))
        game_id = "" if pd.isna(game_id) else str(game_id)
        youtube_url = game.get("youtube_url", "")
        youtube_url = "" if pd.isna(youtube_url) else str(youtube_url)
        browser_path = output_path.parent / f"{game_id}.html"
        browser_exists = browser_path.exists()
        browser_link = (
            f'<a href="{escape(browser_path.name)}">Open browser</a>'
            if browser_exists
            else "<span class=\"missing\">Not generated</span>"
        )
        youtube_link = (
            f'<a href="{escape(youtube_url)}" target="_blank" rel="noopener">YouTube</a>'
            if youtube_url
            else "<span class=\"missing\">Missing</span>"
        )
        matchup = " vs ".join(
            str(game.get(column, "")).upper()
            for column in ["awayTeamID", "homeTeamID"]
            if not pd.isna(game.get(column, np.nan)) and str(game.get(column, "")).strip()
        )
        rows.append(
            "<tr>"
            f"<td>{escape(game_id)}</td>"
            f"<td>{escape(matchup)}</td>"
            f"<td>{escape(str(game.get('startTimestamp', '')))}</td>"
            f"<td>{escape(str(game.get('week', '')))}</td>"
            f"<td>{youtube_link}</td>"
            f"<td>{browser_link}</td>"
            f"<td>{escape(str(game.get('fnf_source', game.get('source', ''))))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Friday Night Frisbee browsers</title>
  <style>
    body {{
      margin: 0;
      padding: 18px;
      background: #f5f7fa;
      color: #0b1a33;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1 {{
      margin: 0 0 14px;
      color: #223a5e;
      font-size: 24px;
    }}
    .panel {{
      background: #fff;
      border: 1px solid #d9e1ea;
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
      overflow-x: auto;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 860px;
      font-size: 14px;
    }}
    th {{
      text-align: left;
      color: #223a5e;
      background: #e9eef5;
      border-bottom: 1px solid #d9e1ea;
      padding: 9px 10px;
    }}
    td {{
      border-bottom: 1px solid #edf1f5;
      padding: 9px 10px;
      vertical-align: top;
    }}
    tr:nth-child(even) td {{ background: #f8fbfe; }}
    a {{ color: #0b4f8a; font-weight: 700; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .missing {{ color: #8a4b00; font-weight: 700; }}
    .hint {{
      margin: 0 0 12px;
      color: #506078;
      line-height: 1.4;
    }}
    code {{
      background: #eef2f7;
      border-radius: 4px;
      padding: 2px 5px;
    }}
  </style>
</head>
<body>
  <h1>Friday Night Frisbee browsers</h1>
  <p class="hint">
    Serve the project folder with <code>python -m http.server 8000</code>,
    then open this page through localhost so YouTube embeds can load.
  </p>
  <div class="panel">
    <table>
      <thead>
        <tr>
          <th>Game ID</th><th>Matchup</th><th>Start</th><th>Week</th>
          <th>YouTube</th><th>Browser</th><th>Source</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _sort_possessions(possessions):
    if possessions.empty:
        return possessions.copy()
    sort_columns = [
        column
        for column in [
            "start_timestamp",
            "GameID",
            "game_quarter",
            "quarter_point",
            "possession_num",
            "is_home_team",
        ]
        if column in possessions
    ]
    return possessions.sort_values(sort_columns).reset_index(drop=True)


def _json_records(frame):
    if frame is None or frame.empty:
        return []
    clean = frame.copy()
    clean = clean.replace({np.nan: None})
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def _render_html_document(payload):
    payload_json = json.dumps(payload, ensure_ascii=True)
    title = f"{payload['gameId']} Friday Night Frisbee browser"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="referrer" content="origin" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      color: #0b1a33;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      background: #f5f7fa;
    }}
    body {{ margin: 0; }}
    .page {{ padding: 18px; }}
    h1 {{
      margin: 0 0 14px;
      font: 700 22px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #223a5e;
    }}
    .browser {{
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 530px;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: #fff;
      border: 1px solid #d9e1ea;
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    .video-wrap {{
      position: relative;
      aspect-ratio: 16 / 9;
      width: 100%;
      background: #071019;
      border-radius: 6px;
      overflow: hidden;
    }}
    #player {{ width: 100%; height: 100%; }}
    .file-warning {{
      display: none;
      position: absolute;
      inset: 0;
      align-items: center;
      justify-content: center;
      padding: 28px;
      background: #1f2933;
      color: #fff;
      font: 700 18px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      text-align: center;
    }}
    .file-warning code {{
      display: inline-block;
      margin-top: 10px;
      padding: 3px 6px;
      border-radius: 4px;
      background: #0b1a33;
      color: #dbeafe;
      font-size: 14px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 155px 90px 90px 1fr auto auto auto;
      gap: 8px;
      align-items: center;
      margin: 12px 0;
    }}
    select, input, button {{
      border: 1px solid #c9d3df;
      border-radius: 4px;
      background: #fff;
      color: #0b1a33;
      min-height: 34px;
      padding: 5px 8px;
      font: inherit;
    }}
    input {{ min-width: 0; }}
    button {{ cursor: pointer; background: #f2f5f8; }}
    button:hover {{ background: #e8edf3; }}
    .count {{ white-space: nowrap; font-weight: 800; }}
    .meta {{
      border-top: 1px solid #d9e1ea;
      margin-top: 10px;
      padding-top: 10px;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid #edf1f5;
      padding: 4px 0;
    }}
    .label {{ color: #637188; }}
    .value {{ font-weight: 800; text-align: right; }}
    .field-layout {{ display: flex; gap: 12px; align-items: flex-start; }}
    #fieldFrame {{
      width: 260px;
      height: 560px;
      background: #f7faf7;
      outline: none;
    }}
    #fieldFrame:focus {{ outline: 2px solid #91a7c2; outline-offset: 4px; }}
    .field {{ fill: #86d973; stroke: #071019; stroke-width: 2.2; }}
    .yard-line {{ stroke: #071019; stroke-width: 1.6; }}
    .center-dot {{ fill: #071019; }}
    .throw line {{ stroke: #071019; stroke-width: 2.2; stroke-linecap: round; }}
    .throw circle {{ fill: #071019; stroke: #071019; stroke-width: 1.2; }}
    .throw:hover line {{ stroke: #c3482b; stroke-width: 3.2; }}
    .throw:hover circle {{ fill: #c3482b; stroke: #071019; stroke-width: 1.8; }}
    .throw.selected line {{ stroke: #c3482b; stroke-width: 3.4; }}
    .throw.selected circle {{ fill: #c3482b; stroke: #071019; stroke-width: 1.9; }}
    .throw {{ cursor: pointer; }}
    .sync-status {{
      display: inline-block;
      margin-top: 8px;
      padding: 3px 7px;
      border-radius: 4px;
      font: 700 12px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef5ff;
      color: #223a5e;
    }}
    .sync-status.missing {{ background: #fff4e5; color: #8a4b00; }}
    .sync-status.point_estimate {{ background: #eef5ff; color: #223a5e; }}
    .sync-status.source_timestamp_estimate {{ background: #eef5ff; color: #223a5e; }}
    .sync-status.possession_anchor {{ background: #e8f8ef; color: #17633a; }}
    #throwDetail {{
      box-sizing: border-box;
      width: 230px;
      min-height: 150px;
      border-left: 1px solid #d8e0e8;
      padding: 8px 0 8px 12px;
      font-size: 12px;
      line-height: 1.4;
    }}
    .detail-kicker {{
      color: #637188;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    .detail-title {{
      font: 800 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin-bottom: 8px;
    }}
    .placeholder {{ color: #637188; }}
    @media (max-width: 980px) {{
      .browser {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{escape(title)}</h1>
    <div class="browser">
      <section class="panel">
        <div class="video-wrap">
          <div id="player"></div>
          <div id="fileWarning" class="file-warning">
            YouTube blocks this embed when opened as a local file.<br />
            Serve the project folder and open this page through localhost.<br />
            <code>python -m http.server 8000</code>
          </div>
        </div>
        <div class="controls">
          <select id="lineFilter" aria-label="Line filter">
            <option value="all">All lines</option>
            <option value="o_line">O-line scores</option>
            <option value="d_line">D-line scores</option>
          </select>
          <input id="minThrowsFilter" type="number" min="0" step="1" aria-label="Minimum throws" title="Minimum throws" />
          <input id="maxThrowsFilter" type="number" min="0" step="1" aria-label="Maximum throws" title="Maximum throws" />
          <select id="possessionSelect" aria-label="Possession selector"></select>
          <button id="previousPossession" type="button">Previous</button>
          <button id="nextPossession" type="button">Next</button>
          <span id="possessionCount" class="count"></span>
        </div>
        <div id="possessionSummary"></div>
      </section>
      <section class="panel field-layout">
        <svg id="fieldFrame" width="260" height="560" viewBox="0 0 260 560" tabindex="0" role="img"></svg>
        <div id="throwDetail"><div class="placeholder">Click a throw on the field.</div></div>
      </section>
    </div>
  </div>
  <script src="https://www.youtube.com/iframe_api"></script>
  <script>
    const DATA = {payload_json};
    let player = null;
    let filteredPossessions = [];
    let selectedPossessionIndex = 0;
    let selectedThrowIndex = 0;

    window.onYouTubeIframeAPIReady = function() {{
      if (window.location.protocol === "file:") {{
        document.getElementById("fileWarning").style.display = "flex";
        return;
      }}
      player = new YT.Player("player", {{
        videoId: DATA.youtubeVideoId,
        playerVars: {{
          rel: 0,
          modestbranding: 1,
          playsinline: 1,
          origin: window.location.origin
        }},
        events: {{
          onReady: function() {{
            seekToPossession(filteredPossessions[selectedPossessionIndex]);
          }}
        }}
      }});
    }};

    function numberValue(value) {{
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    }}

    function fmt(value, digits = 1) {{
      const numeric = numberValue(value);
      return numeric === null ? "-" : numeric.toFixed(digits);
    }}

    function fmtPercent(value) {{
      const numeric = numberValue(value);
      return numeric === null ? "-" : (numeric * 100).toFixed(1) + "%";
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function pathFor(possession) {{
      return DATA.paths[String(possession.possession_id)] || [];
    }}

    function lineLabel(value) {{
      if (value === "o_line") return "O-line";
      if (value === "d_line") return "D-line";
      return "Unknown";
    }}

    function syncStatusLabel(value) {{
      if (value === "possession_anchor") return "Exact possession anchor";
      if (value === "source_timestamp_estimate") return "Source-clock estimate";
      if (value === "point_estimate") return "Point-estimated timestamp";
      return "No timestamp anchor";
    }}

    function sx(value) {{
      const left = 17;
      const fieldWidth = 226;
      return left + ((Number(value) - DATA.field.xMin) / (DATA.field.xMax - DATA.field.xMin)) * fieldWidth;
    }}

    function sy(value) {{
      const top = 17;
      const fieldHeight = 526;
      return top + ((DATA.field.yMax - Number(value)) / (DATA.field.yMax - DATA.field.yMin)) * fieldHeight;
    }}

    function svgEl(tag, attrs = {{}}, text = null) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
      if (text !== null) node.textContent = text;
      return node;
    }}

    function detailHtml(throwRow, index) {{
      return `
        <div class="detail-kicker">Throw ${{index}}</div>
        <div class="detail-title">${{escapeHtml(throwRow.thrower)}} -> ${{escapeHtml(throwRow.receiver)}}</div>
        <div class="row"><span class="label">Distance</span><span class="value">${{fmt(throwRow.throw_distance, 1)}}</span></div>
        <div class="row"><span class="label">CP</span><span class="value">${{fmtPercent(throwRow.cp)}}</span></div>
        <div class="row"><span class="label">aEC</span><span class="value">${{fmt(throwRow.aec, 3)}}</span></div>
        <div class="row"><span class="label">Video time</span><span class="value">${{fmt(throwRow.video_seconds, 1)}}s</span></div>
        <div class="row"><span class="label">Context</span><span class="value">Q${{throwRow.game_quarter ?? "-"}} point ${{throwRow.quarter_point ?? "-"}}</span></div>
      `;
    }}

    function seekToPossession(possession) {{
      if (!possession) return;
      const seconds = numberValue(possession.video_seconds);
      if (seconds !== null && player && typeof player.seekTo === "function") {{
        player.seekTo(seconds, true);
      }}
    }}

    function selectThrow(index) {{
      const possession = filteredPossessions[selectedPossessionIndex];
      if (!possession) return;
      const path = pathFor(possession);
      if (!path.length) return;
      selectedThrowIndex = Math.max(1, Math.min(path.length, index));
      document.querySelectorAll("#fieldFrame .throw.selected").forEach((node) => {{
        node.classList.remove("selected");
      }});
      const node = document.querySelector(`#fieldFrame .throw[data-throw-index="${{selectedThrowIndex}}"]`);
      if (node) node.classList.add("selected");
      const throwRow = path[selectedThrowIndex - 1];
      document.getElementById("throwDetail").innerHTML = detailHtml(throwRow, selectedThrowIndex);
    }}

    function renderField(path) {{
      const svg = document.getElementById("fieldFrame");
      svg.innerHTML = "";
      svg.appendChild(svgEl("rect", {{ class: "field", x: 17, y: 17, width: 226, height: 526 }}));
      [DATA.field.endzoneLowY, DATA.field.endzoneHighY].forEach((yValue) => {{
        svg.appendChild(svgEl("line", {{ class: "yard-line", x1: 17, x2: 243, y1: sy(yValue), y2: sy(yValue) }}));
      }});
      [40, 80].forEach((yValue) => {{
        svg.appendChild(svgEl("circle", {{ class: "center-dot", cx: sx(0), cy: sy(yValue), r: 2.5 }}));
      }});
      path.forEach((throwRow, zeroIndex) => {{
        const index = zeroIndex + 1;
        const group = svgEl("g", {{ class: "throw", "data-throw-index": index }});
        group.appendChild(svgEl("line", {{
          x1: sx(throwRow.ThrowerX),
          y1: sy(throwRow.ThrowerY),
          x2: sx(throwRow.ReceiverX),
          y2: sy(throwRow.ReceiverY)
        }}));
        group.appendChild(svgEl("circle", {{ cx: sx(throwRow.ThrowerX), cy: sy(throwRow.ThrowerY), r: 2.8 }}));
        group.appendChild(svgEl("circle", {{ cx: sx(throwRow.ReceiverX), cy: sy(throwRow.ReceiverY), r: 3.1 }}));
        group.appendChild(svgEl("title", {{}}, `Throw ${{index}}: ${{throwRow.thrower ?? ""}} to ${{throwRow.receiver ?? ""}}`));
        group.addEventListener("mouseenter", () => {{
          document.getElementById("throwDetail").innerHTML = detailHtml(throwRow, index);
        }});
        group.addEventListener("click", () => {{
          svg.focus();
          selectThrow(index);
        }});
        svg.appendChild(group);
      }});
    }}

    function renderSummary(possession, path) {{
      const team = escapeHtml(possession.team_id ?? "-");
      const side = possession.is_home_team ? "Home" : "Away";
      const syncStatus = possession.video_sync_status || "missing";
      const syncNote = possession.video_sync_note ? ` - ${{escapeHtml(possession.video_sync_note)}}` : "";
      return `
        <h2 style="margin:0 0 8px;font:800 18px system-ui;color:#0b1a33">${{team}}</h2>
        <div>${{escapeHtml(possession.GameID)}}</div>
        <div>Q${{possession.game_quarter}} - point ${{possession.quarter_point}} - possession ${{possession.possession_num}} - ${{side}} - ${{lineLabel(possession.line_type)}}</div>
        <div class="sync-status ${{escapeHtml(syncStatus)}}">${{syncStatusLabel(syncStatus)}}${{syncNote}}</div>
        <div class="meta">
          <div class="row"><span class="label">Throws</span><span class="value">${{path.length}}</span></div>
          <div class="row"><span class="label">Start Y</span><span class="value">${{fmt(possession.start_y, 1)}}</span></div>
          <div class="row"><span class="label">End Y</span><span class="value">${{fmt(possession.end_y, 1)}}</span></div>
          <div class="row"><span class="label">Net Y progress</span><span class="value">${{fmt(possession.field_progress, 1)}}</span></div>
          <div class="row"><span class="label">Video time</span><span class="value">${{fmt(possession.video_seconds, 1)}}s</span></div>
          <div class="row"><span class="label">Total aEC</span><span class="value">${{fmt(possession.total_aec, 3)}}</span></div>
          <div class="row"><span class="label">aEC / throw</span><span class="value">${{fmt(possession.aec_per_throw, 3)}}</span></div>
        </div>
      `;
    }}

    function renderPossession(index) {{
      if (!filteredPossessions.length) {{
        selectedPossessionIndex = 0;
        selectedThrowIndex = 0;
        document.getElementById("possessionCount").innerHTML = "<b>0</b> of <b>0</b>";
        document.getElementById("possessionSummary").innerHTML = "<b>No scoring possessions match this line filter.</b>";
        document.getElementById("throwDetail").innerHTML = '<div class="placeholder">Click a throw on the field.</div>';
        renderField([]);
        return;
      }}
      selectedPossessionIndex = Math.max(0, Math.min(filteredPossessions.length - 1, index));
      selectedThrowIndex = 0;
      const possession = filteredPossessions[selectedPossessionIndex];
      const path = pathFor(possession);
      document.getElementById("possessionSelect").value = String(selectedPossessionIndex);
      document.getElementById("possessionCount").innerHTML = `<b>${{selectedPossessionIndex + 1}}</b> of <b>${{filteredPossessions.length}}</b>`;
      document.getElementById("possessionSummary").innerHTML = renderSummary(possession, path);
      renderField(path);
      if (path.length) {{
        selectThrow(1);
      }} else {{
        document.getElementById("throwDetail").innerHTML = '<div class="placeholder">Click a throw on the field.</div>';
      }}
      seekToPossession(possession);
    }}

    function refreshPossessionOptions() {{
      const lineValue = document.getElementById("lineFilter").value;
      const minThrows = numberValue(document.getElementById("minThrowsFilter").value);
      const maxThrows = numberValue(document.getElementById("maxThrowsFilter").value);
      filteredPossessions = DATA.possessions.filter((possession) => {{
        const lineMatches = lineValue === "all" || possession.line_type === lineValue;
        const throwCount = numberValue(possession.throw_count);
        const minMatches = minThrows === null || throwCount === null || throwCount >= minThrows;
        const maxMatches = maxThrows === null || throwCount === null || throwCount <= maxThrows;
        return lineMatches && minMatches && maxMatches;
      }});
      const select = document.getElementById("possessionSelect");
      select.innerHTML = "";
      filteredPossessions.forEach((possession, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${{index + 1}}. ${{lineLabel(possession.line_type)}} | ${{possession.GameID}} | Q${{possession.game_quarter}} P${{possession.quarter_point}} | poss ${{possession.possession_num}} | ${{possession.throw_count}} throws`;
        select.appendChild(option);
      }});
      renderPossession(0);
    }}

    function init() {{
      const throwCounts = DATA.possessions
        .map((possession) => numberValue(possession.throw_count))
        .filter((value) => value !== null);
      const minThrows = throwCounts.length ? Math.min(...throwCounts) : 0;
      const maxThrows = throwCounts.length ? Math.max(...throwCounts) : 0;
      const minThrowsInput = document.getElementById("minThrowsFilter");
      const maxThrowsInput = document.getElementById("maxThrowsFilter");
      minThrowsInput.value = minThrows;
      minThrowsInput.min = minThrows;
      minThrowsInput.max = maxThrows;
      maxThrowsInput.value = maxThrows;
      maxThrowsInput.min = minThrows;
      maxThrowsInput.max = maxThrows;
      document.getElementById("lineFilter").addEventListener("change", refreshPossessionOptions);
      minThrowsInput.addEventListener("change", refreshPossessionOptions);
      maxThrowsInput.addEventListener("change", refreshPossessionOptions);
      const select = document.getElementById("possessionSelect");
      select.addEventListener("change", () => renderPossession(Number(select.value)));
      document.getElementById("previousPossession").addEventListener("click", () => renderPossession(selectedPossessionIndex - 1));
      document.getElementById("nextPossession").addEventListener("click", () => renderPossession(selectedPossessionIndex + 1));
      document.getElementById("fieldFrame").addEventListener("keydown", (event) => {{
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        event.preventDefault();
        const direction = event.key === "ArrowRight" ? 1 : -1;
        selectThrow((selectedThrowIndex || 0) + direction);
      }});
      refreshPossessionOptions();
    }}

    init();
  </script>
</body>
</html>
"""
