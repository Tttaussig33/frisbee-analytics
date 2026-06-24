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
      aspect-ratio: 16 / 9;
      width: 100%;
      background: #071019;
      border-radius: 6px;
      overflow: hidden;
    }}
    #player {{ width: 100%; height: 100%; }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 8px;
      align-items: center;
      margin: 12px 0;
    }}
    select, button {{
      border: 1px solid #c9d3df;
      border-radius: 4px;
      background: #fff;
      color: #0b1a33;
      min-height: 34px;
      padding: 5px 8px;
      font: inherit;
    }}
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
        <div class="video-wrap"><div id="player"></div></div>
        <div class="controls">
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
    let selectedPossessionIndex = 0;
    let selectedThrowIndex = 0;

    window.onYouTubeIframeAPIReady = function() {{
      player = new YT.Player("player", {{
        videoId: DATA.youtubeVideoId,
        playerVars: {{ rel: 0, modestbranding: 1 }}
      }});
    }};

    function numberValue(value) {{
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

    function seekToThrow(throwRow) {{
      const seconds = numberValue(throwRow.video_seconds);
      if (seconds !== null && player && typeof player.seekTo === "function") {{
        player.seekTo(seconds, true);
      }}
    }}

    function selectThrow(index, shouldSeek = false) {{
      const possession = DATA.possessions[selectedPossessionIndex];
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
      if (shouldSeek) seekToThrow(throwRow);
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
          selectThrow(index, true);
        }});
        svg.appendChild(group);
      }});
    }}

    function renderSummary(possession, path) {{
      const team = escapeHtml(possession.team_id ?? "-");
      const side = possession.is_home_team ? "Home" : "Away";
      return `
        <h2 style="margin:0 0 8px;font:800 18px system-ui;color:#0b1a33">${{team}}</h2>
        <div>${{escapeHtml(possession.GameID)}}</div>
        <div>Q${{possession.game_quarter}} - point ${{possession.quarter_point}} - possession ${{possession.possession_num}} - ${{side}}</div>
        <div class="meta">
          <div class="row"><span class="label">Throws</span><span class="value">${{path.length}}</span></div>
          <div class="row"><span class="label">Start Y</span><span class="value">${{fmt(possession.start_y, 1)}}</span></div>
          <div class="row"><span class="label">End Y</span><span class="value">${{fmt(possession.end_y, 1)}}</span></div>
          <div class="row"><span class="label">Net Y progress</span><span class="value">${{fmt(possession.field_progress, 1)}}</span></div>
          <div class="row"><span class="label">Total aEC</span><span class="value">${{fmt(possession.total_aec, 3)}}</span></div>
          <div class="row"><span class="label">aEC / throw</span><span class="value">${{fmt(possession.aec_per_throw, 3)}}</span></div>
        </div>
      `;
    }}

    function renderPossession(index) {{
      selectedPossessionIndex = Math.max(0, Math.min(DATA.possessions.length - 1, index));
      selectedThrowIndex = 0;
      const possession = DATA.possessions[selectedPossessionIndex];
      const path = pathFor(possession);
      document.getElementById("possessionSelect").value = String(selectedPossessionIndex);
      document.getElementById("possessionCount").innerHTML = `<b>${{selectedPossessionIndex + 1}}</b> of <b>${{DATA.possessions.length}}</b>`;
      document.getElementById("possessionSummary").innerHTML = renderSummary(possession, path);
      document.getElementById("throwDetail").innerHTML = '<div class="placeholder">Click a throw on the field.</div>';
      renderField(path);
    }}

    function init() {{
      const select = document.getElementById("possessionSelect");
      DATA.possessions.forEach((possession, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${{index + 1}}. ${{possession.GameID}} | Q${{possession.game_quarter}} P${{possession.quarter_point}} | poss ${{possession.possession_num}} | ${{possession.throw_count}} throws`;
        select.appendChild(option);
      }});
      select.addEventListener("change", () => renderPossession(Number(select.value)));
      document.getElementById("previousPossession").addEventListener("click", () => renderPossession(selectedPossessionIndex - 1));
      document.getElementById("nextPossession").addEventListener("click", () => renderPossession(selectedPossessionIndex + 1));
      document.getElementById("fieldFrame").addEventListener("keydown", (event) => {{
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        event.preventDefault();
        const direction = event.key === "ArrowRight" ? 1 : -1;
        selectThrow((selectedThrowIndex || 0) + direction, true);
      }});
      if (DATA.possessions.length) {{
        renderPossession(0);
      }} else {{
        document.getElementById("possessionSummary").innerHTML = "<b>No scoring possessions available.</b>";
      }}
    }}

    init();
  </script>
</body>
</html>
"""
