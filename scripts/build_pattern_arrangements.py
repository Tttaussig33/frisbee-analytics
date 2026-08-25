"""Build geometry-first arrangement variants from the hand-organized examples.

The browser arrangement files are intentionally simple JSON checkpoints.  This
script treats the existing checkpoint for each example team as weak supervision:
small hand-organized rows provide seed centroids, while oversized generic rows
are repartitioned by field-path geometry.  The original checkpoint is never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ufa.shownspace_paths import (  # noqa: E402
    build_possessions,
    calculate_possession_shape_features,
)


EXAMPLE_TEAMS = ("sol", "empire", "spiders", "windchill")
DEFAULT_TARGET_GROUP_SIZE = 16
DEFAULT_REFERENCE_MAX_SIZE = 24
DEFAULT_MIN_GROUPS = 24
DEFAULT_MAX_GROUPS = 40
SHAPE_CHECKPOINTS = np.linspace(0, 1, 9)
GENERIC_GROUP_RE = re.compile(r"^pattern group(?:\s+\d+)?$", re.IGNORECASE)


def _team_game_files(source_dir: Path, team_id: str) -> list[Path]:
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
    return [csv_path for _, csv_path in candidates]


def _load_team_possessions(source_dir: Path, team_id: str, line_type: str | None = None):
    game_files = _team_game_files(source_dir, team_id)
    if not game_files:
        raise ValueError(f"No cached games found for {team_id} in {source_dir}")

    throws = pd.concat(
        [pd.read_csv(csv_path, low_memory=False) for csv_path in game_files],
        ignore_index=True,
    )
    possessions, paths = build_possessions(
        throws,
        team_id=team_id,
        outcomes=("goal", "turnover"),
    )
    if line_type:
        possessions = possessions.loc[
            possessions["line_type"].eq(line_type)
        ].reset_index(drop=True)
        paths = [
            path
            for path in paths
            if not path.empty and str(path["line_type"].iloc[0]) == line_type
        ]
    return game_files, possessions, paths


def _read_arrangement(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise ValueError(f"Arrangement must contain a groups list: {path}")
    return payload


def _is_reference_group(group: dict, reference_max_size: int) -> bool:
    cards = group.get("possessions") or []
    if not cards or group.get("auto_unsorted"):
        return False

    title = str(group.get("title") or "").strip()
    normalized = title.lower()
    if normalized == "unfiltered" or normalized.startswith("unsorted"):
        return False
    if GENERIC_GROUP_RE.match(title) and len(cards) > reference_max_size:
        return False
    return True


def _arrangement_order(payload: dict) -> tuple[list[str], dict[str, int], dict[str, str]]:
    ordered_ids: list[str] = []
    original_index: dict[str, int] = {}
    original_labels: dict[str, str] = {}
    for group in payload.get("groups", []):
        for card in group.get("possessions") or []:
            possession_id = str(card.get("possession_id", ""))
            if not possession_id or possession_id in original_index:
                continue
            original_index[possession_id] = len(ordered_ids)
            ordered_ids.append(possession_id)
            original_labels[possession_id] = str(card.get("label") or "")
    return ordered_ids, original_index, original_labels


def _feature_frame(possessions: pd.DataFrame, paths: list[pd.DataFrame]):
    shape_features = calculate_possession_shape_features(
        possessions,
        paths,
        checkpoints=SHAPE_CHECKPOINTS,
    )
    enriched = possessions.copy()
    for column in shape_features.columns:
        enriched[column] = shape_features[column].to_numpy()

    shape_columns = [
        column
        for column in enriched.columns
        if column.startswith("shape_")
    ]
    # All clustering signal comes from field locations and path shape. Throw
    # count is deliberately omitted so a five-throw and ten-throw path can
    # still share a spatial group.
    features = (
        enriched.reindex(columns=shape_columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    return enriched, features


def _reference_centroids(
    payload: dict,
    feature_by_id: pd.DataFrame,
    reference_max_size: int,
):
    centroids = []
    for group in payload.get("groups", []):
        if not _is_reference_group(group, reference_max_size):
            continue
        ids = [
            str(card.get("possession_id", ""))
            for card in group.get("possessions") or []
        ]
        ids = [possession_id for possession_id in ids if possession_id in feature_by_id.index]
        if not ids:
            continue
        centroids.append(feature_by_id.loc[ids].mean(axis=0).to_numpy())
    return centroids


def _cluster_count(
    card_count: int,
    target_group_size: int,
    min_groups: int,
    max_groups: int,
    reference_count: int,
) -> int:
    target = int(np.ceil(card_count / max(1, target_group_size)))
    return max(
        1,
        min(max(target, min_groups, reference_count), max_groups),
    )


def _lane(value: float) -> str:
    if pd.isna(value):
        return "mixed"
    if value <= -8.88:
        return "left"
    if value >= 8.88:
        return "right"
    return "middle"


def _flow(value: float) -> str:
    if pd.isna(value):
        return "mixed"
    if value >= 0.75:
        return "direct"
    if value <= 0.50:
        return "winding"
    return "mixed"


def _cluster_title(
    cluster_frame: pd.DataFrame,
    index: int,
    prefix: str = "Codex pattern",
) -> str:
    start_lane = _lane(pd.to_numeric(cluster_frame["shape_start_x"], errors="coerce").median())
    end_lane = _lane(pd.to_numeric(cluster_frame["shape_end_x"], errors="coerce").median())
    flow = _flow(pd.to_numeric(cluster_frame["shape_directness"], errors="coerce").median())
    median_throws = pd.to_numeric(
        cluster_frame["throw_count"], errors="coerce"
    ).median()
    if pd.isna(median_throws):
        throw_text = "mixed throws"
    elif float(median_throws).is_integer():
        throw_text = f"{int(median_throws)}-throw median"
    else:
        throw_text = f"{median_throws:.1f}-throw median"
    return f"{prefix} {index:02d} | {start_lane}-{end_lane} | {flow} | {throw_text}"


def _make_card_label(row: pd.Series) -> str:
    outcome = str(row.get("outcome", "unknown")).title()
    line_type = str(row.get("line_type", "unknown")).replace("_", "-").title()
    game_id = str(row.get("GameID", "-"))
    quarter = row.get("game_quarter", "-")
    point = row.get("quarter_point", "-")
    throws = int(row.get("throw_count", 0))
    return f"{outcome} {line_type} | {game_id} | Q{quarter} P{point} | {throws} throws"


def build_codex_arrangement(
    team_id: str,
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    original_payload: dict,
    *,
    target_group_size: int = DEFAULT_TARGET_GROUP_SIZE,
    reference_max_size: int = DEFAULT_REFERENCE_MAX_SIZE,
    min_groups: int = DEFAULT_MIN_GROUPS,
    max_groups: int = DEFAULT_MAX_GROUPS,
    random_state: int = 2026,
    line_type: str | None = None,
) -> dict:
    arrangement_name = (
        "Codex spatial arrangement"
        if line_type is None
        else f"Codex {'O-line' if line_type == 'o_line' else 'D-line'} spatial arrangement"
    )
    group_title_prefix = (
        "Codex pattern"
        if line_type is None
        else f"Codex {'O-line' if line_type == 'o_line' else 'D-line'} pattern"
    )
    enriched, raw_features = _feature_frame(possessions, paths)
    if enriched.empty:
        return {
            "version": 3,
            "arrangement_name": arrangement_name,
            "team_id": team_id,
            "line_type": line_type or "all",
            "cards_shown": 0,
            "filtered_possessions": 0,
            "groups": [],
        }

    ids = enriched["possession_id"].astype(str).tolist()
    feature_by_id = raw_features.copy()
    feature_by_id.index = ids
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw_features)
    scaled_by_id = pd.DataFrame(scaled, index=ids, columns=raw_features.columns)

    seed_centroids = _reference_centroids(
        original_payload,
        scaled_by_id,
        reference_max_size,
    )
    cluster_count = _cluster_count(
        len(ids),
        target_group_size,
        min_groups,
        max_groups,
        len(seed_centroids),
    )

    if cluster_count == 1:
        labels = np.zeros(len(ids), dtype=int)
    else:
        new_seed_count = max(0, cluster_count - len(seed_centroids))
        if new_seed_count:
            seed_model = KMeans(
                n_clusters=new_seed_count,
                random_state=random_state,
                n_init=10,
            )
            seed_model.fit(scaled)
            new_centroids = seed_model.cluster_centers_
        else:
            new_centroids = np.empty((0, scaled.shape[1]))

        initial_centroids = np.vstack([seed_centroids, new_centroids])
        model = KMeans(
            n_clusters=cluster_count,
            init=initial_centroids[:cluster_count],
            n_init=1,
            random_state=random_state,
            max_iter=300,
        )
        labels = model.fit_predict(scaled)

    enriched = enriched.copy()
    enriched["_cluster"] = labels
    original_order, original_index, _ = _arrangement_order(original_payload)
    order_fallback = {possession_id: len(original_index) + index for index, possession_id in enumerate(ids)}
    enriched["_original_order"] = enriched["possession_id"].map(
        lambda possession_id: original_index.get(str(possession_id), order_fallback[str(possession_id)])
    )

    clusters = []
    for cluster_id, cluster_frame in enriched.groupby("_cluster", sort=False):
        cluster_frame = cluster_frame.sort_values(
            ["throw_count", "start_y", "end_y", "_original_order"],
            na_position="last",
        )
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "frame": cluster_frame,
                "sort_key": (
                    pd.to_numeric(cluster_frame["throw_count"], errors="coerce").median(),
                    pd.to_numeric(cluster_frame["start_y"], errors="coerce").median(),
                    pd.to_numeric(cluster_frame["end_y"], errors="coerce").median(),
                    pd.to_numeric(cluster_frame["shape_start_x"], errors="coerce").median(),
                    int(cluster_id),
                ),
            }
        )
    clusters.sort(key=lambda item: tuple(
        float(value) if pd.notna(value) else float("inf")
        for value in item["sort_key"]
    ))

    groups = []
    for group_index, cluster in enumerate(clusters, start=1):
        cluster_frame = cluster["frame"]
        groups.append(
            {
                "group_index": group_index,
                "title": _cluster_title(
                    cluster_frame,
                    group_index,
                    prefix=group_title_prefix,
                ),
                "break_before": True,
                "auto_unsorted": False,
                "possessions": [
                    {
                        "possession_id": str(row["possession_id"]),
                        "label": _make_card_label(row),
                        "overlay_selected": False,
                    }
                    for _, row in cluster_frame.iterrows()
                ],
            }
        )

    return {
        "version": 3,
        "arrangement_name": arrangement_name,
        "team_id": team_id,
        "line_type": line_type or "all",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_arrangement": f"{team_id}.json",
        "method": {
            "description": (
                "Geometry-first K-means grouping seeded by smaller groups in the "
                "team's hand-organized arrangement."
            ),
            "shape_checkpoints": [float(value) for value in SHAPE_CHECKPOINTS],
            "target_group_size": target_group_size,
            "reference_max_size": reference_max_size,
            "cluster_count": len(groups),
            "random_state": random_state,
            "line_type": line_type or "all",
        },
        "cards_shown": len(enriched),
        "filtered_possessions": len(enriched),
        "groups": groups,
    }


def build_team_arrangement(
    team_id: str,
    source_dir: Path,
    arrangement_dir: Path,
    line_type: str | None = None,
    **kwargs,
) -> tuple[dict, dict]:
    original_path = arrangement_dir / f"{team_id}.json"
    if not original_path.exists():
        raise ValueError(f"Missing hand-organized example: {original_path}")
    original_payload = _read_arrangement(original_path)
    game_files, possessions, paths = _load_team_possessions(
        source_dir,
        team_id,
        line_type=line_type,
    )
    payload = build_codex_arrangement(
        team_id,
        possessions,
        paths,
        original_payload,
        line_type=line_type,
        **kwargs,
    )
    payload["source_games"] = len(game_files)
    return payload, original_payload


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build geometry-first Codex arrangement variants from saved examples."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--team", action="append", dest="teams")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-group-size", type=int, default=DEFAULT_TARGET_GROUP_SIZE)
    parser.add_argument("--reference-max-size", type=int, default=DEFAULT_REFERENCE_MAX_SIZE)
    parser.add_argument("--min-groups", type=int, default=DEFAULT_MIN_GROUPS)
    parser.add_argument("--max-groups", type=int, default=DEFAULT_MAX_GROUPS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--line-type",
        choices=("all", "o_line", "d_line"),
        default="all",
        help="Scope the generated arrangement to one line type; default is all lines.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    source_dir = args.source_dir or (
        REPO_ROOT / "data" / "raw" / f"shownspace_throws_{args.season}_by_game"
    )
    arrangement_dir = REPO_ROOT / "data" / "arrangements" / str(args.season)
    output_dir = args.output_dir or arrangement_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    teams = [team.strip().lower() for team in (args.teams or EXAMPLE_TEAMS)]
    line_type = None if args.line_type == "all" else args.line_type
    line_suffix = line_type.replace("_", "-") if line_type else ""
    output_suffix = "-codex" if line_type is None else f"-codex-{line_suffix}"

    for team_id in teams:
        payload, original_payload = build_team_arrangement(
            team_id,
            source_dir,
            arrangement_dir,
            line_type=line_type,
            target_group_size=max(1, args.target_group_size),
            reference_max_size=max(1, args.reference_max_size),
            min_groups=max(1, args.min_groups),
            max_groups=max(1, args.max_groups),
            random_state=args.seed,
        )
        output_path = output_dir / f"{team_id}{output_suffix}.json"
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        original_cards = sum(
            len(group.get("possessions") or [])
            for group in original_payload.get("groups", [])
        )
        group_sizes = [len(group["possessions"]) for group in payload["groups"]]
        print(
            f"{team_id.title()}: {payload['source_games']} games, "
            f"{payload['cards_shown']} cards, "
            f"{len(payload['groups'])} Codex groups "
            f"({args.line_type}) "
            f"(original {len(original_payload.get('groups', []))} groups / "
            f"{original_cards} cards; size range "
            f"{min(group_sizes) if group_sizes else 0}-{max(group_sizes) if group_sizes else 0}) "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
