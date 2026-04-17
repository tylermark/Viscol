"""Stage 5 — Room detection.

Derives rooms by polygonizing the wall centerline graph. Each closed polygon
above ``room_min_area`` is a candidate room. Text labels whose position falls
inside a polygon are attached to that room. Room type is inferred from
configurable label patterns.

Rooms are inherently lossy: real floor-plan wall graphs have gaps (door
openings, drafting imprecision) that prevent full polygonization. This stage
returns whatever rooms DO close cleanly; the rest of the drawing's spatial
structure lives in walls + openings. Per CLAUDE.md §9, we'd rather emit an
incomplete room list with rule_triggered noted than paper over gaps with
opaque heuristics.
"""

from __future__ import annotations

import re
import uuid

import networkx as nx
from shapely.geometry import MultiLineString, Point, Polygon
from shapely.ops import polygonize, unary_union


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().upper()


def _classify_room_type_from_fixtures(
    polygon: Polygon,
    fixtures: list[dict],
) -> tuple[str | None, str | None]:
    """Return (room_type, fixture_type_matched) based on contained/nearby fixtures.

    Priority order: toilet/sink -> bathroom, stove -> kitchen, washer/dryer -> laundry.
    Returns (None, None) if no defining fixture is found.
    """
    # Fixture type to room type mapping with priority order
    fixture_to_room = [
        (["toilet", "sink"], "bathroom"),
        (["stove", "range", "cooktop"], "kitchen"),
        (["washer", "dryer", "laundry"], "laundry"),
        (["hvac", "furnace", "boiler"], "mechanical"),
    ]

    for fixture in fixtures:
        fixture_type = fixture.get("type", "").lower()
        # Check if fixture is inside the room polygon
        # (centroid or position attribute, depending on fixture structure)
        pos = fixture.get("position") or fixture.get("centroid")
        if not pos or len(pos) < 2:
            continue
        from shapely.geometry import Point
        if polygon.contains(Point(pos[0], pos[1])):
            # Match fixture type to room type
            for fixture_types, room_type in fixture_to_room:
                if any(ft in fixture_type for ft in fixture_types):
                    return room_type, fixture_type
    return None, None


def _classify_room_type_from_labels(labels: list[str], patterns: dict) -> tuple[str, str | None]:
    """Return (room_type, matched_label_substring). 'unknown' if no pattern hits."""
    for label in labels:
        norm = _normalize_label(label)
        for room_type, substrings in patterns.items():
            for substring in substrings:
                if substring in norm:
                    return room_type, substring
    return "unknown", None


def _edges_to_lines(graph: nx.MultiGraph) -> list[list[tuple[float, float]]]:
    lines = []
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        s = data.get("start")
        e = data.get("end")
        if s is None:
            raise ValueError(f"Edge ({_u}, {_v}, key={_key}) missing 'start' attribute")
        if e is None:
            raise ValueError(f"Edge ({_u}, {_v}, key={_key}) missing 'end' attribute")
        if s == e or (abs(s[0] - e[0]) < 1e-9 and abs(s[1] - e[1]) < 1e-9):
            raise ValueError(f"Edge ({_u}, {_v}, key={_key}) has zero length: start={s}, end={e}")
        lines.append([(float(s[0]), float(s[1])), (float(e[0]), float(e[1]))])
    return lines


def _walls_by_segment_id(graph: nx.MultiGraph) -> dict[str, dict]:
    walls: dict[str, dict] = {}
    for _u, _v, key, data in graph.edges(keys=True, data=True):
        walls[key] = data
    return walls


def _wall_midpoint(data: dict) -> tuple[float, float]:
    s = data["start"]
    e = data["end"]
    return ((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0)


def detect_rooms(
    graph: nx.MultiGraph,
    text_blocks: list[dict],
    config: dict,
    fixtures: list[dict] | None = None,
) -> list[dict]:
    """Return a list of room records per the §4 schema (minus room_id/bounding/labels
    which are filled in here).

    Note: modifies each wall edge in ``graph`` to append the bounding room_id
    to its ``adjacent_room_ids`` list so downstream stages can use the link.

    Args:
        graph: Wall centerline graph
        text_blocks: Text labels from Stage 1
        config: Pipeline config
        fixtures: Optional list of fixture records (from future fixture detection stage)
    """
    min_area = float(config["room_min_area"])
    type_patterns = dict(config.get("room_type_label_patterns") or {})
    room_number_regex = re.compile(config["text_room_number_pattern"])

    # Validate graph connectivity before attempting polygonization
    if not nx.is_connected(graph):
        raise ValueError("Wall graph is disconnected; cannot reliably polygonize rooms")

    lines = _edges_to_lines(graph)
    if not lines:
        raise ValueError("No valid wall segments found in graph; cannot detect rooms")

    merged = unary_union(MultiLineString(lines))
    raw_polys = list(polygonize(merged))
    candidate_polys = [p for p in raw_polys if p.area >= min_area]
    if not candidate_polys:
        return []

    walls = _walls_by_segment_id(graph)
    # Default to empty list if no fixtures provided
    fixtures = fixtures or []

    rooms: list[dict] = []
    for poly in candidate_polys:
        room_id = str(uuid.uuid4())
        boundary = poly.exterior

        # Text labels inside the polygon (centroid-in-poly test)
        label_region_ids: list[str] = []
        label_texts: list[str] = []
        for tb in text_blocks:
            bbox = tb.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            if poly.contains(Point(cx, cy)):
                # text_region_id will be assigned later in Stage 7; for now carry the text
                label_texts.append(tb.get("text") or "")
                label_region_ids.append(tb.get("_text_region_id") or "")

        # Try fixture-based typing first (if fixtures are available)
        room_type, matched_substring = _classify_room_type_from_fixtures(poly, fixtures)
        if room_type is None:
            # Fall back to label-based typing
            room_type, matched_substring = _classify_room_type_from_labels(label_texts, type_patterns)

        # Room name/number: split the first non-empty label into number + name if possible.
        # A label that matches the room-number pattern (e.g. "101", "203a") is a number —
        # it must never fall through to room_name, even when room_number is already set.
        room_name: str | None = None
        room_number: str | None = None
        for txt in label_texts:
            stripped = txt.strip()
            candidate = room_number_regex.fullmatch(stripped) if stripped else None
            if candidate:
                if room_number is None:
                    # Prefer a captured group when the pattern provides one,
                    # otherwise the whole matched string.
                    room_number = candidate.group(1) if candidate.groups() else candidate.group(0)
                continue
            if stripped and room_name is None:
                room_name = stripped

        # Walls that bound this room (centerline midpoint within small tolerance of poly boundary)
        tolerance = float(config["room_boundary_tolerance"])
        bounding_segment_ids: list[str] = []
        for segment_id, wall_data in walls.items():
            m = Point(*_wall_midpoint(wall_data))
            if boundary.distance(m) <= tolerance:
                bounding_segment_ids.append(segment_id)
                adj = wall_data.setdefault("adjacent_room_ids", [])
                if room_id not in adj:
                    adj.append(room_id)

        centroid = poly.centroid
        rooms.append(
            {
                "room_id": room_id,
                "polygon": [[float(x), float(y)] for x, y in poly.exterior.coords],
                "area": float(poly.area),
                "centroid": [float(centroid.x), float(centroid.y)],
                "bounding_walls": sorted(set(bounding_segment_ids)),
                "text_labels": [tid for tid in label_region_ids if tid],
                "room_name": room_name,
                "room_number": room_number,
                "room_type": room_type,
                "_label_texts": label_texts,  # internal for cross-stage wiring
                "_matched_substring": matched_substring,
                "rule_triggered": (
                    f"fixture_match:{matched_substring}" if matched_substring and room_type != "unknown"
                    else f"label_match:{matched_substring}" if matched_substring
                    else "polygon_closure"
                ),
                "requires_cross_document_validation": False,
            }
        )

    rooms.sort(key=lambda r: r["area"], reverse=True)
    return rooms