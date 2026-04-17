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


def _classify_room_type(labels: list[str], patterns: dict) -> tuple[str, str | None]:
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
        if s is None or e is None:
            continue
        if s == e:
            continue
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
) -> list[dict]:
    """Return a list of room records per the §4 schema (minus room_id/bounding/labels
    which are filled in here).

    Note: modifies each wall edge in ``graph`` to append the bounding room_id
    to its ``adjacent_room_ids`` list so downstream stages can use the link.
    """
    min_area = float(config["room_min_area"])
    type_patterns = dict(config.get("room_type_label_patterns") or {})

    lines = _edges_to_lines(graph)
    if not lines:
        return []

    merged = unary_union(MultiLineString(lines))
    raw_polys = list(polygonize(merged))
    candidate_polys = [p for p in raw_polys if p.area >= min_area]
    if not candidate_polys:
        return []

    walls = _walls_by_segment_id(graph)

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

        room_type, matched_substring = _classify_room_type(label_texts, type_patterns)

        # Room name/number: split the first non-empty label into number + name if possible
        room_name: str | None = None
        room_number: str | None = None
        for txt in label_texts:
            candidate = re.match(r"^\s*([0-9]{2,4}[a-zA-Z]?)\s*$", txt)
            if candidate and room_number is None:
                room_number = candidate.group(1)
            elif txt.strip() and room_name is None:
                room_name = txt.strip()

        # Walls that bound this room (centerline midpoint within small tolerance of poly boundary)
        tolerance = float(config.get("junction_snap_distance", 3.0)) * 3.0
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
                    f"label_match:{matched_substring}" if matched_substring else "polygon_closure"
                ),
                "requires_cross_document_validation": False,
            }
        )

    rooms.sort(key=lambda r: r["area"], reverse=True)
    return rooms
