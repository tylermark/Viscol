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

import math
import re
import uuid

import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point, Polygon
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


def _is_valid_segment(data: dict) -> bool:
    """Return True when the edge has both endpoints and non-zero length.

    Shared by every consumer that reads a wall's start/end so they all
    see the same set of edges. Silent skip (not raise) is intentional:
    Stage 4 endpoint clustering can produce legitimately-zero-length
    edges when two wall endpoints collapse to a single junction, and
    hard-failing on them would prevent room detection on any real plan.
    """
    s = data.get("start")
    e = data.get("end")
    if s is None or e is None:
        return False
    if abs(s[0] - e[0]) < 1e-9 and abs(s[1] - e[1]) < 1e-9:
        return False
    return True


def _edges_to_lines(graph: nx.MultiGraph) -> list[list[tuple[float, float]]]:
    lines = []
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        if not _is_valid_segment(data):
            continue
        s = data["start"]
        e = data["end"]
        lines.append([(float(s[0]), float(s[1])), (float(e[0]), float(e[1]))])
    return lines


def _walls_by_segment_id(graph: nx.MultiGraph) -> dict[str, dict]:
    """Return a segment_id → edge-data map, filtered to valid segments only.

    Uses the same validity check as _edges_to_lines so downstream consumers
    (bounding_walls loop, _wall_midpoint) can't trip over missing endpoints.
    """
    walls: dict[str, dict] = {}
    for _u, _v, key, data in graph.edges(keys=True, data=True):
        if not _is_valid_segment(data):
            continue
        walls[key] = data
    return walls


def _outward_direction(graph: nx.MultiGraph, node: str) -> tuple[float, float] | None:
    """Unit vector pointing from ``node`` along its single incident wall.

    Only meaningful for degree-1 nodes (endpoint junctions). Returns None if
    the node has no incident edges or the incident edge has zero length.

    The outward direction is node→other_endpoint, i.e. along the wall pointing
    AWAY from ``node``.
    """
    here = graph.nodes[node].get("position")
    if here is None:
        return None
    incident = list(graph.edges(node, data=True))
    if not incident:
        return None
    u, v, _data = incident[0]
    other = v if u == node else u
    other_pos = graph.nodes[other].get("position")
    if other_pos is None:
        return None
    dx = float(other_pos[0]) - float(here[0])
    dy = float(other_pos[1]) - float(here[1])
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    return (dx / n, dy / n)


def _find_virtual_gap_bridges(
    graph: nx.MultiGraph,
    max_distance: float,
    max_angle_drift_deg: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return a list of virtual line segments ((x0, y0), (x1, y1)) that bridge
    near-colinear endpoint pairs — for feeding into ``polygonize()`` only.

    Two endpoints pair-bridge if:
    - Both are degree-1 nodes in ``graph``
    - Their positions are within ``max_distance``
    - Each endpoint's incident wall points "outward toward the other endpoint"
      within ``max_angle_drift_deg``. This filters spurious bridges where two
      unrelated dead-end stubs happen to be close but aren't on a shared wall.

    Each endpoint participates in at most one bridge (nearest neighbor wins).
    """
    # Gather all endpoint nodes with positions
    endpoints: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for node in graph.nodes():
        if graph.degree(node) != 1:
            continue
        pos = graph.nodes[node].get("position")
        if pos is None:
            continue
        direction = _outward_direction(graph, node)
        if direction is None:
            continue
        endpoints.append((node, (float(pos[0]), float(pos[1])), direction))

    if len(endpoints) < 2:
        return []

    cos_threshold = math.cos(math.radians(max_angle_drift_deg))

    # For each endpoint, find the nearest qualifying partner — a classic
    # O(n²) sweep. Endpoint counts in practice are in the hundreds, not
    # thousands, so this is fine.
    best_partner: dict[str, tuple[str, float]] = {}
    for i, (n_a, pos_a, dir_a) in enumerate(endpoints):
        for j, (n_b, pos_b, dir_b) in enumerate(endpoints):
            if i == j:
                continue
            gap_x = pos_b[0] - pos_a[0]
            gap_y = pos_b[1] - pos_a[1]
            gap_len = math.hypot(gap_x, gap_y)
            if gap_len < 1e-9 or gap_len > max_distance:
                continue
            gap_ux, gap_uy = gap_x / gap_len, gap_y / gap_len
            # Each wall should point AWAY from its own endpoint, so the outward
            # direction from A should oppose the gap direction (A→B) and the
            # outward from B should oppose the reverse (B→A). Equivalently,
            # -dir_a · gap_unit ≈ 1 and dir_b · gap_unit ≈ 1.
            colinear_a = -dir_a[0] * gap_ux - dir_a[1] * gap_uy
            colinear_b = dir_b[0] * gap_ux + dir_b[1] * gap_uy
            if colinear_a < cos_threshold or colinear_b < cos_threshold:
                continue
            cur = best_partner.get(n_a)
            if cur is None or gap_len < cur[1]:
                best_partner[n_a] = (n_b, gap_len)

    # Only emit a bridge when both endpoints prefer each other (mutual nearest)
    # to keep bridges unambiguous and avoid over-connecting.
    bridges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[frozenset[str]] = set()
    node_to_pos = {n: p for n, p, _ in endpoints}
    for a, (b, _d) in best_partner.items():
        partner_of_b = best_partner.get(b)
        if partner_of_b is None or partner_of_b[0] != a:
            continue
        key = frozenset({a, b})
        if key in seen:
            continue
        seen.add(key)
        bridges.append((node_to_pos[a], node_to_pos[b]))
    return bridges


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

    # Do NOT require graph connectivity — per the module docstring, real floor-
    # plan wall graphs are almost always disconnected (door openings, drafting
    # gaps, disjoint wings). polygonize() handles multi-component graphs fine;
    # it returns whatever closes cleanly.
    lines = _edges_to_lines(graph)
    if not lines:
        return []

    # Virtual gap closure: real floor plans rarely have fully closed wall loops
    # because doorways/arches/drafting gaps break every room's perimeter. We
    # add near-colinear endpoint bridges to the MultiLineString fed into
    # polygonize() only; the wall graph itself is untouched.
    virtual_bridges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if bool(config.get("room_gap_close_enabled", False)):
        virtual_bridges = _find_virtual_gap_bridges(
            graph,
            max_distance=float(config["room_gap_close_max_distance"]),
            max_angle_drift_deg=float(config["room_gap_close_max_angle_drift_deg"]),
        )
    all_lines = list(lines)
    for a, b in virtual_bridges:
        all_lines.append([a, b])
    bridge_geoms = [LineString([a, b]) for a, b in virtual_bridges]

    merged = unary_union(MultiLineString(all_lines))
    raw_polys = list(polygonize(merged))
    max_aspect = float(config["room_max_aspect_ratio"])
    min_short = float(config["room_min_short_dimension"])
    candidate_polys: list[Polygon] = []
    for p in raw_polys:
        if p.area < min_area:
            continue
        # Use the ORIENTED minimum bounding rectangle, not the axis-aligned
        # bbox. A 45°-rotated 80×8 sliver has an axis-aligned bbox of ~63×63
        # (aspect 1.0) and would escape the filter; its oriented bbox is
        # still 80×8 (aspect 10) and correctly rejected.
        mrr = p.minimum_rotated_rectangle
        corners = list(mrr.exterior.coords)[:4]
        # Two unique edge lengths from the four corners
        edge_a = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
        edge_b = math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
        short = min(edge_a, edge_b)
        long = max(edge_a, edge_b)
        # Reject wall-thickness voids: long thin strips with aspect > N, or
        # polygons whose short dimension is thinner than a typical wall.
        if short < min_short:
            continue
        if short > 1e-9 and (long / short) > max_aspect:
            continue
        candidate_polys.append(p)
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

        # Try fixture-based typing first (if fixtures are available); fall back
        # to label-based typing. Track which classifier fired so rule_triggered
        # can accurately name the source.
        type_source: str | None = None
        room_type, matched_substring = _classify_room_type_from_fixtures(poly, fixtures)
        if room_type is not None:
            type_source = "fixture_match"
        else:
            room_type, matched_substring = _classify_room_type_from_labels(label_texts, type_patterns)
            if matched_substring is not None:
                type_source = "label_match"

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

        # Did this polygon's perimeter rely on any virtual gap bridge? Only
        # bridges that actually lie on or touch the polygon's boundary count —
        # a distance-based check would falsely tag native polygons whenever an
        # unrelated bridge happened to run within tolerance of their perimeter.
        used_virtual = any(
            boundary.intersects(bg) or boundary.touches(bg) for bg in bridge_geoms
        )

        if type_source is not None:
            rule = f"{type_source}:{matched_substring}"
        elif used_virtual:
            rule = "polygon_closure_via_virtual_gap"
        else:
            rule = "polygon_closure"

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
                "rule_triggered": rule,
                "requires_cross_document_validation": False,
            }
        )

    rooms.sort(key=lambda r: r["area"], reverse=True)
    return rooms