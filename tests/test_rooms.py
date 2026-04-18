"""Stage 5 tests — room detection and virtual gap closure."""

from __future__ import annotations

from stages.build_topology import build_topology
from stages.detect_rooms import detect_rooms, _find_virtual_gap_bridges
from tests.fixtures import default_config, make_wall_record


def _build_graph(walls):
    graph, _j, _dropped = build_topology(walls, default_config())
    return graph


def _closed_rectangle_with_door_gap() -> list[dict]:
    """A 400x300 room whose bottom wall has a 40pt doorway gap.

    Bottom wall is split into two pieces at x=180 and x=220 with a gap between.
    Without gap closure, polygonize() returns nothing. With gap closure it
    should recover the one rectangle.
    """
    return [
        make_wall_record((0, 0), (180, 0)),          # bottom-left segment, ends at door
        make_wall_record((220, 0), (400, 0)),        # bottom-right segment, starts at door
        make_wall_record((400, 0), (400, 300)),      # right
        make_wall_record((400, 300), (0, 300)),      # top
        make_wall_record((0, 300), (0, 0)),          # left
    ]


def _config(enabled: bool) -> dict:
    cfg = default_config()
    cfg["room_gap_close_enabled"] = enabled
    return cfg


def test_rectangle_with_door_gap_not_closed_without_bridge():
    """Baseline: with gap closure disabled, a door-gap room is not polygonized."""
    graph = _build_graph(_closed_rectangle_with_door_gap())
    rooms = detect_rooms(graph, text_blocks=[], config=_config(enabled=False))
    assert rooms == [], f"expected no rooms without gap closure, got {len(rooms)}"


def test_rectangle_with_door_gap_closes_with_virtual_bridge():
    """With gap closure enabled, the door-gap rectangle polygonizes to one room."""
    graph = _build_graph(_closed_rectangle_with_door_gap())
    rooms = detect_rooms(graph, text_blocks=[], config=_config(enabled=True))
    assert len(rooms) == 1, f"expected 1 room, got {len(rooms)}: {rooms}"
    room = rooms[0]
    assert room["rule_triggered"] == "polygon_closure_via_virtual_gap"
    # Real walls still bound the room; the virtual bridge isn't a segment_id.
    # The 5 real walls should all appear in bounding_walls.
    assert len(room["bounding_walls"]) == 5


def test_virtual_bridge_respects_distance_threshold():
    """A gap wider than room_gap_close_max_distance is not bridged."""
    walls = [
        make_wall_record((0, 0), (180, 0)),
        make_wall_record((400, 0), (580, 0)),        # gap of 220pt > 60pt default
        make_wall_record((580, 0), (580, 300)),
        make_wall_record((580, 300), (0, 300)),
        make_wall_record((0, 300), (0, 0)),
    ]
    graph = _build_graph(walls)
    cfg = _config(enabled=True)
    # Keep the threshold at the default (60pt); the 220pt gap should NOT close.
    rooms = detect_rooms(graph, text_blocks=[], config=cfg)
    assert rooms == [], f"expected no rooms at 220pt gap, got {len(rooms)}"


def test_virtual_bridge_respects_colinearity_filter():
    """Two loose stub endpoints that are close but perpendicular should not be bridged."""
    walls = [
        make_wall_record((0, 0), (100, 0)),          # horizontal stub ending at (100, 0)
        make_wall_record((130, 30), (130, 130)),     # vertical stub starting at (130, 30)
    ]
    graph = _build_graph(walls)
    # Distance from (100,0) to (130,30) is ~42pt — under the 60pt threshold —
    # but the walls are perpendicular, not colinear. Bridge list should be empty.
    bridges = _find_virtual_gap_bridges(graph, max_distance=60.0, max_angle_drift_deg=20.0)
    assert bridges == [], f"expected no colinear bridges, got {bridges}"


def test_rotated_sliver_rejected_by_oriented_filter():
    """A 45°-rotated 80×8 sliver has axis-aligned bbox ~63×63 (aspect 1.0),
    but its oriented bounding rectangle is still 80×8 (aspect 10). The filter
    must use the oriented bbox to catch it."""
    import math as _math
    # 80-long, 8-thick rectangle rotated 45° around origin
    cos45 = sin45 = _math.cos(_math.radians(45))
    def rot(x, y):
        return (x * cos45 - y * sin45, x * sin45 + y * cos45)
    c1 = rot(0, 0)
    c2 = rot(80, 0)
    c3 = rot(80, 8)
    c4 = rot(0, 8)
    walls = [
        make_wall_record(c1, c2),
        make_wall_record(c2, c3),
        make_wall_record(c3, c4),
        make_wall_record(c4, c1),
    ]
    graph = _build_graph(walls)
    rooms = detect_rooms(graph, text_blocks=[], config=_config(enabled=False))
    assert rooms == [], f"rotated sliver should be rejected, got {rooms}"


def test_sliver_polygon_rejected_by_aspect_filter():
    """A long thin wall-thickness void (80×8) should be rejected as a sliver."""
    # A 80×8 rectangle — exactly the "wall thickness void" shape we saw on real plans
    walls = [
        make_wall_record((0, 0), (80, 0)),          # bottom face
        make_wall_record((80, 0), (80, 8)),         # right endcap
        make_wall_record((80, 8), (0, 8)),          # top face
        make_wall_record((0, 8), (0, 0)),           # left endcap
    ]
    graph = _build_graph(walls)
    cfg = _config(enabled=False)  # no gap closure needed — rectangle is already closed
    rooms = detect_rooms(graph, text_blocks=[], config=cfg)
    # Polygonize would produce one 80×8 polygon with area 640; the aspect filter
    # rejects it (aspect = 10 > 5) and the short-dim filter rejects it (8 < 15).
    assert rooms == [], f"expected sliver rejection, got {rooms}"


def test_room_type_via_label_after_gap_closure():
    """A gap-closed room containing a BATHROOM label gets room_type='bathroom'."""
    graph = _build_graph(_closed_rectangle_with_door_gap())
    # Centroid of a 400x300 rectangle at origin is (200, 150)
    text_blocks = [
        {"text": "BATHROOM", "bbox": [190, 140, 230, 160]},
    ]
    rooms = detect_rooms(graph, text_blocks=text_blocks, config=_config(enabled=True))
    assert len(rooms) == 1
    assert rooms[0]["room_type"] == "bathroom"
    # The rule_triggered prefers label_match over polygon_closure_via_virtual_gap
    # because the label classification is a stronger signal.
    assert rooms[0]["rule_triggered"].startswith("label_match:")
