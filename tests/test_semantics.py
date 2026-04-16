"""Stage 5 tests — semantic role assignment."""

from __future__ import annotations

from stages.assign_semantics import assign_semantics
from stages.build_topology import build_topology
from tests.fixtures import default_config, make_wall_record


def _build(walls):
    return build_topology(walls, default_config())


def _roles(graph):
    return [data["functional_role"] for _, _, _, data in graph.edges(keys=True, data=True)]


def test_rectangle_walls_are_all_exterior():
    walls = [
        make_wall_record((0, 0), (300, 0)),
        make_wall_record((300, 0), (300, 300)),
        make_wall_record((300, 300), (0, 300)),
        make_wall_record((0, 300), (0, 0)),
    ]
    graph, _ = _build(walls)
    assign_semantics(graph, [], default_config())
    roles = _roles(graph)
    assert roles.count("exterior") == 4


def test_long_internal_wall_through_t_junctions_is_demising():
    # Rectangle 500x300, with interior horizontal wall crossing at y=150 from (0,150) to (500,150)
    # The interior wall touches both left and right exterior walls at their interiors (T-junctions).
    walls = [
        make_wall_record((0, 0), (500, 0)),
        make_wall_record((500, 0), (500, 300)),
        make_wall_record((500, 300), (0, 300)),
        make_wall_record((0, 300), (0, 0)),
        make_wall_record((0, 150), (500, 150)),
    ]
    graph, _ = _build(walls)
    assign_semantics(graph, [], default_config())
    roles = _roles(graph)
    # Exterior walls get split by the interior wall at T junctions → 2 pieces each for left/right
    # The interior wall itself should be either demising or bearing (both long + through junctions).
    # First-match-wins favors demising.
    assert roles.count("demising") >= 1


def test_wall_near_bathroom_label_is_wet_wall():
    # Small room whose interior wall is adjacent to a "BATHROOM" text label
    walls = [
        make_wall_record((0, 0), (100, 0)),
        make_wall_record((100, 0), (100, 100)),
        make_wall_record((100, 100), (0, 100)),
        make_wall_record((0, 100), (0, 0)),
        make_wall_record((50, 0), (50, 100)),  # interior partition between two zones
    ]
    graph, _ = _build(walls)
    text_blocks = [{"text": "BATHROOM", "bbox": [60, 40, 95, 60]}]  # right side of interior wall
    assign_semantics(graph, text_blocks, default_config())
    # The interior partition wall (shorter than demising min) should match wet_wall
    # We don't control which wall is "interior" precisely, but at least one wall should be flagged
    wet_count = 0
    cross_doc_ok = True
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        if data["functional_role"] == "wet_wall":
            wet_count += 1
            if not data["requires_cross_document_validation"]:
                cross_doc_ok = False
    assert wet_count >= 1
    assert cross_doc_ok, "wet_wall must always carry requires_cross_document_validation=True"


def test_default_interior_partition_for_isolated_wall():
    walls = [make_wall_record((10, 10), (60, 10))]  # short, not exterior, not near anything
    graph, _ = _build(walls)
    assign_semantics(graph, [], default_config())
    roles = _roles(graph)
    assert roles == ["interior_partition"] or roles == ["unknown"]


def test_cross_doc_flag_invariant_on_bearing_wall():
    # Long continuous path of 4 short walls through corner junctions
    walls = [
        make_wall_record((0, 0), (200, 0)),
        make_wall_record((200, 0), (200, 200)),
        make_wall_record((200, 200), (400, 200)),
        make_wall_record((400, 200), (400, 0)),
    ]
    graph, _ = _build(walls)
    assign_semantics(graph, [], default_config())
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        if data["functional_role"] == "bearing_wall":
            assert data["requires_cross_document_validation"] is True
