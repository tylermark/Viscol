"""Stage 4 tests — topology construction."""

from __future__ import annotations

from stages.build_topology import build_topology
from tests.fixtures import default_config, make_wall_record


def _junction_types(junctions):
    return sorted(j["junction_type"] for j in junctions)


def test_rectangle_has_four_corners_and_four_walls():
    walls = [
        make_wall_record((0, 0), (100, 0)),
        make_wall_record((100, 0), (100, 100)),
        make_wall_record((100, 100), (0, 100)),
        make_wall_record((0, 100), (0, 0)),
    ]
    graph, junctions = build_topology(walls, default_config())
    assert len(junctions) == 4
    assert _junction_types(junctions) == ["corner", "corner", "corner", "corner"]
    assert graph.number_of_edges() == 4


def test_t_junction_from_endpoint_on_body_splits_host_wall():
    # Horizontal wall from (0,0) to (100,0). Vertical wall from (50,0) to (50,50).
    # The vertical wall's endpoint at (50, 0) should split the horizontal wall at 50.
    walls = [
        make_wall_record((0, 0), (100, 0)),
        make_wall_record((50, 0), (50, 50)),
    ]
    graph, junctions = build_topology(walls, default_config())
    # 3 junctions total: (0,0), (100,0), (50,0), (50,50) -- one of these is the T
    assert len(junctions) == 4
    types = _junction_types(junctions)
    assert types.count("t-junction") == 1
    assert types.count("endpoint") == 3
    # Horizontal wall split into 2 edges, plus the vertical = 3 edges total.
    assert graph.number_of_edges() == 3


def test_endpoints_without_neighbors_are_endpoint_type():
    walls = [make_wall_record((0, 0), (50, 0))]
    _graph, junctions = build_topology(walls, default_config())
    assert len(junctions) == 2
    assert _junction_types(junctions) == ["endpoint", "endpoint"]


def test_connected_segment_ids_are_populated():
    walls = [
        make_wall_record((0, 0), (100, 0)),
        make_wall_record((100, 0), (100, 100)),
    ]
    graph, junctions = build_topology(walls, default_config())
    corner = next(j for j in junctions if j["junction_type"] == "corner")
    assert len(corner["connected_segment_ids"]) == 2
    # Walls should reference each other in connected_segment_ids
    edge_data = list(graph.edges(keys=True, data=True))
    for _u, _v, _k, data in edge_data:
        assert len(data["connected_segment_ids"]) == 1
