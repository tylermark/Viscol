"""Stage 3 tests — wall detection from parallel pairs."""

from __future__ import annotations

from stages.classify_properties import classify_paths
from stages.detect_walls import detect_walls
from tests.fixtures import default_config, make_path, parallel_wall_pair


def _pipeline(paths):
    config = default_config()
    extracted = {"paths": list(paths), "text_blocks": [], "page_size": [600, 400]}
    classify_paths(extracted, config)
    return detect_walls(extracted, config), config


def test_two_parallel_lines_produce_one_centerline():
    walls, _ = _pipeline(parallel_wall_pair((10, 100), length=100.0, thickness=6.0))
    assert len(walls) == 1
    w = walls[0]
    assert abs(w["length"] - 100.0) < 1e-3
    assert abs(w["thickness"] - 6.0) < 1e-3
    assert abs(w["start"][1] - 103.0) < 1e-3
    assert abs(w["end"][1] - 103.0) < 1e-3
    assert "parallel_pairing" in w["rules_passed"]
    assert "thickness_in_range" in w["rules_passed"]
    assert "orthogonality" in w["rules_passed"]


def test_non_parallel_pair_is_rejected():
    p1 = make_path((0, 0), (100, 0), stroke_width=1.0)
    p2 = make_path((0, 6), (100, 30), stroke_width=1.0)  # diverging — far from parallel
    walls, _ = _pipeline([p1, p2])
    assert walls == []


def test_too_thick_pair_is_rejected():
    # Thickness 200 is > wall_max_thickness (40)
    p1 = make_path((0, 0), (100, 0), stroke_width=1.0)
    p2 = make_path((0, 200), (100, 200), stroke_width=1.0)
    walls, _ = _pipeline([p1, p2])
    assert walls == []


def test_diagonal_pair_is_flagged_non_orthogonal():
    # 45 degree parallel pair, thickness 6
    dx = 100.0 / (2 ** 0.5)
    dy = 100.0 / (2 ** 0.5)
    offset_x, offset_y = -6.0 / (2 ** 0.5), 6.0 / (2 ** 0.5)  # perpendicular offset of 6
    p1 = make_path((0, 0), (dx, dy), stroke_width=1.0)
    p2 = make_path((offset_x, offset_y), (dx + offset_x, dy + offset_y), stroke_width=1.0)
    walls, _ = _pipeline([p1, p2])
    assert len(walls) == 1
    assert "orthogonality" in walls[0]["rules_failed"]


def test_short_overlap_pair_is_rejected():
    # Segments barely overlap — below min_overlap_ratio (default 0.3)
    p1 = make_path((0, 0), (100, 0), stroke_width=1.0)
    p2 = make_path((95, 6), (200, 6), stroke_width=1.0)
    walls, _ = _pipeline([p1, p2])
    assert walls == []
