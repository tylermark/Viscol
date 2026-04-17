"""Stage 3 tests — wall detection from parallel pairs."""

from __future__ import annotations

from stages.classify_properties import classify_paths
from stages.detect_walls import detect_walls
from tests.fixtures import default_config, make_path, parallel_wall_pair


def _pipeline(paths):
    """Return (kept_walls, dropped_by_thickness, config)."""
    config = default_config()
    extracted = {"paths": list(paths), "text_blocks": [], "page_size": [600, 400]}
    classify_paths(extracted, config)
    kept, dropped = detect_walls(extracted, config)
    return kept, dropped, config


def test_two_parallel_lines_produce_one_centerline():
    walls, _dropped, _cfg = _pipeline(parallel_wall_pair((10, 100), length=100.0, thickness=6.0))
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
    walls, _dropped, _cfg = _pipeline([p1, p2])
    assert walls == []


def test_too_thick_pair_is_rejected():
    # Thickness 200 is > wall_max_thickness (40)
    p1 = make_path((0, 0), (100, 0), stroke_width=1.0)
    p2 = make_path((0, 200), (100, 200), stroke_width=1.0)
    walls, _dropped, _cfg = _pipeline([p1, p2])
    assert walls == []


def test_diagonal_pair_is_flagged_non_orthogonal():
    # 45 degree parallel pair, thickness 6
    dx = 100.0 / (2 ** 0.5)
    dy = 100.0 / (2 ** 0.5)
    offset_x, offset_y = -6.0 / (2 ** 0.5), 6.0 / (2 ** 0.5)  # perpendicular offset of 6
    p1 = make_path((0, 0), (dx, dy), stroke_width=1.0)
    p2 = make_path((offset_x, offset_y), (dx + offset_x, dy + offset_y), stroke_width=1.0)
    walls, _dropped, _cfg = _pipeline([p1, p2])
    assert len(walls) == 1
    assert "orthogonality" in walls[0]["rules_failed"]


def test_short_overlap_pair_is_rejected():
    # Segments barely overlap — below min_overlap_ratio (default 0.3)
    p1 = make_path((0, 0), (100, 0), stroke_width=1.0)
    p2 = make_path((95, 6), (200, 6), stroke_width=1.0)
    walls, _dropped, _cfg = _pipeline([p1, p2])
    assert walls == []


# --- v0.3.0 thickness clustering tests ---


def test_thickness_clustering_rejects_scattered_noise():
    """80 wall pairs tightly at ~6pt + 20 scattered-thickness noise pairs (no second peak) → noise dropped."""
    from stages.detect_walls import _cluster_by_thickness

    config = default_config()
    walls = []
    for i in range(80):
        walls.append({"thickness": 6.0 + (i % 3) * 0.1, "segment_id": f"w{i}"})
    # Scatter 20 "noise" pairs across a wide range so they don't form a peak.
    noise_values = [1.2, 2.3, 3.7, 4.1, 11.5, 14.2, 18.3, 22.1, 27.5, 33.0,
                    35.2, 37.9, 2.8, 12.3, 16.4, 19.8, 24.6, 29.1, 31.5, 38.4]
    for i, t in enumerate(noise_values):
        walls.append({"thickness": t, "segment_id": f"n{i}"})
    kept, dropped = _cluster_by_thickness(walls, config)
    assert len(kept) == 80
    assert len(dropped) == 20
    assert all("thickness_cluster_outlier" in w["rules_failed"] for w in dropped)


def test_thickness_clustering_bimodal_keeps_two_modes():
    """External walls ~10pt + internal walls ~6pt + noise → both modes kept, noise dropped."""
    from stages.detect_walls import _cluster_by_thickness

    config = default_config()
    walls = []
    for i in range(50):
        walls.append({"thickness": 10.0 + (i % 3) * 0.1, "segment_id": f"e{i}"})
    for i in range(40):
        walls.append({"thickness": 6.0 + (i % 3) * 0.1, "segment_id": f"i{i}"})
    for i in range(10):
        walls.append({"thickness": 22.0 + i * 0.2, "segment_id": f"n{i}"})  # spread noise
    kept, dropped = _cluster_by_thickness(walls, config)
    thicknesses = sorted({round(w["thickness"]) for w in kept})
    assert 6 in thicknesses
    assert 10 in thicknesses
    assert 22 not in thicknesses
    assert len(kept) == 90
    assert len(dropped) == 10


def test_thickness_clustering_skipped_below_min_candidates():
    """Fewer than min_candidates (default 10) → fallback: no clustering, everything kept."""
    from stages.detect_walls import _cluster_by_thickness

    config = default_config()
    walls = [
        {"thickness": 6.0, "segment_id": "a"},
        {"thickness": 1.5, "segment_id": "b"},  # would be rejected if clustering ran
        {"thickness": 40.0, "segment_id": "c"},
    ]
    kept, dropped = _cluster_by_thickness(walls, config)
    assert len(kept) == 3
    assert dropped == []
