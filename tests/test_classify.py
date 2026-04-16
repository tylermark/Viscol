"""Stage 2 tests — property classification."""

from __future__ import annotations

from stages.classify_properties import classify_paths
from tests.fixtures import default_config, make_path


def _extracted(paths, text_blocks=None):
    return {"paths": list(paths), "text_blocks": list(text_blocks or []), "page_size": [600, 400]}


def test_thick_solid_black_line_is_wall_candidate():
    config = default_config()
    p = make_path((0, 0), (100, 0), stroke_width=1.0, stroke_rgb=(0, 0, 0))
    result = classify_paths(_extracted([p]), config)
    assert result["paths"][0]["candidate_type"] == "wall_candidate"
    assert result["paths"][0]["classification_confidence"] >= 0.5


def test_thin_line_without_text_is_annotation():
    config = default_config()
    p = make_path((0, 0), (50, 0), stroke_width=0.2, stroke_rgb=(0, 0, 0))
    result = classify_paths(_extracted([p]), config)
    assert result["paths"][0]["candidate_type"] == "annotation"


def test_thin_line_near_text_is_dimension():
    config = default_config()
    p = make_path((0, 0), (50, 0), stroke_width=0.2, stroke_rgb=(0, 0, 0))
    text = {"text": "5'-0\"", "bbox": [10, -5, 40, 5]}  # close to path midpoint (25, 0)
    result = classify_paths(_extracted([p], [text]), config)
    assert result["paths"][0]["candidate_type"] == "dimension"


def test_dashed_line_is_not_wall_candidate():
    config = default_config()
    p = make_path((0, 0), (100, 0), stroke_width=1.0, stroke_rgb=(0, 0, 0), is_dashed=True)
    result = classify_paths(_extracted([p]), config)
    assert result["paths"][0]["candidate_type"] != "wall_candidate"


def test_light_color_thick_line_is_not_wall():
    config = default_config()
    p = make_path((0, 0), (100, 0), stroke_width=1.0, stroke_rgb=(0.9, 0.9, 0.9))
    result = classify_paths(_extracted([p]), config)
    assert result["paths"][0]["candidate_type"] != "wall_candidate"


def test_classify_never_discards_paths():
    config = default_config()
    paths = [
        make_path((0, 0), (100, 0), stroke_width=1.0),
        make_path((0, 10), (50, 10), stroke_width=0.1),
        make_path((0, 20), (50, 20), stroke_width=0.4),
    ]
    result = classify_paths(_extracted(paths), config)
    assert len(result["paths"]) == 3
    for p in result["paths"]:
        assert "candidate_type" in p
        assert 0.0 <= p["classification_confidence"] <= 1.0
