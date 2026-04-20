"""Unit tests for scripts/render_labeling_ui.py.

We don't drive the browser — we only sanity-check that build_html() produces a
page whose embedded context carries the expected rooms + xref targets, and
that the placeholder was replaced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.render_labeling_ui import _PageGeometry, build_html  # noqa: E402


def _geom(rotation: int = 0, page_w: float = 612.0, page_h: float = 792.0) -> _PageGeometry:
    """A geometry whose image coords equal its rect coords (72 DPI, 1:1)."""
    if rotation in (90, 270):
        mb_w, mb_h = page_h, page_w
        rect_w, rect_h = page_w, page_h
    else:
        mb_w, mb_h = page_w, page_h
        rect_w, rect_h = page_w, page_h
    return _PageGeometry(
        mediabox_width=mb_w,
        mediabox_height=mb_h,
        rect_width=rect_w,
        rect_height=rect_h,
        rotation=rotation,
        image_width=rect_w,
        image_height=rect_h,
    )


def test_build_html_embeds_rooms_and_xrefs():
    doc = {
        "metadata": {"source_pdf": "foo.pdf", "pipeline_version": "0.6.0"},
        "rooms": [
            {
                "room_id": "r-1",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "centroid": [50, 50],
                "area": 10000.0,
                "room_type": "bathroom",
                "room_name": None,
                "room_number": None,
            },
        ],
        "cross_references": [
            {"target_sheet": "A302"},
            {"target_sheet": "F10."},
            {"target_sheet": "A302"},  # dup — dedup expected
        ],
    }
    html = build_html(doc, b"\x89PNG\r\n\x1a\nfakepng", _geom())

    assert "__CONTEXT_JSON__" not in html, "placeholder should be replaced"
    # The context payload is embedded as a single JSON blob. Parse it out.
    marker = "const CTX = "
    start = html.index(marker) + len(marker)
    end = html.index(";", start)
    ctx = json.loads(html[start:end])

    assert len(ctx["rooms"]) == 1
    assert ctx["rooms"][0]["room_id"] == "r-1"
    assert ctx["rooms"][0]["detected_type"] == "bathroom"
    # Polygon should have been converted into an SVG points string
    assert "svg_points" in ctx["rooms"][0]
    assert "," in ctx["rooms"][0]["svg_points"]
    # XRef targets deduped and sorted
    assert ctx["xref_targets"] == ["A302", "F10."]
    # Allowed types list matches the evaluator's expectation
    assert "bathroom" in ctx["allowed_types"]
    assert "unknown" in ctx["allowed_types"]


def test_build_html_handles_plan_with_no_rooms():
    doc = {
        "metadata": {"source_pdf": "bar.pdf", "pipeline_version": "0.6.0"},
        "rooms": [],
        "cross_references": [],
    }
    html = build_html(doc, b"\x89PNG\r\n\x1a\nfakepng", _geom(page_w=100, page_h=100))
    marker = "const CTX = "
    start = html.index(marker) + len(marker)
    end = html.index(";", start)
    ctx = json.loads(html[start:end])
    assert ctx["rooms"] == []
    assert ctx["xref_targets"] == []


def test_rotation_transform_maps_mediabox_corners_to_rect_corners():
    """Ground-truth validation of the PDF /Rotate handling.

    Pipeline stores points using `schema_y = page.rect.height - mediabox_y`
    (that mixing is what extract_paths.py does today). For a 270°-rotated
    page with mediabox=1728x2592 and rect=2592x1728, mediabox corner
    (946, 652) should end up at rect pixel (652, 782) — verified against
    PyMuPDF's native text extraction for the Shattuck "2-BR" label.
    """
    geom = _PageGeometry(
        mediabox_width=1728.0,
        mediabox_height=2592.0,
        rect_width=2592.0,
        rect_height=1728.0,
        rotation=270,
        image_width=2592.0,   # 1:1 at 72 DPI
        image_height=1728.0,
    )
    # What the pipeline would have stored for native (946, 652):
    schema_x = 946.0
    schema_y = 1728.0 - 652.0  # = 1076
    px, py = geom.to_image_px(schema_x, schema_y)
    assert abs(px - 652.0) < 0.01, f"expected rect_x=652, got {px}"
    assert abs(py - 782.0) < 0.01, f"expected rect_y=782, got {py}"


def test_rotation_zero_is_identity():
    geom = _geom(rotation=0, page_w=600, page_h=800)
    # Pipeline stored this point as schema (100, 700) meaning mediabox (100, 100).
    # With rotation=0, image px should be (100, 100).
    px, py = geom.to_image_px(100.0, 700.0)
    assert abs(px - 100.0) < 0.01
    assert abs(py - 100.0) < 0.01


def test_rotation_90_swaps_and_reflects():
    # 90° CW rotation: mediabox (W=800, H=600) in portrait → rect (600, 800)? Wait:
    # if page is rotated 90° CW for display, rect dims are (H_m, W_m).
    # For mediabox (800, 600), rect = (600, 800).
    geom = _PageGeometry(
        mediabox_width=800.0,
        mediabox_height=600.0,
        rect_width=600.0,
        rect_height=800.0,
        rotation=90,
        image_width=600.0,
        image_height=800.0,
    )
    # Mediabox top-right (800, 0) should map to rect bottom-right (600, 800).
    # Pipeline schema of (800, 0) under a y-flip using page.rect.height=800:
    #   schema = (800, 800 - 0) = (800, 800)
    px, py = geom.to_image_px(800.0, 800.0)
    assert abs(px - 600.0) < 0.01
    assert abs(py - 800.0) < 0.01


def test_build_html_escapes_closing_script_tag_in_payload():
    """A text region containing '</script>' must not terminate the embedded
    <script> tag. json.dumps doesn't escape this by default, so we do it
    manually after serialization."""
    doc = {
        "metadata": {"source_pdf": "x.pdf", "pipeline_version": "0.6.0"},
        "rooms": [
            {
                "room_id": "r-xss",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "centroid": [5, 5],
                "area": 100,
                "room_type": "unknown",
                "room_name": "evil </script><img src=x onerror=alert(1)> name",
                "room_number": None,
            }
        ],
        "cross_references": [],
    }
    html = build_html(doc, b"\x89PNG\r\n\x1a\nfakepng", _geom())
    # Literal "</script>" must not appear inside the script payload — only the
    # real closing tag at the end of the script block is allowed.
    script_close_count = html.count("</script>")
    assert script_close_count == 1, (
        f"expected exactly one literal </script> (the real tag), "
        f"got {script_close_count}"
    )


def test_build_html_raises_on_missing_polygon():
    """A room entry without a polygon should fail loudly, not render nothing."""
    import pytest
    doc = {
        "metadata": {},
        "rooms": [{"room_id": "bad", "centroid": [0, 0], "area": 10}],
        "cross_references": [],
    }
    with pytest.raises(ValueError, match="bad"):
        build_html(doc, b"\x89PNG", _geom())


def test_polygon_rejects_nan_coordinate():
    """NaN/Inf in polygon coords must raise, not silently render invisible."""
    import math
    import pytest
    doc = {
        "metadata": {},
        "rooms": [{
            "room_id": "nan-room",
            "polygon": [[0, 0], [10, 0], [10, math.nan]],
            "centroid": [5, 5],
            "area": 100,
            "room_type": "unknown",
        }],
        "cross_references": [],
    }
    with pytest.raises(ValueError, match="finite"):
        build_html(doc, b"\x89PNG", _geom())


# --------------------------------- download → eval round-trip stability

def test_downloaded_yaml_round_trips_through_eval_loader():
    """Regression guard: a YAML produced in the format the UI downloader
    emits (see toYaml() in the embedded JS) must load cleanly via
    yaml.safe_load — the same loader scripts/eval_coordinator_tasks.py uses.
    Room IDs that look numeric, multi-word room names, and all three
    cross_references sublists must round-trip intact."""
    import yaml
    # This mimics the exact shape toYaml() emits for a plan with one
    # detected room (with a numeric-looking room_number), one missed
    # room, and xref sections with both kept and missed targets.
    # All string values are double-quoted (yamlEscape always quotes),
    # matching the JS side's "JSON.stringify(String(v))" serialization.
    downloaded_yaml = '''plan_stem: "test-plan"
source_pdf: "test.pdf"
pipeline_version: "0.6.0"
rooms:
- room_id: "r-abc-123"
  centroid:
  - 50.0
  - 50.0
  area: 1234.5
  detected_type: "unknown"
  room_name: "Unit 2.1a"
  room_number: "203"
  correct_type: "unit"
missed_rooms:
- centroid:
  - 999.0
  - 999.0
  correct_type: "kitchen"
cross_references:
  detected_targets:
  - "A302"
  - "F10."
  valid_targets:
  - "A302"
  missed_targets:
  - "A401"
'''
    parsed = yaml.safe_load(downloaded_yaml)

    assert parsed["plan_stem"] == "test-plan"
    assert len(parsed["rooms"]) == 1
    # room_id stays a string even though a bare "203" would normally coerce
    # to int — because yamlEscape always quotes.
    assert parsed["rooms"][0]["room_id"] == "r-abc-123"
    assert isinstance(parsed["rooms"][0]["room_number"], str)
    assert parsed["rooms"][0]["room_number"] == "203"
    assert parsed["rooms"][0]["correct_type"] == "unit"
    assert parsed["rooms"][0]["centroid"] == [50.0, 50.0]

    assert parsed["missed_rooms"] == [{
        "centroid": [999.0, 999.0],
        "correct_type": "kitchen",
    }]

    xref = parsed["cross_references"]
    assert xref["detected_targets"] == ["A302", "F10."]
    assert xref["valid_targets"] == ["A302"]
    assert xref["missed_targets"] == ["A401"]


def test_inverse_transform_round_trips():
    """pixelToSchema must invert to_image_px. Forward then back should
    give back the original schema point (within float epsilon) for every
    supported rotation. This is the Python mirror of the JS inverse the
    UI uses for click-to-place missed rooms."""
    cases = [
        # rotation, mediabox (w, h), rect (w, h), image (w, h)
        (0,   (100, 200), (100, 200), (200, 400)),
        (90,  (200, 100), (100, 200), (200, 400)),  # mediabox portrait rotated
        (180, (100, 200), (100, 200), (200, 400)),
        (270, (200, 100), (100, 200), (200, 400)),
    ]
    for rot, (mbw, mbh), (rw, rh), (iw, ih) in cases:
        g = _PageGeometry(mbw, mbh, rw, rh, rot, iw, ih)
        for sx, sy in [(10.0, 20.0), (50.0, 100.0), (0.0, 0.0)]:
            px, py = g.to_image_px(sx, sy)
            rx, ry = g.from_image_px(px, py)
            assert abs(rx - sx) < 1e-6, f"rot={rot} x mismatch: {sx}→{rx}"
            assert abs(ry - sy) < 1e-6, f"rot={rot} y mismatch: {sy}→{ry}"
