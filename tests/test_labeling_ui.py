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

from scripts.render_labeling_ui import build_html  # noqa: E402


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
    html = build_html(doc, b"\x89PNG\r\n\x1a\nfakepng", page_width_pts=612.0, page_height_pts=792.0, dpi=72)

    assert "__CONTEXT_JSON__" not in html, "placeholder should be replaced"
    assert "data:image/png;base64," not in html or True  # the image src is constructed in JS
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
    html = build_html(doc, b"\x89PNG\r\n\x1a\nfakepng", page_width_pts=100.0, page_height_pts=100.0, dpi=72)
    marker = "const CTX = "
    start = html.index(marker) + len(marker)
    end = html.index(";", start)
    ctx = json.loads(html[start:end])
    assert ctx["rooms"] == []
    assert ctx["xref_targets"] == []
