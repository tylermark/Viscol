"""Stage 10 — Graph assembly.

Emit the final §4 structured-graph JSON from the outputs of all prior stages.
This is the canonical FM-training-ready representation.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import networkx as nx


PIPELINE_VERSION = "0.6.0"
COORDINATE_SYSTEM = "pdf_points_bottom_left_origin"


def _segment_record(start_junction_id: str, end_junction_id: str, data: dict) -> dict:
    s = data["start"]
    e = data["end"]
    length = float(math.hypot(e[0] - s[0], e[1] - s[1]))
    angle = math.degrees(math.atan2(e[1] - s[1], e[0] - s[0])) % 180.0
    return {
        "segment_id": data["segment_id"],
        "geometry": {
            "start": [float(s[0]), float(s[1])],
            "end": [float(e[0]), float(e[1])],
            "centerline_length": length,
            "angle_degrees": float(angle),
            "thickness": float(data.get("thickness", 0.0)),
        },
        "visual_properties": {
            "line_weight": float(data.get("stroke_width", 0.0)),
            "color_rgb": list(data.get("color_rgb") or [0.0, 0.0, 0.0]),
            "is_dashed": bool(data.get("is_dashed", False)),
            "layer_name": data.get("layer_name"),
        },
        "topology": {
            "start_junction_id": start_junction_id,
            "end_junction_id": end_junction_id,
            "start_junction_type": data.get("start_junction_type"),
            "end_junction_type": data.get("end_junction_type"),
            "connected_segment_ids": list(data.get("connected_segment_ids") or []),
            "adjacent_room_ids": list(data.get("adjacent_room_ids") or []),
        },
        "semantic": {
            "functional_role": data.get("functional_role", "unknown"),
            "confidence": float(data.get("confidence", 0.0)),
            "rule_triggered": data.get("rule_triggered"),
            "requires_cross_document_validation": bool(
                data.get("requires_cross_document_validation", False)
            ),
        },
    }


def _strip_internal_keys(room: dict) -> dict:
    """Remove leading-underscore working fields before emission."""
    return {k: v for k, v in room.items() if not k.startswith("_")}


def assemble(
    *,
    source_pdf_name: str,
    page_size: tuple[float, float],
    graph: nx.MultiGraph,
    junctions: list[dict],
    rooms: list[dict],
    openings: list[dict],
    text_regions: list[dict],
    grid_lines: list[dict],
    cross_references: list[dict],
) -> dict[str, Any]:
    segments = [
        _segment_record(u, v, data)
        for u, v, _, data in graph.edges(keys=True, data=True)
    ]
    segments.sort(key=lambda s: s["segment_id"])

    junctions_out = [
        {
            "junction_id": j["junction_id"],
            "position": [float(j["position"][0]), float(j["position"][1])],
            "junction_type": j["junction_type"],
            "connected_segment_ids": list(j.get("connected_segment_ids") or []),
        }
        for j in junctions
    ]
    junctions_out.sort(key=lambda j: j["junction_id"])

    rooms_out = [_strip_internal_keys(r) for r in rooms]

    # Set enclosing_room_id on text regions that fall inside any room polygon.
    # Also populate room.text_labels with real text_region_ids now that those IDs exist.
    from shapely.geometry import Point, Polygon as ShPoly  # local import to avoid top-level shapely load
    poly_by_room = {r["room_id"]: ShPoly(r["polygon"]) for r in rooms_out if len(r.get("polygon") or []) >= 3}
    for region in text_regions:
        bbox = region.get("bbox") or [0, 0, 0, 0]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for room_id, poly in poly_by_room.items():
            if poly.contains(Point(cx, cy)):
                region["enclosing_room_id"] = room_id
                break

    for room in rooms_out:
        room["text_labels"] = [
            r["text_region_id"]
            for r in text_regions
            if r.get("enclosing_room_id") == room["room_id"]
        ]

    doc = {
        "metadata": {
            "source_pdf": source_pdf_name,
            "extraction_date": date.today().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "coordinate_system": COORDINATE_SYSTEM,
            "page_size": [float(page_size[0]), float(page_size[1])],
            "entity_counts": {
                "rooms": len(rooms_out),
                "walls": len(segments),
                "openings": len(openings),
                "text_regions": len(text_regions),
                "grid_lines": len(grid_lines),
                "junctions": len(junctions_out),
            },
        },
        "rooms": rooms_out,
        "walls": segments,
        "openings": openings,
        "text_regions": text_regions,
        "grid_lines": grid_lines,
        "junctions": junctions_out,
        "cross_references": cross_references,
    }
    return doc
