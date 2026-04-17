"""Schema validator tests — v0.6.0 multi-entity graph schema."""

from __future__ import annotations

import copy

from schema_validator import validate


def _valid_doc() -> dict:
    return {
        "metadata": {
            "source_pdf": "x.pdf",
            "extraction_date": "2026-04-16",
            "pipeline_version": "0.6.0",
            "coordinate_system": "pdf_points_bottom_left_origin",
            "page_size": [600.0, 400.0],
            "entity_counts": {
                "rooms": 1,
                "walls": 1,
                "openings": 0,
                "text_regions": 1,
                "grid_lines": 0,
                "junctions": 2,
                "cross_references": 0,
            },
        },
        "rooms": [
            {
                "room_id": "room-1",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "area": 10000.0,
                "centroid": [50, 50],
                "bounding_walls": ["seg-1"],
                "text_labels": ["txt-1"],
                "room_name": "Bathroom",
                "room_number": "101",
                "room_type": "bathroom",
                "rule_triggered": "label_match:BATH",
                "requires_cross_document_validation": False,
            }
        ],
        "walls": [
            {
                "segment_id": "seg-1",
                "geometry": {
                    "start": [0, 0],
                    "end": [100, 0],
                    "centerline_length": 100.0,
                    "angle_degrees": 0.0,
                    "thickness": 6.0,
                },
                "visual_properties": {
                    "line_weight": 1.0,
                    "color_rgb": [0, 0, 0],
                    "is_dashed": False,
                    "layer_name": None,
                },
                "topology": {
                    "start_junction_id": "j-1",
                    "end_junction_id": "j-2",
                    "start_junction_type": "endpoint",
                    "end_junction_type": "endpoint",
                    "connected_segment_ids": [],
                    "adjacent_room_ids": ["room-1"],
                },
                "semantic": {
                    "functional_role": "interior_partition",
                    "confidence": 0.6,
                    "rule_triggered": "default_interior",
                    "requires_cross_document_validation": False,
                },
            }
        ],
        "openings": [],
        "text_regions": [
            {
                "text_region_id": "txt-1",
                "text": "BATHROOM",
                "bbox": [20, 20, 80, 40],
                "classification": "room_label",
                "references": [],
                "enclosing_room_id": "room-1",
                "linked_entity_ids": [],
            }
        ],
        "grid_lines": [],
        "junctions": [
            {
                "junction_id": "j-1",
                "position": [0, 0],
                "junction_type": "endpoint",
                "connected_segment_ids": ["seg-1"],
            },
            {
                "junction_id": "j-2",
                "position": [100, 0],
                "junction_type": "endpoint",
                "connected_segment_ids": ["seg-1"],
            },
        ],
        "cross_references": [],
    }


def test_valid_doc_passes():
    errs = validate(_valid_doc())
    assert errs == [], f"unexpected errors: {errs}"


def test_missing_top_level_key_fails():
    doc = _valid_doc()
    del doc["rooms"]
    errs = validate(doc)
    assert any("rooms" in e for e in errs)


def test_wet_wall_without_cross_doc_flag_fails():
    doc = _valid_doc()
    doc["walls"][0]["semantic"]["functional_role"] = "wet_wall"
    doc["walls"][0]["semantic"]["requires_cross_document_validation"] = False
    errs = validate(doc)
    assert any("requires_cross_document_validation" in e for e in errs)


def test_bearing_wall_without_cross_doc_flag_fails():
    doc = _valid_doc()
    doc["walls"][0]["semantic"]["functional_role"] = "bearing_wall"
    doc["walls"][0]["semantic"]["requires_cross_document_validation"] = False
    errs = validate(doc)
    assert any("requires_cross_document_validation" in e for e in errs)


def test_unknown_functional_role_fails():
    doc = _valid_doc()
    doc["walls"][0]["semantic"]["functional_role"] = "not_a_role"
    errs = validate(doc)
    assert any("functional_role" in e for e in errs)


def test_unknown_room_type_fails():
    doc = _valid_doc()
    doc["rooms"][0]["room_type"] = "not_a_type"
    errs = validate(doc)
    assert any("room_type" in e for e in errs)


def test_unknown_junction_type_fails():
    doc = _valid_doc()
    doc["junctions"][0]["junction_type"] = "nope"
    errs = validate(doc)
    assert any("junction_type" in e for e in errs)


def test_duplicate_segment_id_fails():
    doc = _valid_doc()
    dup = copy.deepcopy(doc["walls"][0])
    doc["walls"].append(dup)
    doc["metadata"]["entity_counts"]["walls"] = 2
    errs = validate(doc)
    assert any("duplicate" in e for e in errs)


def test_entity_counts_mismatch_fails():
    doc = _valid_doc()
    doc["metadata"]["entity_counts"]["walls"] = 99
    errs = validate(doc)
    assert any("entity_counts" in e for e in errs)


def test_room_references_unknown_segment_fails():
    doc = _valid_doc()
    doc["rooms"][0]["bounding_walls"].append("ghost")
    errs = validate(doc)
    assert any("ghost" in e for e in errs)


def test_wall_adjacent_to_unknown_room_fails():
    """adjacent_room_ids must reference real room_ids — the module docstring
    promises this validation, and a dangling reference indicates stage drift."""
    doc = _valid_doc()
    doc["walls"][0]["topology"]["adjacent_room_ids"] = ["ghost-room"]
    errs = validate(doc)
    assert any("ghost-room" in e and "adjacent_room_ids" in e for e in errs)


def test_text_region_enclosing_unknown_room_fails():
    doc = _valid_doc()
    doc["text_regions"][0]["enclosing_room_id"] = "ghost-room"
    errs = validate(doc)
    assert any("ghost-room" in e or "enclosing_room_id" in e for e in errs)


def test_opening_references_unknown_wall_fails():
    doc = _valid_doc()
    doc["openings"].append({
        "opening_id": "op-1",
        "type": "door",
        "position": [10, 10],
        "width": 30.0,
        "swing_arc": None,
        "wall_segment_id": "ghost-wall",
        "connects_room_ids": None,
        "confidence": 0.5,
        "rule_triggered": "test",
    })
    doc["metadata"]["entity_counts"]["openings"] = 1
    errs = validate(doc)
    assert any("ghost-wall" in e for e in errs)


def test_empty_optional_entity_lists_pass():
    doc = _valid_doc()
    doc["rooms"] = []
    doc["openings"] = []
    doc["grid_lines"] = []
    doc["cross_references"] = []
    doc["metadata"]["entity_counts"]["rooms"] = 0
    # Need to also clean up references to the room we deleted
    doc["text_regions"][0]["enclosing_room_id"] = None
    doc["walls"][0]["topology"]["adjacent_room_ids"] = []
    errs = validate(doc)
    assert errs == [], f"unexpected errors: {errs}"
