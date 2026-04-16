"""schema_validator tests."""

from __future__ import annotations

import copy

from schema_validator import validate


def _valid_doc() -> dict:
    return {
        "metadata": {
            "source_pdf": "x.pdf",
            "extraction_date": "2026-04-16",
            "pipeline_version": "0.1.0",
            "total_segments": 1,
            "coordinate_system": "pdf_points_bottom_left_origin",
        },
        "segments": [
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
                },
                "semantic": {
                    "functional_role": "interior_partition",
                    "confidence": 0.6,
                    "rule_triggered": "default_interior",
                    "requires_cross_document_validation": False,
                },
            }
        ],
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
    }


def test_valid_doc_passes():
    assert validate(_valid_doc()) == []


def test_missing_top_level_key_fails():
    doc = _valid_doc()
    del doc["junctions"]
    errs = validate(doc)
    assert any("junctions" in e for e in errs)


def test_wet_wall_without_cross_doc_flag_fails():
    doc = _valid_doc()
    doc["segments"][0]["semantic"]["functional_role"] = "wet_wall"
    doc["segments"][0]["semantic"]["requires_cross_document_validation"] = False
    errs = validate(doc)
    assert any("requires_cross_document_validation" in e for e in errs)


def test_bearing_wall_without_cross_doc_flag_fails():
    doc = _valid_doc()
    doc["segments"][0]["semantic"]["functional_role"] = "bearing_wall"
    doc["segments"][0]["semantic"]["requires_cross_document_validation"] = False
    errs = validate(doc)
    assert any("requires_cross_document_validation" in e for e in errs)


def test_unknown_functional_role_fails():
    doc = _valid_doc()
    doc["segments"][0]["semantic"]["functional_role"] = "not_a_role"
    errs = validate(doc)
    assert any("functional_role" in e for e in errs)


def test_unknown_junction_type_fails():
    doc = _valid_doc()
    doc["junctions"][0]["junction_type"] = "nope"
    errs = validate(doc)
    assert any("junction_type" in e for e in errs)


def test_duplicate_segment_id_fails():
    doc = _valid_doc()
    dup = copy.deepcopy(doc["segments"][0])
    doc["segments"].append(dup)
    doc["metadata"]["total_segments"] = 2
    errs = validate(doc)
    assert any("duplicate" in e for e in errs)


def test_total_segments_mismatch_fails():
    doc = _valid_doc()
    doc["metadata"]["total_segments"] = 99
    errs = validate(doc)
    assert any("total_segments" in e for e in errs)


def test_junction_references_unknown_segment_fails():
    doc = _valid_doc()
    doc["junctions"][0]["connected_segment_ids"].append("ghost")
    errs = validate(doc)
    assert any("ghost" in e for e in errs)
