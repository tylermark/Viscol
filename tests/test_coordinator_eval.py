"""Tests for the coordinator-task evaluation harness.

Two scripts:
  - scripts/generate_labeling_template.py — turns a pipeline output JSON into
    a pre-filled YAML template the user edits to record ground truth.
  - scripts/eval_coordinator_tasks.py — reads the labeled YAML + pipeline
    output and computes precision/recall per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import eval_coordinator_tasks as evaluator  # noqa: E402
from scripts import generate_labeling_template as generator  # noqa: E402


def _pipeline_doc_with_rooms() -> dict:
    return {
        "metadata": {"source_pdf": "test.pdf", "pipeline_version": "0.6.0"},
        "rooms": [
            {
                "room_id": "r-1",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "area": 10000.0,
                "centroid": [50.0, 50.0],
                "bounding_walls": [],
                "text_labels": [],
                "room_name": "BEDROOM 1",
                "room_number": None,
                "room_type": "unknown",
                "rule_triggered": "polygon_closure",
                "requires_cross_document_validation": False,
            },
            {
                "room_id": "r-2",
                "polygon": [[200, 0], [300, 0], [300, 80], [200, 80]],
                "area": 8000.0,
                "centroid": [250.0, 40.0],
                "bounding_walls": [],
                "text_labels": [],
                "room_name": "BATH",
                "room_number": None,
                "room_type": "bathroom",
                "rule_triggered": "label_match:BATH",
                "requires_cross_document_validation": False,
            },
        ],
        "walls": [],
        "openings": [],
        "text_regions": [],
        "grid_lines": [],
        "junctions": [],
        "cross_references": [
            {
                "cross_reference_id": "x-1",
                "source_text_region_id": "t-1",
                "target_sheet": "A302",
                "target_detail": None,
                "context": "sheet_callout",
                "rule_triggered": "sheet_callout_token_in_text",
                "requires_cross_document_validation": True,
            },
            {
                "cross_reference_id": "x-2",
                "source_text_region_id": "t-2",
                "target_sheet": "F10.",
                "target_detail": None,
                "context": "sheet_callout",
                "rule_triggered": "sheet_callout_token_in_text",
                "requires_cross_document_validation": True,
            },
        ],
    }


# ---------------------------------------------------------------- generator


def test_generator_emits_every_detected_room():
    doc = _pipeline_doc_with_rooms()
    template = generator.build_template(doc, plan_stem="test")
    assert len(template["rooms"]) == 2
    ids = [r["room_id"] for r in template["rooms"]]
    assert ids == ["r-1", "r-2"]
    # correct_type is left as TODO sentinel so users know what to fill in
    assert all(r["correct_type"] == "TODO" for r in template["rooms"])
    # detected_type is preserved from the pipeline output
    assert template["rooms"][0]["detected_type"] == "unknown"
    assert template["rooms"][1]["detected_type"] == "bathroom"


def test_generator_raises_on_malformed_centroid():
    """Malformed centroid should raise ValueError naming the room_id, not
    silently coerce to [0, 0]."""
    doc = {
        "metadata": {},
        "rooms": [
            {"room_id": "bad-centroid", "centroid": [1.0], "area": 100.0},
        ],
    }
    with pytest.raises(ValueError, match="bad-centroid"):
        generator.build_template(doc, plan_stem="test")


def test_generator_raises_on_negative_area():
    """Non-numeric or negative area should raise with room context."""
    doc = {
        "metadata": {},
        "rooms": [
            {"room_id": "bad-area", "centroid": [1.0, 2.0], "area": -5.0},
        ],
    }
    with pytest.raises(ValueError, match="bad-area"):
        generator.build_template(doc, plan_stem="test")


def test_generator_initializes_missed_and_cross_refs():
    doc = _pipeline_doc_with_rooms()
    template = generator.build_template(doc, plan_stem="test")
    assert template["missed_rooms"] == []
    assert set(template["cross_references"]["detected_targets"]) == {"A302", "F10."}
    # valid_targets starts as a copy; the user edits to remove noise
    assert set(template["cross_references"]["valid_targets"]) == {"A302", "F10."}
    assert template["cross_references"]["missed_targets"] == []


# ----------------------------------------------------------------- evaluator


def _labeled_perfect(doc: dict) -> dict:
    """A labeling where everything the pipeline produced is confirmed correct."""
    return {
        "plan_stem": "test",
        "rooms": [
            # r-1 was "unknown" but the label says it's a unit
            {"room_id": "r-1", "centroid": [50.0, 50.0], "correct_type": "unit"},
            # r-2 was bathroom and that's right
            {"room_id": "r-2", "centroid": [250.0, 40.0], "correct_type": "bathroom"},
        ],
        "missed_rooms": [],
        "cross_references": {
            "detected_targets": ["A302", "F10."],
            "valid_targets": ["A302"],       # F10. is noise
            "missed_targets": [],
        },
    }


def test_eval_room_detection_precision_and_recall():
    doc = _pipeline_doc_with_rooms()
    labeled = _labeled_perfect(doc)
    result = evaluator.evaluate(doc, labeled)
    # Both detected rooms have labels and neither is "wrong_detection"; no missed:
    # precision = 2/2, recall = 2/(2+0) = 1.0
    assert result["room_detection"]["precision"] == 1.0
    assert result["room_detection"]["recall"] == 1.0
    assert result["room_detection"]["tp"] == 2
    assert result["room_detection"]["fp"] == 0
    assert result["room_detection"]["fn"] == 0


def test_eval_room_type_accuracy():
    doc = _pipeline_doc_with_rooms()
    labeled = _labeled_perfect(doc)
    result = evaluator.evaluate(doc, labeled)
    # r-1 detected=unknown vs labeled=unit → wrong type
    # r-2 detected=bathroom vs labeled=bathroom → correct
    # type_accuracy = 1 correct / 2 detected-that-were-real-rooms
    assert result["room_type"]["correct"] == 1
    assert result["room_type"]["total"] == 2
    assert result["room_type"]["accuracy"] == 0.5


def test_eval_room_detection_with_wrong_detection_and_missed():
    doc = _pipeline_doc_with_rooms()
    labeled = {
        "plan_stem": "test",
        "rooms": [
            {"room_id": "r-1", "centroid": [50.0, 50.0], "correct_type": "unit"},
            # r-2 was actually not a room (sliver, etc.)
            {"room_id": "r-2", "centroid": [250.0, 40.0], "correct_type": "wrong_detection"},
        ],
        "missed_rooms": [
            {"centroid": [500, 500], "correct_type": "kitchen"},
        ],
        "cross_references": {
            "detected_targets": ["A302", "F10."],
            "valid_targets": ["A302"],
            "missed_targets": [],
        },
    }
    result = evaluator.evaluate(doc, labeled)
    # TP = 1 (r-1 matched and real)
    # FP = 1 (r-2 is wrong_detection)
    # FN = 1 (missed kitchen)
    assert result["room_detection"]["tp"] == 1
    assert result["room_detection"]["fp"] == 1
    assert result["room_detection"]["fn"] == 1
    assert result["room_detection"]["precision"] == 0.5
    assert result["room_detection"]["recall"] == 0.5


def test_eval_referenced_sheets_precision_and_recall():
    doc = _pipeline_doc_with_rooms()
    labeled = _labeled_perfect(doc)
    result = evaluator.evaluate(doc, labeled)
    # detected_targets = {A302, F10.}, valid = {A302}, missed = {}
    # TP = 1 (A302 is both detected and valid)
    # FP = 1 (F10. detected but not valid)
    # FN = 0 (nothing missed)
    # precision = 1/2, recall = 1/1
    assert result["referenced_sheets"]["precision"] == 0.5
    assert result["referenced_sheets"]["recall"] == 1.0


def test_eval_referenced_sheets_with_missed():
    doc = _pipeline_doc_with_rooms()
    labeled = {
        "plan_stem": "test",
        # Full room coverage required now; types don't matter for this test.
        "rooms": [
            {"room_id": "r-1", "centroid": [50.0, 50.0], "correct_type": "unit"},
            {"room_id": "r-2", "centroid": [250.0, 40.0], "correct_type": "bathroom"},
        ],
        "missed_rooms": [],
        "cross_references": {
            "detected_targets": ["A302", "F10."],
            "valid_targets": ["A302"],
            "missed_targets": ["A401"],  # user saw this on the drawing but we missed it
        },
    }
    result = evaluator.evaluate(doc, labeled)
    # precision = 1 TP / (1 TP + 1 FP) = 0.5
    # recall = 1 TP / (1 TP + 1 FN) = 0.5
    assert result["referenced_sheets"]["precision"] == 0.5
    assert result["referenced_sheets"]["recall"] == 0.5


# --------------------------------------------- malformed-label-input guards


def test_eval_rejects_invalid_correct_type():
    """A typo like 'bethroom' must fail fast, not silently count as anything."""
    doc = _pipeline_doc_with_rooms()
    labeled = _labeled_perfect(doc)
    labeled["rooms"][0]["correct_type"] = "bethroom"
    with pytest.raises(ValueError, match="invalid correct_type"):
        evaluator.evaluate(doc, labeled)


def test_eval_rejects_detected_room_missing_from_labels():
    """If a detected room has no label row (template is stale vs. pipeline),
    the evaluator refuses to run rather than producing silently-biased metrics."""
    doc = _pipeline_doc_with_rooms()
    labeled = _labeled_perfect(doc)
    # Drop the label entry for r-2; pipeline still has r-2 detected
    labeled["rooms"] = [r for r in labeled["rooms"] if r["room_id"] != "r-2"]
    with pytest.raises(ValueError, match="no label entry"):
        evaluator.evaluate(doc, labeled)


def test_eval_tampered_detected_targets_does_not_affect_metrics():
    """detected_targets in the labeled YAML is informational — the pipeline's
    cross_references are the source of truth. A user who deletes entries from
    detected_targets to inflate precision must not succeed."""
    doc = _pipeline_doc_with_rooms()
    # Pipeline says we detected {A302, F10.}
    tampered = _labeled_perfect(doc)
    # User removes F10. from detected_targets AND valid_targets to try to
    # pretend we never detected it.
    tampered["cross_references"]["detected_targets"] = ["A302"]
    tampered["cross_references"]["valid_targets"] = ["A302"]
    result = evaluator.evaluate(doc, tampered)
    # Metrics should match the un-tampered baseline exactly.
    baseline = evaluator.evaluate(doc, _labeled_perfect(doc))
    assert result["referenced_sheets"] == baseline["referenced_sheets"]
    # And specifically: F10. is still counted as an FP (precision=0.5),
    # because pipeline is the authority on what was detected.
    assert result["referenced_sheets"]["fp"] == 1
    assert result["referenced_sheets"]["precision"] == 0.5


def test_eval_rejects_template_with_todo_placeholders():
    """An unedited template (correct_type still 'TODO') should fail loudly."""
    doc = _pipeline_doc_with_rooms()
    labeled = {
        "plan_stem": "test",
        "rooms": [
            {"room_id": "r-1", "centroid": [50.0, 50.0], "correct_type": "TODO"},
        ],
        "missed_rooms": [],
        "cross_references": {
            "detected_targets": ["A302"],
            "valid_targets": ["A302"],
            "missed_targets": [],
        },
    }
    with pytest.raises(ValueError, match="TODO"):
        evaluator.evaluate(doc, labeled)
