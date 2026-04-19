"""Coordinator-task evaluation harness.

Reads a labeled YAML (produced by generate_labeling_template.py, then hand-
edited) alongside the pipeline's JSON output, and computes precision/recall
per coordinator task per CLAUDE.md §6.

Tasks covered (v1):
  1. Room detection — did we find the rooms? (ignoring type)
  2. Room type classification — among correctly-detected rooms, what fraction
     carry the right type?
  3. Referenced sheets — what's the overlap between detected and valid
     cross-reference targets?

Deferred for v2:
  - Wall-to-schedule-tag assignment (requires more labeling machinery)

Usage:
    python scripts/eval_coordinator_tasks.py <labels.yaml> <pipeline_output.json>
                                             [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml


# From schema_validator.py — duplicated here to keep this script standalone.
ALLOWED_ROOM_TYPES = {
    "unit", "bathroom", "kitchen", "stair", "hallway", "mechanical",
    "laundry", "storage", "office", "unknown",
}
# Sentinel the user can write for any room we detected that isn't really a room.
WRONG_DETECTION_SENTINEL = "wrong_detection"
TODO_SENTINEL = "TODO"


def _check_no_todos(labeled: dict) -> None:
    """Raise if any TODO placeholders are still in the file.

    We don't want to silently evaluate a half-labeled template — that would
    produce misleading precision/recall numbers.
    """
    for r in labeled.get("rooms") or []:
        if r.get("correct_type") == TODO_SENTINEL:
            raise ValueError(
                f"room {r.get('room_id')} still has correct_type={TODO_SENTINEL!r}. "
                "Finish labeling before running the evaluation."
            )


def _room_precision_recall(doc: dict, labeled: dict) -> dict:
    """Room detection: did we find the real rooms, regardless of type?

    - True positive:  a detected room whose label is NOT "wrong_detection".
    - False positive: a detected room whose label IS "wrong_detection".
    - False negative: an entry in missed_rooms (rooms the user saw but we
      didn't detect).
    """
    labeled_by_id: dict[str, str] = {
        r["room_id"]: r["correct_type"]
        for r in (labeled.get("rooms") or [])
        if r.get("room_id") is not None
    }
    tp = 0
    fp = 0
    for room in doc.get("rooms") or []:
        rid = room.get("room_id")
        correct = labeled_by_id.get(rid)
        if correct is None:
            # Detected but not in the label set — treat as unreviewed; skip.
            continue
        if correct == WRONG_DETECTION_SENTINEL:
            fp += 1
        else:
            tp += 1
    fn = len(labeled.get("missed_rooms") or [])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


def _room_type_accuracy(doc: dict, labeled: dict) -> dict:
    """Among TRUE-positive rooms, what fraction have the correct type?"""
    labeled_by_id = {
        r["room_id"]: r["correct_type"]
        for r in (labeled.get("rooms") or [])
        if r.get("room_id") is not None
        and r["correct_type"] not in (WRONG_DETECTION_SENTINEL, TODO_SENTINEL)
    }
    detected_type_by_id = {r["room_id"]: r.get("room_type") for r in (doc.get("rooms") or [])}

    correct = 0
    total = 0
    mismatches: list[dict] = []
    for rid, correct_type in labeled_by_id.items():
        detected = detected_type_by_id.get(rid, "unknown")
        total += 1
        if detected == correct_type:
            correct += 1
        else:
            mismatches.append({
                "room_id": rid,
                "detected": detected,
                "correct": correct_type,
            })
    accuracy = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 3),
        "mismatches": mismatches,
    }


def _referenced_sheets_metrics(labeled: dict) -> dict:
    """Set-level precision/recall on cross-reference targets."""
    xref = labeled.get("cross_references") or {}
    detected = set(xref.get("detected_targets") or [])
    valid = set(xref.get("valid_targets") or [])
    missed = set(xref.get("missed_targets") or [])

    # TP: targets the user confirmed as valid (must also be in detected)
    tp = len(detected & valid)
    # FP: detected but not in the valid set
    fp = len(detected - valid)
    # FN: targets the user wrote under missed_targets
    fn = len(missed)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "detected_noise_samples": sorted(detected - valid)[:10],
        "missed_samples": sorted(missed)[:10],
    }


def evaluate(doc: dict, labeled: dict) -> dict:
    _check_no_todos(labeled)
    return {
        "plan_stem": labeled.get("plan_stem"),
        "source_pdf": labeled.get("source_pdf") or doc.get("metadata", {}).get("source_pdf"),
        "room_detection": _room_precision_recall(doc, labeled),
        "room_type": _room_type_accuracy(doc, labeled),
        "referenced_sheets": _referenced_sheets_metrics(labeled),
    }


def _render_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Coordinator-task evaluation — {result.get('plan_stem') or '?'}")
    lines.append("")
    lines.append(f"- **Source:** `{result.get('source_pdf')}`")
    lines.append("")
    rd = result["room_detection"]
    lines.append("## Task 1: Room detection")
    lines.append(
        f"- precision = **{rd['precision']:.1%}** "
        f"(tp={rd['tp']}, fp={rd['fp']})"
    )
    lines.append(
        f"- recall    = **{rd['recall']:.1%}** "
        f"(tp={rd['tp']}, fn={rd['fn']})"
    )
    lines.append("")
    rt = result["room_type"]
    lines.append("## Task 2: Room type accuracy (among detected)")
    lines.append(f"- accuracy = **{rt['accuracy']:.1%}** ({rt['correct']}/{rt['total']})")
    if rt["mismatches"]:
        lines.append("")
        lines.append("  | room_id | detected | correct |")
        lines.append("  |---|---|---|")
        for m in rt["mismatches"][:20]:
            lines.append(f"  | `{m['room_id']}` | {m['detected']} | {m['correct']} |")
    lines.append("")
    rs = result["referenced_sheets"]
    lines.append("## Task 3: Referenced-sheets set overlap")
    lines.append(
        f"- precision = **{rs['precision']:.1%}** "
        f"(tp={rs['tp']}, fp={rs['fp']})"
    )
    lines.append(
        f"- recall    = **{rs['recall']:.1%}** "
        f"(tp={rs['tp']}, fn={rs['fn']})"
    )
    if rs["detected_noise_samples"]:
        lines.append("")
        lines.append(f"- Noise samples (detected but not valid): {rs['detected_noise_samples']}")
    if rs["missed_samples"]:
        lines.append(f"- Missed samples (user added, we didn't detect): {rs['missed_samples']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate pipeline output against hand-labeled ground truth.")
    parser.add_argument("labels", type=Path, help="Labeled YAML (from generate_labeling_template.py).")
    parser.add_argument("pipeline_output", type=Path, help="Pipeline output JSON for the same plan.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown report path. Defaults to evaluation/results/coordinator_eval_<stem>_<date>.md (gitignored).",
    )
    args = parser.parse_args(argv)

    with args.labels.open("r", encoding="utf-8") as f:
        labeled = yaml.safe_load(f)
    with args.pipeline_output.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    result = evaluate(doc, labeled)

    if args.report is None:
        stem = labeled.get("plan_stem") or args.labels.stem
        args.report = Path("evaluation") / "results" / f"coordinator_eval_{stem}_{date.today().isoformat()}.md"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.report.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with args.report.open("w", encoding="utf-8") as f:
        f.write(_render_markdown(result))

    print(f"Report (markdown): {args.report}")
    print(f"Report (json):     {json_path}")
    print()
    rd = result["room_detection"]
    rt = result["room_type"]
    rs = result["referenced_sheets"]
    print(f"Room detection:     P={rd['precision']:.1%}  R={rd['recall']:.1%}  (tp={rd['tp']} fp={rd['fp']} fn={rd['fn']})")
    print(f"Room type accuracy: {rt['accuracy']:.1%}  ({rt['correct']}/{rt['total']})")
    print(f"Referenced sheets:  P={rs['precision']:.1%}  R={rs['recall']:.1%}  (tp={rs['tp']} fp={rs['fp']} fn={rs['fn']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
