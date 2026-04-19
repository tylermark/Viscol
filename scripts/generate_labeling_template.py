"""Generate a pre-filled YAML ground-truth labeling template.

Turns a pipeline output JSON into a template the user edits by hand to record
ground truth for the coordinator-task evaluation. The template pre-fills every
detected entity (rooms, cross-reference targets) with its current pipeline
answer; the user just confirms or corrects.

Design choice: we do NOT label from scratch. We label by *editing* what the
pipeline already produced. This reduces the labeling cost from hours to
minutes per plan — the user is reviewing ~15 rooms and a sheet list, not
drawing polygons on a blank canvas.

Usage:
    python scripts/generate_labeling_template.py <pipeline_output.json>
                                                 [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def build_template(doc: dict, plan_stem: str) -> dict:
    """Pure function — no I/O. Enables clean unit-testing."""
    rooms_out: list[dict] = []
    for room in doc.get("rooms") or []:
        rooms_out.append({
            "room_id": room.get("room_id"),
            "centroid": [round(float(v), 2) for v in (room.get("centroid") or [0, 0])],
            "area": round(float(room.get("area") or 0), 1),
            "detected_type": room.get("room_type") or "unknown",
            "room_name": room.get("room_name"),
            "room_number": room.get("room_number"),
            # "TODO" is the sentinel that signals "not labeled yet" to the eval.
            "correct_type": "TODO",
        })

    xref_targets = sorted({
        x.get("target_sheet")
        for x in (doc.get("cross_references") or [])
        if x.get("target_sheet")
    })

    return {
        "plan_stem": plan_stem,
        "source_pdf": doc.get("metadata", {}).get("source_pdf"),
        "pipeline_version": doc.get("metadata", {}).get("pipeline_version"),
        "_instructions": (
            "Fill in `correct_type` for each room (use one of ALLOWED_ROOM_TYPES "
            "or 'wrong_detection'). Add any rooms we missed under `missed_rooms`. "
            "For cross_references, keep only the target IDs that are real sheet "
            "references in `valid_targets`; list any you see on the drawing that "
            "we missed under `missed_targets`. Run scripts/eval_coordinator_tasks.py "
            "when done."
        ),
        "rooms": rooms_out,
        "missed_rooms": [],
        "cross_references": {
            "detected_targets": list(xref_targets),
            # Starts as a copy; user removes noise (e.g. fixture callouts).
            "valid_targets": list(xref_targets),
            "missed_targets": [],
        },
    }


def _yaml_dump(template: dict) -> str:
    # Preserve key order and use block style for readability when hand-editing.
    return yaml.safe_dump(template, sort_keys=False, default_flow_style=False, width=120)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a ground-truth labeling template.")
    parser.add_argument("pipeline_output", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    with args.pipeline_output.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    plan_stem = args.pipeline_output.stem
    template = build_template(doc, plan_stem=plan_stem)

    out_path = args.out or Path("evaluation") / "labels" / f"{plan_stem}.labels.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(
            f"WARNING: {out_path} already exists. Refusing to overwrite "
            "(it may contain manual edits). Remove it explicitly if you "
            "want to regenerate.",
            file=sys.stderr,
        )
        return 1

    with out_path.open("w", encoding="utf-8") as f:
        f.write(_yaml_dump(template))

    print(f"Wrote labeling template: {out_path}")
    print(f"  rooms to review:      {len(template['rooms'])}")
    print(f"  xref targets to triage: {len(template['cross_references']['detected_targets'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
