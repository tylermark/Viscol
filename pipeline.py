"""Phase 1 extraction pipeline — entry point.

Produces the §4 structured graph per drawing (rooms + walls + openings +
text_regions + grid_lines + junctions + cross_references).

Usage:
    python pipeline.py <pdf_path> [--page N] [--config config.yaml] [--output-dir output]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schema_validator import validate
from stages._config import load_config
from stages.extract_paths import extract_paths
from stages.classify_properties import classify_paths
from stages.detect_walls import detect_walls
from stages.build_topology import build_topology
from stages.detect_rooms import detect_rooms
from stages.detect_openings import detect_openings
from stages.classify_text import classify_text_regions
from stages.detect_grid import detect_grid
from stages.assign_semantics import assign_semantics
from stages.assemble_graph import assemble, PIPELINE_VERSION


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 multi-entity extraction pipeline.")
    parser.add_argument("pdf_path", help="Path to the input vector PDF.")
    parser.add_argument("--page", type=int, default=None, help="0-indexed page (multi-page PDFs).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--output-dir", default="output", help="Directory for the JSON output.")
    return parser.parse_args(argv)


def _print_summary(pdf_path: Path, doc: dict, output_path: Path) -> None:
    m = doc["metadata"]
    counts = m["entity_counts"]
    print(f"Processed: {pdf_path.name}  (v{m['pipeline_version']})")
    print(f"  Page size: {m['page_size'][0]:.0f} x {m['page_size'][1]:.0f}")
    print(f"  Entities extracted:")
    for entity, count in counts.items():
        print(f"    {entity:<15} {count:>6}")
    if doc["rooms"]:
        from collections import Counter
        room_types = Counter(r.get("room_type") or "unknown" for r in doc["rooms"])
        print(f"  Room types: {dict(room_types)}")
    if doc["walls"]:
        from collections import Counter
        wall_roles = Counter(s["semantic"]["functional_role"] for s in doc["walls"])
        print(f"  Wall roles: {dict(wall_roles)}")
    print(f"  Output: {output_path}")


def run(
    pdf_path: str | Path,
    page_num: int | None,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    # Stage 1-4: geometry pipeline (unchanged)
    extracted = extract_paths(pdf_path, page_num, config)
    classify_paths(extracted, config)
    walls, dropped_thickness = detect_walls(extracted, config)
    graph, junctions, dropped_isolation = build_topology(walls, config)

    # Stage 7 first (text classification) because rooms and grid need text_region_ids
    text_regions, cross_references = classify_text_regions(extracted, config)

    # Stage 5: rooms (needs walls graph + text_blocks with text_region_ids attached)
    rooms = detect_rooms(graph, extracted.get("text_blocks") or [], config)

    # Stage 6: openings (needs walls + rooms)
    openings = detect_openings(extracted, graph, rooms, config)

    # Stage 8: grid (needs text_regions)
    grid_lines = detect_grid(extracted, text_regions, config)

    # Stage 9: room-aware semantic assignment
    assign_semantics(graph, extracted.get("text_blocks") or [], config, rooms=rooms)

    # Stage 10: assemble the graph output
    doc = assemble(
        source_pdf_name=pdf_path.name,
        page_size=tuple(extracted["page_size"]),
        graph=graph,
        junctions=junctions,
        rooms=rooms,
        openings=openings,
        text_regions=text_regions,
        grid_lines=grid_lines,
        cross_references=cross_references,
    )

    errors = validate(doc)
    # Include the page number in the filename so different pages of a
    # multi-page PDF can't clobber each other in output_dir.
    page_suffix = "" if page_num is None else f"_p{page_num}"
    output_path = output_dir / f"{pdf_path.stem}{page_suffix}.json"

    if errors:
        # Write the invalid doc to a debug sidecar for inspection, then fail loudly
        # so callers/CI never silently ingest data that doesn't satisfy the schema.
        debug_path = output_path.with_name(f"{output_path.stem}.invalid.json")
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        preview = "\n  - ".join(errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(
            f"Pipeline output failed schema validation ({len(errors)} errors); "
            f"invalid output written to {debug_path}:\n  - {preview}{suffix}"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    # Write dropped-candidate sidecar for visualization and debugging
    dropped_path = output_path.with_name(f"{output_path.stem}_dropped.json")
    dropped_doc = {
        "metadata": {
            "source_pdf": pdf_path.name,
            "pipeline_version": PIPELINE_VERSION,
            "dropped_counts": {
                "dropped_by_thickness": len(dropped_thickness),
                "dropped_by_isolation": len(dropped_isolation),
            },
        },
        "dropped_by_thickness": dropped_thickness,
        "dropped_by_isolation": dropped_isolation,
    }
    with dropped_path.open("w", encoding="utf-8") as f:
        json.dump(dropped_doc, f, indent=2)

    _print_summary(pdf_path, doc, output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args.pdf_path, args.page, args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())