"""Floor plan semantic extraction pipeline — entry point.

Usage:
    python pipeline.py <pdf_path> [--page N] [--config config.yaml] [--output-dir output]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import networkx as nx

from schema_validator import validate
from stages._config import load_config
from stages.assign_semantics import assign_semantics
from stages.build_topology import build_topology
from stages.classify_properties import classify_paths
from stages.detect_walls import detect_walls
from stages.extract_paths import extract_paths


PIPELINE_VERSION = "0.1.0"
COORDINATE_SYSTEM = "pdf_points_bottom_left_origin"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract semantic wall segments from a vector PDF floor plan.")
    parser.add_argument("pdf_path", help="Path to the input vector PDF.")
    parser.add_argument("--page", type=int, default=None, help="0-indexed page (required for multi-page PDFs).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--output-dir", default="output", help="Directory to write the JSON output.")
    return parser.parse_args(argv)


def _segment_json(start_junction_id: str, end_junction_id: str, data: dict) -> dict:
    s = data["start"]
    e = data["end"]
    length = float(math.hypot(e[0] - s[0], e[1] - s[1]))
    dx, dy = e[0] - s[0], e[1] - s[1]
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
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


def _serialize(
    graph: nx.MultiGraph,
    junctions: list[dict],
    source_pdf: str,
) -> dict:
    segments = [
        _segment_json(u, v, data)
        for u, v, _, data in graph.edges(keys=True, data=True)
    ]
    segments.sort(key=lambda s: s["segment_id"])
    junctions_out = [
        {
            "junction_id": j["junction_id"],
            "position": list(j["position"]),
            "junction_type": j["junction_type"],
            "connected_segment_ids": list(j["connected_segment_ids"]),
        }
        for j in junctions
    ]
    junctions_out.sort(key=lambda j: j["junction_id"])
    return {
        "metadata": {
            "source_pdf": source_pdf,
            "extraction_date": date.today().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "total_segments": len(segments),
            "coordinate_system": COORDINATE_SYSTEM,
        },
        "segments": segments,
        "junctions": junctions_out,
    }


def _print_summary(
    pdf_path: Path,
    n_paths: int,
    n_wall_candidates: int,
    n_walls: int,
    n_junctions: int,
    role_counts: Counter,
    output_path: Path,
) -> None:
    print(f"Processed: {pdf_path.name}")
    print(f"  Paths extracted: {n_paths:,}")
    print(f"  Wall candidates: {n_wall_candidates:,}")
    print(f"  Confirmed walls: {n_walls:,}")
    print(f"  Junctions: {n_junctions:,}")
    print("  Semantic assignments:")

    order = ["exterior", "demising", "interior_partition", "wet_wall", "bearing_wall", "unknown"]
    label_width = max(len(o) for o in order) + 1
    for role in order:
        count = role_counts.get(role, 0)
        suffix = ""
        if role in ("wet_wall", "bearing_wall") and count > 0:
            suffix = " (cross-doc validation required)"
        print(f"    {role + ':':<{label_width + 1}} {count:<3}{suffix}")
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

    extracted = extract_paths(pdf_path, page_num, config)
    classify_paths(extracted, config)
    n_paths = len(extracted["paths"])
    n_wall_candidates = sum(
        1 for p in extracted["paths"] if p.get("candidate_type") == "wall_candidate"
    )

    walls = detect_walls(extracted, config)
    graph, junctions = build_topology(walls, config)
    assign_semantics(graph, extracted["text_blocks"], config)

    doc = _serialize(graph, junctions, pdf_path.name)
    errors = validate(doc)
    if errors:
        raise ValueError(
            "Pipeline output failed schema validation:\n  - "
            + "\n  - ".join(errors)
        )

    output_path = output_dir / f"{pdf_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    role_counts = Counter(s["semantic"]["functional_role"] for s in doc["segments"])
    _print_summary(
        pdf_path,
        n_paths=n_paths,
        n_wall_candidates=n_wall_candidates,
        n_walls=len(doc["segments"]),
        n_junctions=len(doc["junctions"]),
        role_counts=role_counts,
        output_path=output_path,
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args.pdf_path, args.page, args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
