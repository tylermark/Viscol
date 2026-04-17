"""Annotate a source PDF with pipeline extraction results.

Overlays colored centerlines (one color per functional_role), junction dots,
and a legend on a copy of the input PDF.

Usage:
    python visualize.py <pipeline_output.json> <source_pdf> [--page N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz


ROLE_COLORS: dict[str, tuple[float, float, float]] = {
    "exterior":           (0.85, 0.20, 0.20),  # red
    "bearing_wall":       (0.95, 0.55, 0.10),  # orange
    "demising":           (0.55, 0.25, 0.75),  # purple
    "wet_wall":           (0.20, 0.55, 0.85),  # blue
    "interior_partition": (0.30, 0.65, 0.30),  # green
    "unknown":            (0.55, 0.55, 0.55),  # gray
}

REJECTED_COLOR = (0.75, 0.75, 0.75)  # faint gray for isolation-filtered segments
REJECTED_WIDTH = 0.8

JUNCTION_COLORS: dict[str, tuple[float, float, float]] = {
    "corner":     (0.10, 0.10, 0.10),
    "t-junction": (0.95, 0.75, 0.10),
    "x-junction": (0.90, 0.10, 0.60),
    "endpoint":   (0.70, 0.70, 0.70),
}

CENTERLINE_WIDTH = 1.8
JUNCTION_RADIUS = 2.5
LEGEND_BOX_SIZE = 10.0
LEGEND_ROW_HEIGHT = 14.0
LEGEND_PAD = 8.0
LEGEND_FONT_SIZE = 8.0


def _flip_y(y: float, page_height: float) -> float:
    return page_height - y


def _draw_centerlines(page: "fitz.Page", segments: list[dict], page_height: float) -> None:
    shape = page.new_shape()
    for seg in segments:
        role = seg["semantic"]["functional_role"]
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        s = seg["geometry"]["start"]
        e = seg["geometry"]["end"]
        p1 = fitz.Point(s[0], _flip_y(s[1], page_height))
        p2 = fitz.Point(e[0], _flip_y(e[1], page_height))
        shape.draw_line(p1, p2)
        shape.finish(color=color, width=CENTERLINE_WIDTH)
    shape.commit()


def _draw_junctions(page: "fitz.Page", junctions: list[dict], page_height: float) -> None:
    shape = page.new_shape()
    for j in junctions:
        color = JUNCTION_COLORS.get(j["junction_type"], (0.5, 0.5, 0.5))
        x, y = j["position"]
        center = fitz.Point(x, _flip_y(y, page_height))
        shape.draw_circle(center, JUNCTION_RADIUS)
        shape.finish(color=color, fill=color, width=0.5)
    shape.commit()


def _draw_rejected(page: "fitz.Page", rejected: list[dict], page_height: float) -> None:
    """Draw isolation-filtered segments in faint gray, thinner than real walls."""
    if not rejected:
        return
    shape = page.new_shape()
    for seg in rejected:
        s = seg["geometry"]["start"]
        e = seg["geometry"]["end"]
        p1 = fitz.Point(s[0], _flip_y(s[1], page_height))
        p2 = fitz.Point(e[0], _flip_y(e[1], page_height))
        shape.draw_line(p1, p2)
        shape.finish(color=REJECTED_COLOR, width=REJECTED_WIDTH)
    shape.commit()


def _draw_cross_doc_markers(page: "fitz.Page", segments: list[dict], page_height: float) -> None:
    """Small star-like mark on wet_wall / bearing_wall segments to flag the H2 gap."""
    shape = page.new_shape()
    for seg in segments:
        if not seg["semantic"]["requires_cross_document_validation"]:
            continue
        s = seg["geometry"]["start"]
        e = seg["geometry"]["end"]
        mx = (s[0] + e[0]) / 2.0
        my = (s[1] + e[1]) / 2.0
        c = fitz.Point(mx, _flip_y(my, page_height))
        shape.draw_line(fitz.Point(c.x - 4, c.y), fitz.Point(c.x + 4, c.y))
        shape.draw_line(fitz.Point(c.x, c.y - 4), fitz.Point(c.x, c.y + 4))
        shape.finish(color=(0.0, 0.0, 0.0), width=0.8)
    shape.commit()


def _draw_legend(page: "fitz.Page", role_counts: dict[str, int], rejected_count: int | None = None) -> None:
    page_rect = page.rect
    entries = [
        (role, ROLE_COLORS[role], role_counts.get(role, 0))
        for role in ROLE_COLORS
    ]
    if rejected_count is not None:
        entries.append(("rejected", REJECTED_COLOR, rejected_count))
    rows = len(entries) + len(JUNCTION_COLORS) + 2  # roles + junction types + heading
    box_w = 170.0
    box_h = rows * LEGEND_ROW_HEIGHT + 2 * LEGEND_PAD
    x0 = page_rect.x1 - box_w - LEGEND_PAD
    y0 = page_rect.y0 + LEGEND_PAD
    x1, y1 = x0 + box_w, y0 + box_h

    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(x0, y0, x1, y1))
    shape.finish(color=(0, 0, 0), fill=(1, 1, 1), width=0.8, fill_opacity=0.9)
    shape.commit()

    cursor_y = y0 + LEGEND_PAD + LEGEND_FONT_SIZE
    page.insert_text(
        fitz.Point(x0 + LEGEND_PAD, cursor_y),
        "Pipeline extraction",
        fontsize=LEGEND_FONT_SIZE + 1,
        color=(0, 0, 0),
    )
    cursor_y += LEGEND_ROW_HEIGHT

    for role, color, count in entries:
        swatch_rect = fitz.Rect(
            x0 + LEGEND_PAD,
            cursor_y - LEGEND_BOX_SIZE + 2,
            x0 + LEGEND_PAD + LEGEND_BOX_SIZE,
            cursor_y + 2,
        )
        sh = page.new_shape()
        sh.draw_rect(swatch_rect)
        sh.finish(color=color, fill=color, width=0.5)
        sh.commit()
        label = f"{role}: {count}"
        page.insert_text(
            fitz.Point(x0 + LEGEND_PAD + LEGEND_BOX_SIZE + 6, cursor_y),
            label,
            fontsize=LEGEND_FONT_SIZE,
            color=(0, 0, 0),
        )
        cursor_y += LEGEND_ROW_HEIGHT

    page.insert_text(
        fitz.Point(x0 + LEGEND_PAD, cursor_y),
        "Junctions",
        fontsize=LEGEND_FONT_SIZE + 1,
        color=(0, 0, 0),
    )
    cursor_y += LEGEND_ROW_HEIGHT

    for jtype, jcolor in JUNCTION_COLORS.items():
        center = fitz.Point(
            x0 + LEGEND_PAD + LEGEND_BOX_SIZE / 2,
            cursor_y - LEGEND_BOX_SIZE / 2 + 2,
        )
        sh = page.new_shape()
        sh.draw_circle(center, JUNCTION_RADIUS)
        sh.finish(color=jcolor, fill=jcolor, width=0.4)
        sh.commit()
        page.insert_text(
            fitz.Point(x0 + LEGEND_PAD + LEGEND_BOX_SIZE + 6, cursor_y),
            jtype,
            fontsize=LEGEND_FONT_SIZE,
            color=(0, 0, 0),
        )
        cursor_y += LEGEND_ROW_HEIGHT


def run(
    pipeline_output: Path,
    source_pdf: Path,
    page_num: int | None,
    out_path: Path,
    show_rejected: bool = False,
) -> Path:
    with pipeline_output.open("r", encoding="utf-8") as f:
        doc_data = json.load(f)

    segments = doc_data["segments"]
    junctions = doc_data["junctions"]
    role_counts: dict[str, int] = {}
    for s in segments:
        role = s["semantic"]["functional_role"]
        role_counts[role] = role_counts.get(role, 0) + 1

    rejected_segments: list[dict] = []
    if show_rejected:
        dropped_path = pipeline_output.with_name(pipeline_output.stem + "_dropped.json")
        if dropped_path.exists():
            with dropped_path.open("r", encoding="utf-8") as f:
                rejected_segments = json.load(f).get("segments", [])
        else:
            print(f"--show-rejected set but {dropped_path} not found; rendering without rejected overlay.")

    pdf = fitz.open(source_pdf)
    target_page = 0 if page_num is None else page_num
    if target_page >= pdf.page_count:
        raise ValueError(f"--page {target_page} out of range [0, {pdf.page_count - 1}]")
    page = pdf[target_page]
    page_height = float(page.rect.height)

    # Draw rejected first so real walls render on top of them
    _draw_rejected(page, rejected_segments, page_height)
    _draw_centerlines(page, segments, page_height)
    _draw_junctions(page, junctions, page_height)
    _draw_cross_doc_markers(page, segments, page_height)
    _draw_legend(
        page,
        role_counts,
        rejected_count=len(rejected_segments) if show_rejected else None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(out_path)
    pdf.close()
    print(f"Wrote annotated PDF: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render an annotated PDF from pipeline output.")
    parser.add_argument("pipeline_output", type=Path, help="Path to the pipeline JSON output.")
    parser.add_argument("source_pdf", type=Path, help="Path to the original source PDF.")
    parser.add_argument("--page", type=int, default=None, help="0-indexed page (default: 0).")
    parser.add_argument("--out", type=Path, default=None, help="Output annotated PDF path.")
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Also draw segments that Stage 4's isolation filter dropped (from <stem>_dropped.json) in faint gray.",
    )
    args = parser.parse_args(argv)

    out = args.out or Path("output") / f"{args.source_pdf.stem}_annotated.pdf"
    run(args.pipeline_output, args.source_pdf, args.page, out, show_rejected=args.show_rejected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
