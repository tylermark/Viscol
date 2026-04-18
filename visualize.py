"""Annotate a source PDF with pipeline extraction results (v0.6 entities).

Overlays on the original PDF:
  - Walls: colored centerlines, one color per functional_role
  - Junctions: dots colored by junction_type
  - Rooms: translucent polygon fill + outline, labeled with room_type/name
  - Openings: circle marker + "D" / "W" / "?" type badge
  - Grid lines: long dashed strokes with the grid label
  - Cross-reference sources: small "@" marker at the referring text's bbox

Use --entities to pick a subset (e.g. --entities walls,rooms).

Usage:
    python visualize.py <pipeline_output.json> <source_pdf> [--page N]
                        [--out PATH] [--entities walls,junctions,rooms,openings,grid,xrefs]
                        [--show-rejected]
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

REJECTED_COLOR = (0.75, 0.75, 0.75)
REJECTED_WIDTH = 0.8

JUNCTION_COLORS: dict[str, tuple[float, float, float]] = {
    "corner":     (0.10, 0.10, 0.10),
    "t-junction": (0.95, 0.75, 0.10),
    "x-junction": (0.90, 0.10, 0.60),
    "endpoint":   (0.70, 0.70, 0.70),
}

ROOM_TYPE_COLORS: dict[str, tuple[float, float, float]] = {
    "bathroom":   (0.20, 0.55, 0.85),
    "kitchen":    (0.85, 0.45, 0.10),
    "laundry":    (0.40, 0.70, 0.90),
    "stair":      (0.55, 0.25, 0.75),
    "hallway":    (0.70, 0.70, 0.30),
    "mechanical": (0.30, 0.30, 0.30),
    "office":     (0.20, 0.65, 0.55),
    "storage":    (0.60, 0.50, 0.40),
    "unit":       (0.30, 0.65, 0.30),
    "unknown":    (0.60, 0.60, 0.60),
}

OPENING_COLORS: dict[str, tuple[float, float, float]] = {
    "door":    (0.95, 0.50, 0.10),
    "window":  (0.20, 0.55, 0.85),
    "unknown": (0.60, 0.60, 0.60),
}

GRID_COLOR = (0.05, 0.40, 0.70)
XREF_MARKER_COLOR = (0.75, 0.10, 0.50)

CENTERLINE_WIDTH = 1.8
JUNCTION_RADIUS = 2.5
ROOM_OUTLINE_WIDTH = 1.2
ROOM_FILL_OPACITY = 0.18
OPENING_RADIUS = 4.0
GRID_LINE_WIDTH = 0.7
GRID_DASH = "[4 3] 0"
XREF_MARKER_SIZE = 3.0

LEGEND_BOX_SIZE = 10.0
LEGEND_ROW_HEIGHT = 14.0
LEGEND_PAD = 8.0
LEGEND_FONT_SIZE = 8.0

ROOM_LABEL_FONT = 7.0
OPENING_LABEL_FONT = 6.0
GRID_LABEL_FONT = 8.0

ENTITY_CHOICES = ("walls", "junctions", "rooms", "openings", "grid", "xrefs")


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


def _draw_rooms(page: "fitz.Page", rooms: list[dict], page_height: float) -> None:
    """Translucent polygon fill + outline, labeled at the room centroid."""
    for room in rooms:
        poly = room.get("polygon") or []
        if len(poly) < 3:
            continue
        room_type = room.get("room_type") or "unknown"
        color = ROOM_TYPE_COLORS.get(room_type, ROOM_TYPE_COLORS["unknown"])

        shape = page.new_shape()
        pts = [fitz.Point(float(p[0]), _flip_y(float(p[1]), page_height)) for p in poly]
        # polyline-with-close as a filled outline
        shape.draw_polyline(pts)
        shape.finish(
            color=color, fill=color, width=ROOM_OUTLINE_WIDTH,
            fill_opacity=ROOM_FILL_OPACITY,
            closePath=True,
        )
        shape.commit()

        # Label
        cx, cy = room.get("centroid") or [0, 0]
        label_parts: list[str] = []
        if room.get("room_number"):
            label_parts.append(str(room["room_number"]))
        if room.get("room_name"):
            label_parts.append(str(room["room_name"])[:24])
        if not label_parts:
            label_parts.append(room_type)
        label = " · ".join(label_parts)
        page.insert_text(
            fitz.Point(float(cx), _flip_y(float(cy), page_height)),
            label,
            fontsize=ROOM_LABEL_FONT,
            color=(0.05, 0.05, 0.05),
        )


def _draw_openings(page: "fitz.Page", openings: list[dict], page_height: float) -> None:
    """Circle marker at each opening position, with a single-letter type badge."""
    shape = page.new_shape()
    for op in openings:
        pos = op.get("position")
        if not pos or len(pos) < 2:
            continue
        color = OPENING_COLORS.get(op.get("type", "unknown"), OPENING_COLORS["unknown"])
        c = fitz.Point(float(pos[0]), _flip_y(float(pos[1]), page_height))
        shape.draw_circle(c, OPENING_RADIUS)
        shape.finish(color=color, fill=color, width=0.6, fill_opacity=0.55)
    shape.commit()

    for op in openings:
        pos = op.get("position")
        if not pos or len(pos) < 2:
            continue
        t = (op.get("type") or "?")[:1].upper()
        page.insert_text(
            fitz.Point(float(pos[0]) - 2.5, _flip_y(float(pos[1]), page_height) + 2.5),
            t,
            fontsize=OPENING_LABEL_FONT,
            color=(1, 1, 1),
        )


def _draw_grid(page: "fitz.Page", grid_lines: list[dict], page_height: float) -> None:
    """Long dashed strokes with the grid label at one end."""
    shape = page.new_shape()
    for g in grid_lines:
        s = g.get("start")
        e = g.get("end")
        if not s or not e:
            continue
        p1 = fitz.Point(float(s[0]), _flip_y(float(s[1]), page_height))
        p2 = fitz.Point(float(e[0]), _flip_y(float(e[1]), page_height))
        shape.draw_line(p1, p2)
        shape.finish(color=GRID_COLOR, width=GRID_LINE_WIDTH, dashes=GRID_DASH)
    shape.commit()

    for g in grid_lines:
        label = g.get("label")
        s = g.get("start")
        if not label or not s:
            continue
        page.insert_text(
            fitz.Point(float(s[0]) + 2, _flip_y(float(s[1]), page_height) - 2),
            str(label),
            fontsize=GRID_LABEL_FONT,
            color=GRID_COLOR,
        )


def _draw_xref_sources(page: "fitz.Page", cross_refs: list[dict], text_regions: list[dict], page_height: float) -> None:
    """Small '@' marker at the source-text position of each cross-reference."""
    tid_to_region = {t["text_region_id"]: t for t in text_regions}
    drawn_ids: set[str] = set()
    for x in cross_refs:
        src_id = x.get("source_text_region_id")
        if src_id is None or src_id in drawn_ids:
            continue
        region = tid_to_region.get(src_id)
        if region is None:
            continue
        bbox = region.get("bbox") or [0, 0, 0, 0]
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        page.insert_text(
            fitz.Point(cx, _flip_y(cy, page_height)),
            "@",
            fontsize=OPENING_LABEL_FONT + 2,
            color=XREF_MARKER_COLOR,
        )
        drawn_ids.add(src_id)


def _draw_rejected(page: "fitz.Page", rejected: list[dict], page_height: float) -> None:
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


def _draw_legend(
    page: "fitz.Page",
    role_counts: dict[str, int],
    entity_counts: dict[str, int],
    rejected_count: int | None = None,
) -> None:
    page_rect = page.rect
    wall_entries = [
        (role, ROLE_COLORS[role], role_counts.get(role, 0))
        for role in ROLE_COLORS
    ]
    if rejected_count is not None:
        wall_entries.append(("rejected", REJECTED_COLOR, rejected_count))

    rows = (
        1 +                                 # "Walls" header
        len(wall_entries) +
        1 +                                 # "Junctions" header
        len(JUNCTION_COLORS) +
        1 +                                 # "Entities" header
        len(entity_counts)
    )
    box_w = 190.0
    box_h = rows * LEGEND_ROW_HEIGHT + 2 * LEGEND_PAD
    x0 = page_rect.x1 - box_w - LEGEND_PAD
    y0 = page_rect.y0 + LEGEND_PAD
    x1, y1 = x0 + box_w, y0 + box_h

    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(x0, y0, x1, y1))
    shape.finish(color=(0, 0, 0), fill=(1, 1, 1), width=0.8, fill_opacity=0.9)
    shape.commit()

    cursor_y = y0 + LEGEND_PAD + LEGEND_FONT_SIZE

    def _heading(text: str) -> None:
        nonlocal cursor_y
        page.insert_text(
            fitz.Point(x0 + LEGEND_PAD, cursor_y),
            text,
            fontsize=LEGEND_FONT_SIZE + 1,
            color=(0, 0, 0),
        )
        cursor_y += LEGEND_ROW_HEIGHT

    def _swatch_row(color: tuple[float, float, float], label: str) -> None:
        nonlocal cursor_y
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
        page.insert_text(
            fitz.Point(x0 + LEGEND_PAD + LEGEND_BOX_SIZE + 6, cursor_y),
            label,
            fontsize=LEGEND_FONT_SIZE,
            color=(0, 0, 0),
        )
        cursor_y += LEGEND_ROW_HEIGHT

    _heading("Walls (by role)")
    for role, color, count in wall_entries:
        _swatch_row(color, f"{role}: {count}")

    _heading("Junctions")
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

    _heading("v0.6 entities")
    for entity, count in entity_counts.items():
        page.insert_text(
            fitz.Point(x0 + LEGEND_PAD, cursor_y),
            f"{entity}: {count}",
            fontsize=LEGEND_FONT_SIZE,
            color=(0, 0, 0),
        )
        cursor_y += LEGEND_ROW_HEIGHT


def run(
    pipeline_output: Path,
    source_pdf: Path,
    page_num: int | None,
    out_path: Path,
    entities: set[str] | None = None,
    show_rejected: bool = False,
) -> Path:
    entities = entities or set(ENTITY_CHOICES)

    with pipeline_output.open("r", encoding="utf-8") as f:
        doc_data = json.load(f)

    segments = doc_data["walls"]
    junctions = doc_data["junctions"]
    rooms = doc_data.get("rooms") or []
    openings = doc_data.get("openings") or []
    grid_lines = doc_data.get("grid_lines") or []
    text_regions = doc_data.get("text_regions") or []
    cross_refs = doc_data.get("cross_references") or []

    role_counts: dict[str, int] = {}
    for s in segments:
        role = s["semantic"]["functional_role"]
        role_counts[role] = role_counts.get(role, 0) + 1

    rejected_segments: list[dict] = []
    if show_rejected:
        dropped_path = pipeline_output.with_name(pipeline_output.stem + "_dropped.json")
        if dropped_path.exists():
            with dropped_path.open("r", encoding="utf-8") as f:
                dropped_doc = json.load(f)
            rejected_segments = list(dropped_doc.get("dropped_by_thickness") or []) + list(
                dropped_doc.get("dropped_by_isolation") or []
            )
        else:
            print(f"--show-rejected set but {dropped_path} not found; rendering without rejected overlay.")

    pdf = fitz.open(source_pdf)
    target_page = 0 if page_num is None else page_num
    if target_page < 0 or target_page >= pdf.page_count:
        raise ValueError(f"--page {target_page} out of range [0, {pdf.page_count - 1}]")
    page = pdf[target_page]
    page_height = float(page.rect.height)

    # Rooms first so their fill sits behind other overlays.
    if "rooms" in entities:
        _draw_rooms(page, rooms, page_height)
    if show_rejected:
        _draw_rejected(page, rejected_segments, page_height)
    if "walls" in entities:
        _draw_centerlines(page, segments, page_height)
    if "grid" in entities:
        _draw_grid(page, grid_lines, page_height)
    if "junctions" in entities:
        _draw_junctions(page, junctions, page_height)
    if "openings" in entities:
        _draw_openings(page, openings, page_height)
    if "xrefs" in entities:
        _draw_xref_sources(page, cross_refs, text_regions, page_height)

    entity_counts = {
        "rooms": len(rooms),
        "openings": len(openings),
        "grid_lines": len(grid_lines),
        "text_regions": len(text_regions),
        "cross_references": len(cross_refs),
    }
    _draw_legend(
        page,
        role_counts,
        entity_counts,
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
        "--entities",
        type=str,
        default=",".join(ENTITY_CHOICES),
        help=(
            "Comma-separated subset of entity overlays to draw. "
            f"Choices: {','.join(ENTITY_CHOICES)}. Default: all."
        ),
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Also draw dropped-sidecar walls (dropped_by_thickness + dropped_by_isolation) in faint gray.",
    )
    args = parser.parse_args(argv)

    requested = {e.strip() for e in args.entities.split(",") if e.strip()}
    invalid = requested - set(ENTITY_CHOICES)
    if invalid:
        parser.error(f"unknown --entities values: {sorted(invalid)} (choices: {ENTITY_CHOICES})")

    out = args.out or Path("output") / f"{args.source_pdf.stem}_annotated.pdf"
    run(
        args.pipeline_output,
        args.source_pdf,
        args.page,
        out,
        entities=requested,
        show_rejected=args.show_rejected,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
