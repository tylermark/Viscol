"""Generate a tiny synthetic floor plan PDF for smoke-testing the pipeline.

Run:
    python tests/fixtures/make_synthetic_pdf.py

Produces `tests/fixtures/synthetic_floorplan.pdf`.

The plan:
  - 500x300 building at (50, 50) to (550, 350) in top-left page coords.
  - Exterior walls are drawn as parallel-line pairs (thickness 6pt).
  - One interior partition wall (thickness 6pt) splitting the building vertically at x=300.
  - A "BATHROOM" text label in the right half.
"""

from __future__ import annotations

from pathlib import Path

import fitz


OUTPUT = Path(__file__).resolve().parent / "synthetic_floorplan.pdf"

PAGE_WIDTH = 612.0  # Letter portrait
PAGE_HEIGHT = 792.0

WALL_STROKE_WIDTH = 1.2  # above 0.5 wall_min_line_weight
WALL_COLOR = (0.0, 0.0, 0.0)
WALL_THICKNESS = 6.0

# Building rectangle in PDF top-left origin
BUILD_X0, BUILD_Y0 = 50.0, 50.0
BUILD_X1, BUILD_Y1 = 550.0, 350.0
INTERIOR_X = 300.0


def _wall_pair(page, p1, p2):
    """Draw a wall as two parallel lines offset by WALL_THICKNESS.

    p1, p2 are the inner-edge endpoints; outer edge is offset perpendicular to (p2-p1).
    """
    import math

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    # Perpendicular unit vector (rotate 90 CCW)
    nx, ny = -dy / length, dx / length
    offset = WALL_THICKNESS
    p1b = (p1[0] + nx * offset, p1[1] + ny * offset)
    p2b = (p2[0] + nx * offset, p2[1] + ny * offset)

    shape = page.new_shape()
    shape.draw_line(fitz.Point(*p1), fitz.Point(*p2))
    shape.finish(color=WALL_COLOR, width=WALL_STROKE_WIDTH)
    shape.commit()

    shape = page.new_shape()
    shape.draw_line(fitz.Point(*p1b), fitz.Point(*p2b))
    shape.finish(color=WALL_COLOR, width=WALL_STROKE_WIDTH)
    shape.commit()


def main() -> Path:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    # Exterior walls (outer edge; wall thickness extends inward for top/left, outward for bottom/right)
    # Top wall: horizontal at y=BUILD_Y0, from x0 to x1. Offset downward (into the building) by WALL_THICKNESS.
    _wall_pair(page, (BUILD_X0, BUILD_Y0), (BUILD_X1, BUILD_Y0))
    # Right wall: vertical at x=BUILD_X1, from y0 to y1. Offset leftward (inward).
    _wall_pair(page, (BUILD_X1, BUILD_Y1), (BUILD_X1, BUILD_Y0))
    # Bottom wall: horizontal at y=BUILD_Y1, from x1 to x0. Offset upward (inward).
    _wall_pair(page, (BUILD_X1, BUILD_Y1), (BUILD_X0, BUILD_Y1))
    # Left wall: vertical at x=BUILD_X0, from y1 to y0. Offset rightward (inward).
    _wall_pair(page, (BUILD_X0, BUILD_Y0), (BUILD_X0, BUILD_Y1))

    # Interior partition wall
    _wall_pair(page, (INTERIOR_X, BUILD_Y0 + WALL_THICKNESS), (INTERIOR_X, BUILD_Y1 - WALL_THICKNESS))

    # BATHROOM label in right half
    page.insert_text(
        fitz.Point(380, 200),
        "BATHROOM",
        fontsize=10,
        color=(0, 0, 0),
    )

    doc.save(OUTPUT)
    doc.close()
    print(f"Wrote {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    main()
