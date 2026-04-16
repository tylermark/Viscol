"""Stage 1 — Path extraction from vector PDF via PyMuPDF.

Produces raw geometric primitives (lines, polylines, rectangles, curves) plus
positioned text blocks. All coordinates returned in bottom-left origin (PDF
convention) — PyMuPDF returns top-left by default, so one Y-flip happens here.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import fitz  # type: ignore


def _parse_dash_pattern(dashes: str | None) -> tuple[list[float] | None, bool]:
    """Parse PyMuPDF dash string like '[] 0' or '[2 2] 0' → (pattern, is_dashed)."""
    if not dashes or not isinstance(dashes, str):
        return None, False
    start = dashes.find("[")
    end = dashes.find("]")
    if start < 0 or end < 0 or end <= start:
        return None, False
    body = dashes[start + 1 : end].strip()
    if not body:
        return None, False
    try:
        pattern = [float(tok) for tok in body.split()]
    except ValueError:
        return None, True
    return (pattern or None), bool(pattern)


def _flip_y_point(pt: tuple[float, float], page_height: float, flip: bool) -> list[float]:
    x, y = float(pt[0]), float(pt[1])
    return [x, page_height - y] if flip else [x, y]


def _flip_y_bbox(bbox: tuple[float, float, float, float], page_height: float, flip: bool) -> list[float]:
    x0, y0, x1, y1 = bbox
    if flip:
        return [float(x0), float(page_height - y1), float(x1), float(page_height - y0)]
    return [float(x0), float(y0), float(x1), float(y1)]


def _extract_text_blocks(page: "fitz.Page", page_height: float, flip: bool, page_num: int) -> list[dict]:
    blocks: list[dict] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                bbox = span.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                blocks.append(
                    {
                        "text": text,
                        "bbox": _flip_y_bbox(tuple(bbox), page_height, flip),
                        "page_num": page_num,
                    }
                )
    return blocks


def _drawing_base(drawing: dict, page_num: int) -> dict:
    stroke = drawing.get("color")
    fill = drawing.get("fill")
    dashes = drawing.get("dashes")
    pattern, is_dashed = _parse_dash_pattern(dashes)
    width = drawing.get("width")
    stroke_rgb = list(stroke) if stroke else None
    fill_rgb = list(fill) if fill else None
    return {
        "stroke_width": float(width) if width is not None else 0.0,
        "stroke_rgb": stroke_rgb,
        "fill_rgb": fill_rgb,
        "dash_pattern": pattern,
        "is_dashed": is_dashed,
        "layer_name": None,
        "page_num": page_num,
    }


def _emit_line(base: dict, p1, p2, page_height: float, flip: bool) -> dict:
    return {
        **base,
        "id": str(uuid.uuid4()),
        "kind": "line",
        "points": [
            _flip_y_point((p1.x, p1.y), page_height, flip),
            _flip_y_point((p2.x, p2.y), page_height, flip),
        ],
    }


def _emit_rect_edges(base: dict, rect, page_height: float, flip: bool) -> list[dict]:
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    edges: list[dict] = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        edges.append(
            {
                **base,
                "id": str(uuid.uuid4()),
                "kind": "line",
                "points": [
                    _flip_y_point(a, page_height, flip),
                    _flip_y_point(b, page_height, flip),
                ],
            }
        )
    return edges


def _emit_curve(base: dict, p_start, p_end, page_height: float, flip: bool) -> dict:
    return {
        **base,
        "id": str(uuid.uuid4()),
        "kind": "arc",
        "points": [
            _flip_y_point((p_start.x, p_start.y), page_height, flip),
            _flip_y_point((p_end.x, p_end.y), page_height, flip),
        ],
    }


def _drawing_to_paths(drawing: dict, page_height: float, flip: bool, page_num: int) -> list[dict]:
    base = _drawing_base(drawing, page_num)
    paths: list[dict] = []
    for item in drawing.get("items", []):
        if not item:
            continue
        kind = item[0]
        if kind == "l" and len(item) >= 3:
            paths.append(_emit_line(base, item[1], item[2], page_height, flip))
        elif kind == "re" and len(item) >= 2:
            paths.extend(_emit_rect_edges(base, item[1], page_height, flip))
        elif kind == "c" and len(item) >= 5:
            paths.append(_emit_curve(base, item[1], item[4], page_height, flip))
        elif kind == "qu" and len(item) >= 2:
            quad = item[1]
            corners: list[Any] = [quad.ul, quad.ur, quad.lr, quad.ll]
            for i in range(4):
                a = corners[i]
                b = corners[(i + 1) % 4]
                paths.append(
                    {
                        **base,
                        "id": str(uuid.uuid4()),
                        "kind": "line",
                        "points": [
                            _flip_y_point((a.x, a.y), page_height, flip),
                            _flip_y_point((b.x, b.y), page_height, flip),
                        ],
                    }
                )
    return paths


def extract_paths(pdf_path: str | Path, page_num: int | None, config: dict) -> dict:
    """Extract vector paths and text blocks from a vector PDF.

    Raises on multi-page PDFs if ``page_num`` is None, per v0.1.0 scope.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    flip = bool(config["coordinate_flip_y"])

    with fitz.open(path) as doc:
        n_pages = doc.page_count
        if n_pages == 0:
            raise ValueError(f"PDF has no pages: {path}")
        if n_pages > 1 and page_num is None:
            raise ValueError(
                f"Multi-page PDF ({n_pages} pages): pass --page N to select one (0-indexed)"
            )
        target = 0 if page_num is None else page_num
        if not (0 <= target < n_pages):
            raise ValueError(f"--page {target} out of range [0, {n_pages - 1}]")

        page = doc[target]
        page_height = float(page.rect.height)
        page_width = float(page.rect.width)

        drawings = page.get_drawings()
        if not drawings:
            raise ValueError(
                f"Page {target} has zero vector drawings — likely a raster scan, "
                "which this pipeline cannot process."
            )

        paths: list[dict] = []
        for drawing in drawings:
            paths.extend(_drawing_to_paths(drawing, page_height, flip, target))

        text_blocks = _extract_text_blocks(page, page_height, flip, target)

    return {
        "page_size": [page_width, page_height],
        "page_num": target,
        "paths": paths,
        "text_blocks": text_blocks,
    }
