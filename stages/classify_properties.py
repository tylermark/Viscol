"""Stage 2 — Property classification.

Tags each raw path with a candidate_type based on visual properties only.
No spatial reasoning here except text-proximity for dimensions.
"""

from __future__ import annotations


def _perceived_lightness(rgb: list[float] | None) -> float:
    """Return 0 (black) to 1 (white). None treated as black."""
    if rgb is None:
        return 0.0
    r, g, b = (float(c) for c in rgb[:3])
    return 0.299 * r + 0.587 * g + 0.114 * b


def _nearest_text_distance(path: dict, text_blocks: list[dict]) -> float:
    """Minimum distance from the path's midpoint to any text bbox. inf if no text."""
    pts = path.get("points") or []
    if len(pts) < 2:
        return float("inf")
    mx = (pts[0][0] + pts[-1][0]) / 2.0
    my = (pts[0][1] + pts[-1][1]) / 2.0
    best = float("inf")
    for tb in text_blocks:
        x0, y0, x1, y1 = tb["bbox"]
        dx = max(x0 - mx, 0, mx - x1)
        dy = max(y0 - my, 0, my - y1)
        d = (dx * dx + dy * dy) ** 0.5
        if d < best:
            best = d
    return best


def _classify_one(path: dict, text_blocks: list[dict], config: dict) -> tuple[str, float]:
    width = float(path.get("stroke_width") or 0.0)
    is_dashed = bool(path.get("is_dashed"))
    lightness = _perceived_lightness(path.get("stroke_rgb"))

    wall_min = float(config["wall_min_line_weight"])
    ann_max = float(config["annotation_max_line_weight"])
    darkness_cap = float(config["wall_color_max_darkness"])
    dim_prox = float(config["dimension_text_proximity"])

    if width >= wall_min and not is_dashed and lightness < darkness_cap:
        width_score = min(1.0, (width - wall_min) / max(wall_min, 1e-6) + 0.5)
        color_score = 1.0 - (lightness / max(darkness_cap, 1e-6))
        confidence = max(0.5, min(1.0, 0.5 * width_score + 0.5 * color_score))
        return "wall_candidate", confidence

    if width <= ann_max:
        near = _nearest_text_distance(path, text_blocks)
        if near <= dim_prox:
            confidence = max(0.4, min(0.9, 1.0 - near / max(dim_prox, 1e-6)))
            return "dimension", confidence
        return "annotation", 0.7

    return "unknown", 0.3


def classify_paths(extracted: dict, config: dict) -> dict:
    """Annotate each path with candidate_type and classification_confidence in-place.

    Returns the same dict for fluent chaining. Never discards paths.
    """
    text_blocks = extracted.get("text_blocks") or []
    paths = extracted.get("paths") or []
    for path in paths:
        ctype, confidence = _classify_one(path, text_blocks, config)
        path["candidate_type"] = ctype
        path["classification_confidence"] = float(confidence)
    return extracted
