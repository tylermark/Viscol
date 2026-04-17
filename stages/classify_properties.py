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

    v0.4.1: If the primary stroke-width-based classifier produces too few wall
    candidates (e.g. the PDF was exported with all line weights collapsed to a
    single value, which is common for some CAD exporters), retry with a
    length-based fallback: any solid dark line longer than
    ``wall_fallback_min_length`` becomes a wall_candidate.
    """
    text_blocks = extracted.get("text_blocks") or []
    paths = extracted.get("paths") or []
    for path in paths:
        ctype, confidence = _classify_one(path, text_blocks, config)
        path["candidate_type"] = ctype
        path["classification_confidence"] = float(confidence)

    n_wall_candidates = sum(1 for p in paths if p.get("candidate_type") == "wall_candidate")
    min_required = int(config.get("wall_candidate_min_count", 100))
    # Only fall back on substantial drawings — tiny inputs (tests, synthetic)
    # legitimately have few wall candidates and shouldn't trigger the fallback.
    min_paths_for_fallback = int(config.get("wall_fallback_min_paths", 500))
    if n_wall_candidates < min_required and len(paths) > min_paths_for_fallback:
        _length_fallback_classify(paths, config)
    return extracted


def _length_fallback_classify(paths: list[dict], config: dict) -> None:
    """Fallback: when stroke width is uninformative, mark long solid dark lines as walls.

    Capped at ``wall_fallback_max_candidates`` (default 3000) to keep Stage 3's
    O(n²) pair search tractable. Selects the longest qualifying lines first.
    """
    fallback_min_len = float(config.get("wall_fallback_min_length", 20.0))
    fallback_max_count = int(config.get("wall_fallback_max_candidates", 3000))
    darkness_cap = float(config["wall_color_max_darkness"])

    qualifying: list[tuple[float, dict]] = []
    for path in paths:
        if path.get("candidate_type") == "wall_candidate":
            continue
        if path.get("kind") != "line":
            continue
        if path.get("is_dashed"):
            continue
        if _perceived_lightness(path.get("stroke_rgb")) >= darkness_cap:
            continue
        pts = path.get("points") or []
        if len(pts) < 2:
            continue
        dx = float(pts[-1][0] - pts[0][0])
        dy = float(pts[-1][1] - pts[0][1])
        length = (dx * dx + dy * dy) ** 0.5
        if length < fallback_min_len:
            continue
        qualifying.append((length, path))

    # Take the longest N to keep Stage 3 tractable
    qualifying.sort(key=lambda x: x[0], reverse=True)
    for _length, path in qualifying[:fallback_max_count]:
        path["candidate_type"] = "wall_candidate"
        path["classification_confidence"] = 0.4  # lower than primary-pass walls
