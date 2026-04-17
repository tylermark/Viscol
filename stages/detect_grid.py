"""Stage 8 — Grid detection.

Architectural/structural grids are long near-axis-aligned lines that
terminate at letter/number callouts (A, B, 1, 2, ...). In a typical
floor-plan PDF they span the full building width or height and extend
slightly beyond the building outline into the margin where the callouts
sit.

Detection heuristic:
1. Find all long (> grid_line_min_length) line primitives that are near
   horizontal or near vertical.
2. For each, check if a text region classified as ``grid_label`` sits
   near either endpoint (within ``grid_label_proximity``).
3. If yes, emit a grid_line with the label attached.

Grid lines are often dashed/thin/gray in the source PDF — we don't gate
on stroke width here because grid conventions vary wildly per office.
"""

from __future__ import annotations

import math
import uuid


def _line_length(pts: list) -> float:
    if not pts or len(pts) < 2:
        return 0.0
    dx = float(pts[-1][0] - pts[0][0])
    dy = float(pts[-1][1] - pts[0][1])
    return math.hypot(dx, dy)


def _line_axis(pts: list, angle_tol_deg: float = 5.0) -> str | None:
    if not pts or len(pts) < 2:
        return None
    dx = float(pts[-1][0] - pts[0][0])
    dy = float(pts[-1][1] - pts[0][1])
    if dx == 0 and dy == 0:
        return None
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    if angle <= angle_tol_deg or angle >= 180.0 - angle_tol_deg:
        return "horizontal"
    if abs(angle - 90.0) <= angle_tol_deg:
        return "vertical"
    return None


def _nearest_label(
    endpoint: tuple[float, float],
    text_regions: list[dict],
    max_distance: float,
) -> dict | None:
    px, py = endpoint
    best = None
    best_d = max_distance
    for region in text_regions:
        if region.get("classification") != "grid_label":
            continue
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d = d
            best = region
    return best


def detect_grid(
    extracted: dict,
    text_regions: list[dict],
    config: dict,
) -> list[dict]:
    """Return grid_line records per §4."""
    min_length = float(config["grid_line_min_length"])
    label_prox = float(config["grid_label_proximity"])

    paths = extracted.get("paths") or []
    grid_lines: list[dict] = []

    # First pass: collect candidate grid lines
    candidates: list[tuple[dict, dict | None]] = []
    for path in paths:
        if path.get("kind") != "line":
            continue
        pts = path.get("points") or []
        if len(pts) < 2:
            continue
        if _line_length(pts) < min_length:
            continue
        axis = _line_axis(pts)
        if axis is None:
            continue
        p1, p2 = pts[0], pts[-1]
        label_a = _nearest_label(tuple(p1), text_regions, label_prox)
        label_b = _nearest_label(tuple(p2), text_regions, label_prox)
        if label_a is None and label_b is None:
            continue
        label_region = label_a or label_b
        label_text = (label_region.get("text") or "").strip() if label_region else None
        candidates.append((
            {
                "axis": axis,
                "label": label_text,
                "start": [float(p1[0]), float(p1[1])],
                "end": [float(p2[0]), float(p2[1])],
            },
            label_region,
        ))

    # Dedup: group by (axis, label, ~position). Grid lines are drawn
    # many times (inner extension, outer extension, bubble tick) with the
    # same label — we want one canonical line per grid.
    seen_keys: dict[tuple, int] = {}  # (axis, label, rounded_position) -> index into grid_lines
    for cand, label_region in candidates:
        # Canonical position: for horizontal grid, the Y; for vertical, the X
        if cand["axis"] == "horizontal":
            canonical = round((cand["start"][1] + cand["end"][1]) / 2.0)
        else:
            canonical = round((cand["start"][0] + cand["end"][0]) / 2.0)
        key = (cand["axis"], cand["label"], canonical)
        if key in seen_keys:
            # Merge: keep whichever candidate is longer, but always link this
            # candidate's label_region to the preserved grid_id — the label may
            # differ from the one that first created the entry.
            existing_idx = seen_keys[key]
            existing = grid_lines[existing_idx]
            grid_id = existing["grid_id"]
            new_len = _line_length([cand["start"], cand["end"]])
            existing_len = _line_length([existing["start"], existing["end"]])
            if new_len > existing_len:
                grid_lines[existing_idx] = {**cand, "grid_id": grid_id}
            if label_region is not None:
                linked = label_region.setdefault("linked_entity_ids", [])
                if grid_id not in linked:
                    linked.append(grid_id)
            continue
        entry = {"grid_id": str(uuid.uuid4()), **cand}
        grid_lines.append(entry)
        seen_keys[key] = len(grid_lines) - 1
        if label_region is not None:
            linked = label_region.setdefault("linked_entity_ids", [])
            if entry["grid_id"] not in linked:
                linked.append(entry["grid_id"])

    return grid_lines
