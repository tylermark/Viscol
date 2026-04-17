"""Stage 3 — Wall candidate detection via HFV2013-style geometric rules.

Pairs parallel line segments separated by a plausible wall thickness and
emits centerlines. Uses numpy only (no Shapely in this stage).
"""

from __future__ import annotations

import math
import uuid

import numpy as np


def _segments_from_paths(paths: list[dict]) -> list[dict]:
    """Extract straight-line segments from wall_candidate paths."""
    segs: list[dict] = []
    for p in paths:
        if p.get("candidate_type") != "wall_candidate":
            continue
        if p.get("kind") != "line":
            continue
        pts = p.get("points") or []
        if len(pts) != 2:
            continue
        a = np.array(pts[0], dtype=float)
        b = np.array(pts[1], dtype=float)
        if np.linalg.norm(b - a) < 1e-9:
            continue
        segs.append(
            {
                "path_id": p["id"],
                "a": a,
                "b": b,
                "stroke_width": float(p.get("stroke_width") or 0.0),
                "stroke_rgb": list(p.get("stroke_rgb") or [0.0, 0.0, 0.0]),
                "is_dashed": bool(p.get("is_dashed")),
                "layer_name": p.get("layer_name"),
            }
        )
    return segs


def _angle_deg(v: np.ndarray) -> float:
    """Angle of vector v in degrees, normalized to [0, 180)."""
    deg = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
    if deg < 0:
        deg += 180.0
    return deg


def _angular_diff(a_deg: float, b_deg: float) -> float:
    """Smallest angular difference between two angles, mod 180."""
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def _perp_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Perpendicular distance from point p to infinite line through a,b."""
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length < 1e-9:
        return float(np.linalg.norm(p - a))
    n = np.array([-ab[1], ab[0]]) / length
    return abs(float(np.dot(p - a, n)))


def _project_scalar(p: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    return float(np.dot(p - origin, direction))


def _pair_overlap_centerline(
    seg_i: dict, seg_j: dict, min_overlap_ratio: float
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Return (centerline_start, centerline_end, overlap_length, min_seg_length) or None."""
    ai, bi = seg_i["a"], seg_i["b"]
    aj, bj = seg_j["a"], seg_j["b"]
    di = bi - ai
    len_i = float(np.linalg.norm(di))
    if len_i < 1e-9:
        return None
    dir_i = di / len_i

    ti_a = _project_scalar(ai, ai, dir_i)
    ti_b = _project_scalar(bi, ai, dir_i)
    tj_a = _project_scalar(aj, ai, dir_i)
    tj_b = _project_scalar(bj, ai, dir_i)
    ti_lo, ti_hi = sorted((ti_a, ti_b))
    tj_lo, tj_hi = sorted((tj_a, tj_b))
    lo, hi = max(ti_lo, tj_lo), min(ti_hi, tj_hi)
    overlap = hi - lo
    if overlap <= 0:
        return None

    len_j = float(np.linalg.norm(bj - aj))
    min_len = min(len_i, len_j)
    if overlap < min_overlap_ratio * min_len:
        return None

    def point_on_seg(t: float, a: np.ndarray, b: np.ndarray, t_a: float, t_b: float) -> np.ndarray:
        if abs(t_b - t_a) < 1e-9:
            return a
        u = (t - t_a) / (t_b - t_a)
        return a + u * (b - a)

    p_i_lo = point_on_seg(lo, ai, bi, ti_a, ti_b)
    p_i_hi = point_on_seg(hi, ai, bi, ti_a, ti_b)
    p_j_lo = point_on_seg(lo, aj, bj, tj_a, tj_b)
    p_j_hi = point_on_seg(hi, aj, bj, tj_a, tj_b)

    c_start = (p_i_lo + p_j_lo) / 2.0
    c_end = (p_i_hi + p_j_hi) / 2.0
    return c_start, c_end, overlap, min_len


def _is_orthogonal(angle_deg: float, tol: float) -> bool:
    d0 = _angular_diff(angle_deg, 0.0)
    d90 = _angular_diff(angle_deg, 90.0)
    return min(d0, d90) <= tol


def _find_thickness_modes(
    thicknesses: list[float],
    weights: list[float] | None,
    bin_width: float,
    min_support: float,
    max_modes: int,
) -> list[float]:
    """Return centers of up to ``max_modes`` dominant thickness peaks.

    v0.4.0 change: bins are now weighted by ``weights`` (typically the
    centerline length of each pair). This biases mode-finding toward real
    walls (which are long) over short noise pairs (cabinets, fixtures,
    hatches) that happen to share a thickness. ``min_support`` is the
    minimum *total weight* for a bin to qualify as a peak (in pt of length
    if length-weighted).
    """
    if not thicknesses:
        return []
    arr = np.asarray(thicknesses, dtype=float)
    if weights is None:
        w = np.ones_like(arr)
    else:
        w = np.asarray(weights, dtype=float)
    t_min = float(arr.min())
    t_max = float(arr.max())
    if t_max - t_min < bin_width:
        return [float(np.average(arr, weights=w))]
    n_bins = max(3, int(np.ceil((t_max - t_min) / bin_width)) + 1)
    edges = np.linspace(t_min, t_min + n_bins * bin_width, n_bins + 1)
    counts, _ = np.histogram(arr, bins=edges, weights=w)
    centers = 0.5 * (edges[:-1] + edges[1:])

    peaks: list[tuple[float, float]] = []
    window = 3
    for i in range(len(counts)):
        c = float(counts[i])
        if c < min_support:
            continue
        is_peak = True
        for k in range(1, window + 1):
            if i - k >= 0 and counts[i - k] >= c:
                is_peak = False
                break
            if i + k < len(counts) and counts[i + k] > c:
                is_peak = False
                break
        if is_peak:
            peaks.append((c, float(centers[i])))

    peaks.sort(key=lambda p: p[0], reverse=True)
    return [center for _, center in peaks[:max_modes]]


def _cluster_by_thickness(
    walls: list[dict], config: dict
) -> tuple[list[dict], list[dict]]:
    """HFV2013 Assumption 4: global thickness consistency.

    Histogram candidate thicknesses, find dominant peaks, reject pairs whose
    thickness is outside ``tolerance * peak`` of any mode. Returns
    ``(kept, dropped)``. If fewer than ``min_candidates`` walls exist, skips
    clustering entirely (pure fallback).
    """
    min_candidates = int(config["thickness_cluster_min_candidates"])
    if len(walls) < min_candidates:
        return list(walls), []

    bin_width = float(config["thickness_cluster_bin_width"])
    min_support_count = int(config["thickness_cluster_min_support"])
    tolerance = float(config["thickness_cluster_tolerance"])
    max_modes = int(config["thickness_cluster_max_modes"])

    # v0.4.0: length-weighted histogram. Real walls are long; cabinets and
    # decorative pairs are short. Weighting by length surfaces wall thickness
    # as the dominant mode even when noise pair COUNT is higher.
    thicknesses = [float(w["thickness"]) for w in walls]
    lengths = [float(w.get("length", 0.0)) for w in walls]
    use_length_weighting = any(L > 0 for L in lengths)
    if use_length_weighting:
        weights = lengths
        median_len = float(np.median([L for L in lengths if L > 0]))
        min_support_weight = float(min_support_count) * max(median_len, 1.0)
    else:
        # Fallback for tests / synthetic data without length info.
        weights = None
        min_support_weight = float(min_support_count)

    modes = _find_thickness_modes(
        thicknesses,
        weights=weights,
        bin_width=bin_width,
        min_support=min_support_weight,
        max_modes=max_modes,
    )
    # v0.4.1: cap mode acceptance at a realistic max wall thickness.
    # Spurious clusters at 30-40pt come from columns/symbols/decorative elements,
    # not walls. GT data shows architectural walls cap at ~12pt for typical scales.
    mode_max = float(config.get("thickness_cluster_mode_max_realistic", 18.0))
    modes = [m for m in modes if m <= mode_max]
    if not modes:
        return list(walls), []

    kept: list[dict] = []
    dropped: list[dict] = []
    for w in walls:
        t = float(w["thickness"])
        accepted = any(abs(t - m) <= tolerance * m for m in modes)
        if accepted:
            w.setdefault("rules_passed", []).append("thickness_cluster_member")
            kept.append(w)
        else:
            w.setdefault("rules_failed", []).append("thickness_cluster_outlier")
            dropped.append(w)
    return kept, dropped


def _centerlines_are_duplicate(w1: dict, w2: dict, tol: float) -> bool:
    c1 = (np.array(w1["start"]) + np.array(w1["end"])) / 2.0
    c2 = (np.array(w2["start"]) + np.array(w2["end"])) / 2.0
    if float(np.linalg.norm(c1 - c2)) > tol:
        return False
    return _angular_diff(w1["angle_degrees"], w2["angle_degrees"]) <= tol


def _detect_structural_junctions(
    segs: list[dict], snap_radius: float, min_cluster_size: int
) -> list[np.ndarray]:
    """Junction-first detection (v0.5.0 Path A): find structural corners by
    clustering raw wall-candidate line endpoints. Clusters of ``min_cluster_size``
    or more endpoints (default 3) are declared structural junctions.

    Returns a list of junction centroids. Each centroid is where 2+ walls meet
    (at a corner, 4+ line endpoints converge; at a T, 3 converge from the
    terminating wall + 2 continuing from the through-wall; at a clean
    line-break on a single wall, only 2 converge — those are excluded).
    """
    if not segs:
        return []
    endpoints = []
    for s in segs:
        endpoints.append(np.asarray(s["a"], dtype=float))
        endpoints.append(np.asarray(s["b"], dtype=float))
    n = len(endpoints)

    # Grid-bucket union-find clustering (same pattern as Stage 4)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, p in enumerate(endpoints):
        key = (int(p[0] // snap_radius), int(p[1] // snap_radius))
        buckets.setdefault(key, []).append(idx)

    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (kx, ky), idxs in buckets.items():
        neighbors: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbors.extend(buckets.get((kx + dx, ky + dy), []))
        for i in idxs:
            for j in neighbors:
                if j <= i:
                    continue
                diff = endpoints[i] - endpoints[j]
                if float(diff @ diff) <= snap_radius * snap_radius:
                    union(i, j)

    clusters: dict[int, list[np.ndarray]] = {}
    for idx in range(n):
        r = find(idx)
        clusters.setdefault(r, []).append(endpoints[idx])

    junctions: list[np.ndarray] = []
    for pts in clusters.values():
        if len(pts) >= min_cluster_size:
            junctions.append(np.mean(np.stack(pts), axis=0))
    return junctions


def _snap_to_junction(
    point: np.ndarray, junctions: list[np.ndarray], tolerance: float
) -> np.ndarray | None:
    """Return the nearest junction within tolerance, or None."""
    best = None
    best_d2 = tolerance * tolerance
    for j in junctions:
        diff = point - j
        d2 = float(diff @ diff)
        if d2 <= best_d2:
            best_d2 = d2
            best = j
    return best


def detect_walls(classified: dict, config: dict) -> tuple[list[dict], list[dict]]:
    """Emit wall-segment records from classified paths.

    Returns ``(kept, dropped_by_thickness)``. ``dropped_by_thickness`` contains
    candidate pairs that Stage 3's filters rejected.

    v0.5.0 Path A: junction-first anchoring. Before thickness clustering,
    we detect structural junctions from raw wall-candidate endpoint
    convergence. Candidate walls whose endpoints don't snap to pre-detected
    junctions are dropped — these are floating wall fragments (cabinets,
    fixtures, decorative pairs) with no structural context.
    """
    paths = classified.get("paths") or []
    segs = _segments_from_paths(paths)

    parallel_tol = float(config["parallel_angle_tolerance"])
    ortho_tol = float(config["orthogonal_angle_tolerance"])
    min_thick = float(config["wall_min_thickness"])
    max_thick = float(config["wall_max_thickness"])
    min_overlap_ratio = float(config["wall_min_overlap_ratio"])
    dedup_tol = float(config["junction_snap_distance"])

    pre_angle = [_angle_deg(s["b"] - s["a"]) for s in segs]
    walls: list[dict] = []

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if _angular_diff(pre_angle[i], pre_angle[j]) > parallel_tol:
                continue
            mid_j = (segs[j]["a"] + segs[j]["b"]) / 2.0
            thickness = _perp_distance(segs[i]["a"], segs[i]["b"], mid_j)
            if thickness < min_thick or thickness > max_thick:
                continue
            overlap_result = _pair_overlap_centerline(segs[i], segs[j], min_overlap_ratio)
            if overlap_result is None:
                continue
            c_start, c_end, overlap_length, _ = overlap_result
            length = float(np.linalg.norm(c_end - c_start))
            if length < 1e-6:
                continue
            angle = _angle_deg(c_end - c_start)

            rules_passed = ["parallel_pairing", "thickness_in_range"]
            rules_failed: list[str] = []
            if _is_orthogonal(angle, ortho_tol):
                rules_passed.append("orthogonality")
            else:
                rules_failed.append("orthogonality")

            avg_stroke = 0.5 * (segs[i]["stroke_width"] + segs[j]["stroke_width"])
            avg_rgb = [
                0.5 * (segs[i]["stroke_rgb"][k] + segs[j]["stroke_rgb"][k]) for k in range(3)
            ]

            walls.append(
                {
                    "segment_id": str(uuid.uuid4()),
                    "start": [float(c_start[0]), float(c_start[1])],
                    "end": [float(c_end[0]), float(c_end[1])],
                    "length": length,
                    "angle_degrees": angle,
                    "thickness": float(thickness),
                    "stroke_width": float(avg_stroke),
                    "color_rgb": avg_rgb,
                    "is_dashed": bool(segs[i]["is_dashed"] or segs[j]["is_dashed"]),
                    "layer_name": segs[i]["layer_name"] or segs[j]["layer_name"],
                    "source_path_ids": [segs[i]["path_id"], segs[j]["path_id"]],
                    "rules_passed": rules_passed,
                    "rules_failed": rules_failed,
                    "overlap_length": float(overlap_length),
                }
            )

    walls.sort(key=lambda w: w["length"], reverse=True)
    deduped: list[dict] = []
    for w in walls:
        if any(_centerlines_are_duplicate(w, kept, dedup_tol) for kept in deduped):
            continue
        deduped.append(w)

    # v0.5.0 Path A: junction-first anchoring. A real wall has both endpoints
    # at structural corners (places where multiple wall lines converge).
    # Walls whose centerline endpoints don't snap to pre-detected junctions
    # are dropped. This drops floating pairs from cabinets/fixtures/decor
    # at the source and produces a wall graph that's connected by construction.
    #
    # Skipped for small inputs (tests, synthetic) where too few endpoints
    # exist to form reliable junction clusters.
    dropped_by_anchor: list[dict] = []
    anchor_enabled = bool(config.get("junction_anchor_enabled", False))
    anchor_tag_only = bool(config.get("junction_anchor_tag_only", False))
    min_segments = int(config["junction_anchor_min_segments"])
    if (anchor_enabled or anchor_tag_only) and len(segs) >= min_segments:
        snap_radius = float(config["junction_raw_snap_radius"])
        min_cluster = int(config["junction_anchor_min_cluster_size"])
        snap_tol = float(config["junction_anchor_snap_tolerance"])
        junctions_xy = _detect_structural_junctions(segs, snap_radius, min_cluster)
        if junctions_xy:
            anchored: list[dict] = []
            for w in deduped:
                start = np.asarray(w["start"], dtype=float)
                end = np.asarray(w["end"], dtype=float)
                snap_start = _snap_to_junction(start, junctions_xy, snap_tol)
                snap_end = _snap_to_junction(end, junctions_xy, snap_tol)
                # Accept wall if at least ONE endpoint snaps to a structural
                # junction. Peninsula walls (one end at a corner, other end
                # free) are legitimate — interior half-walls, counter stubs,
                # etc. A wall with NEITHER endpoint at a junction is a
                # free-floating pair (cabinet, fixture, decoration).
                if snap_start is None and snap_end is None:
                    w.setdefault("rules_failed", []).append("no_junction_anchor")
                    if anchor_enabled and not anchor_tag_only:
                        dropped_by_anchor.append(w)
                        continue
                    # tag-only mode: keep the wall but carry the failed tag
                    anchored.append(w)
                    continue
                # At least one endpoint snaps. In enabled (filter) mode we
                # modify geometry; in tag-only mode we only add a signal tag
                # and leave the wall untouched.
                if anchor_enabled and not anchor_tag_only:
                    if snap_start is not None:
                        w["start"] = [float(snap_start[0]), float(snap_start[1])]
                    if snap_end is not None:
                        w["end"] = [float(snap_end[0]), float(snap_end[1])]
                    new_start = np.asarray(w["start"], dtype=float)
                    new_end = np.asarray(w["end"], dtype=float)
                    new_len = float(np.linalg.norm(new_end - new_start))
                    if new_len < 1e-6:
                        # Both endpoints snapped to the same junction — drop
                        # rather than halt on an otherwise-valid drawing.
                        dropped_by_anchor.append(w)
                        continue
                    w["length"] = new_len
                    # Update angle_degrees to match new geometry
                    dx = float(new_end[0] - new_start[0])
                    dy = float(new_end[1] - new_start[1])
                    w["angle_degrees"] = math.degrees(math.atan2(dy, dx)) % 180.0
                w.setdefault("rules_passed", []).append(
                    "junction_anchored_both"
                    if snap_start is not None and snap_end is not None
                    else "junction_anchored_one"
                )
                anchored.append(w)
            deduped = anchored

    # Legacy 2σ flag (kept for backward compatibility; purely informational)
    if deduped:
        thicknesses = np.array([w["thickness"] for w in deduped], dtype=float)
        mean_t = float(thicknesses.mean())
        std_t = float(thicknesses.std()) if len(thicknesses) > 1 else 0.0
        for w in deduped:
            if std_t > 0 and abs(w["thickness"] - mean_t) > 2.0 * std_t:
                if "thickness_outlier" not in w["rules_failed"]:
                    w["rules_failed"].append("thickness_outlier")
            else:
                if "thickness_consistent" not in w["rules_passed"]:
                    w["rules_passed"].append("thickness_consistent")

    # HFV2013 Assumption 4: global thickness clustering filter
    kept, dropped_by_thickness = _cluster_by_thickness(deduped, config)
    # Merge anchor drops into the dropped return for reporting in visualize
    dropped_by_thickness.extend(dropped_by_anchor)
    return kept, dropped_by_thickness