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


def _centerlines_are_duplicate(w1: dict, w2: dict, tol: float) -> bool:
    c1 = (np.array(w1["start"]) + np.array(w1["end"])) / 2.0
    c2 = (np.array(w2["start"]) + np.array(w2["end"])) / 2.0
    if float(np.linalg.norm(c1 - c2)) > tol:
        return False
    return _angular_diff(w1["angle_degrees"], w2["angle_degrees"]) <= tol


def detect_walls(classified: dict, config: dict) -> list[dict]:
    """Emit wall-segment records from classified paths."""
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

    return deduped
