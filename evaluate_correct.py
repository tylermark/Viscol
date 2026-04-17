"""Evaluate pipeline output against a hand-marked `*-correct.pdf` ground truth.

The convention in `data/ground_truth/`: a copy of the source PDF where the
true walls are overdrawn in pure red `(1.0, 0.0, 0.0)`. We extract those red
drawing items as ground-truth segments and match them against our pipeline's
wall centerlines.

Matching rule: each detected wall pair has a centerline + thickness. The
expected GT segments are two parallel lines at perpendicular distance
``±thickness/2`` from the centerline. A detected wall is a true positive if
**both** expected GT lines have a near-collinear match within tolerance.

Usage:
    python evaluate_correct.py <pipeline_output.json> <source_pdf> <correct_pdf>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np


@dataclass
class GTSegment:
    a: np.ndarray
    b: np.ndarray
    length: float


def _extract_red_segments(correct_pdf: Path, page_height: float, flip_y: bool) -> list[GTSegment]:
    """Return all red-stroked line segments from the correct.pdf."""
    segs: list[GTSegment] = []
    doc = fitz.open(correct_pdf)
    page = doc[0]
    for d in page.get_drawings():
        c = d.get("color")
        if not c or not (c[0] > 0.8 and c[1] < 0.3 and c[2] < 0.3):
            continue
        for item in d.get("items", []):
            if not item:
                continue
            kind = item[0]
            if kind == "l" and len(item) >= 3:
                p1, p2 = item[1], item[2]
                a = np.array([p1.x, page_height - p1.y if flip_y else p1.y], dtype=float)
                b = np.array([p2.x, page_height - p2.y if flip_y else p2.y], dtype=float)
                length = float(np.linalg.norm(b - a))
                if length > 1e-6:
                    segs.append(GTSegment(a=a, b=b, length=length))
            elif kind == "re" and len(item) >= 2:
                rect = item[1]
                rect_width = abs(rect.x1 - rect.x0)
                rect_height = abs(rect.y1 - rect.y0)
                # A hand-marked wall drawn as a rectangle has the two long
                # parallel sides representing the wall; the two short sides
                # are endcaps. Emit only the long pair so GT seg counts stay
                # consistent with the "two parallel lines per wall" convention.
                if rect_width >= rect_height:
                    edge_pairs = [
                        ((rect.x0, rect.y0), (rect.x1, rect.y0)),
                        ((rect.x0, rect.y1), (rect.x1, rect.y1)),
                    ]
                else:
                    edge_pairs = [
                        ((rect.x0, rect.y0), (rect.x0, rect.y1)),
                        ((rect.x1, rect.y0), (rect.x1, rect.y1)),
                    ]
                for p1, p2 in edge_pairs:
                    a = np.array([p1[0], page_height - p1[1] if flip_y else p1[1]], dtype=float)
                    b = np.array([p2[0], page_height - p2[1] if flip_y else p2[1]], dtype=float)
                    length = float(np.linalg.norm(b - a))
                    if length > 1e-6:
                        segs.append(GTSegment(a=a, b=b, length=length))
    doc.close()
    return segs


def _angle_deg(v: np.ndarray) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0


def _angular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _seg_contains_point(seg: GTSegment, p: np.ndarray, perp_tol: float, parallel_tol_deg: float) -> bool:
    """True if p lies within perp_tol of seg's infinite line AND inside its [0,1] projection."""
    ab = seg.b - seg.a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-9:
        return False
    t = float(np.dot(p - seg.a, ab) / ab_len2)
    if t < -0.05 or t > 1.05:
        return False
    foot = seg.a + t * ab
    return float(np.linalg.norm(p - foot)) <= perp_tol


def _find_match(
    centerline_start: np.ndarray,
    centerline_end: np.ndarray,
    perp_offset: float,
    gt_segs: list[GTSegment],
    perp_tol: float,
    parallel_tol_deg: float,
    claimed: set[int],
) -> tuple[int | None, int | None]:
    """Return (gt_above_index, gt_below_index) — indices of unclaimed GT segments matching the wall pair."""
    direction = centerline_end - centerline_start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None, None
    dir_unit = direction / length
    perp_unit = np.array([-dir_unit[1], dir_unit[0]])

    expected_start_above = centerline_start + perp_unit * perp_offset
    expected_end_above = centerline_end + perp_unit * perp_offset
    expected_start_below = centerline_start - perp_unit * perp_offset
    expected_end_below = centerline_end - perp_unit * perp_offset

    centerline_angle = _angle_deg(direction)

    above_match = below_match = None
    for idx, seg in enumerate(gt_segs):
        if idx in claimed:
            continue
        seg_angle = _angle_deg(seg.b - seg.a)
        if _angular_diff(seg_angle, centerline_angle) > parallel_tol_deg:
            continue
        if (
            above_match is None
            and _seg_contains_point(seg, expected_start_above, perp_tol, parallel_tol_deg)
            and _seg_contains_point(seg, expected_end_above, perp_tol, parallel_tol_deg)
        ):
            above_match = idx
        if (
            below_match is None
            and _seg_contains_point(seg, expected_start_below, perp_tol, parallel_tol_deg)
            and _seg_contains_point(seg, expected_end_below, perp_tol, parallel_tol_deg)
        ):
            below_match = idx
        if above_match is not None and below_match is not None:
            break
    return above_match, below_match


def evaluate(
    pipeline_output: Path,
    source_pdf: Path,
    correct_pdf: Path,
    perp_tol: float = 6.0,
    parallel_tol_deg: float = 5.0,
) -> dict:
    with pipeline_output.open("r", encoding="utf-8") as f:
        pred = json.load(f)

    src = fitz.open(source_pdf)
    page_height = float(src[0].rect.height)
    src.close()

    gt_segs = _extract_red_segments(correct_pdf, page_height, flip_y=True)

    claimed_gt: set[int] = set()
    tp = 0
    fp_walls: list[dict] = []

    # Sort by length descending so longer (more reliable) walls claim GT first.
    sorted_preds = sorted(
        pred["walls"],
        key=lambda s: s["geometry"]["centerline_length"],
        reverse=True,
    )
    for seg in sorted_preds:
        cs = np.array(seg["geometry"]["start"], dtype=float)
        ce = np.array(seg["geometry"]["end"], dtype=float)
        thickness = float(seg["geometry"]["thickness"])
        offset = max(thickness / 2.0, 0.5)
        above_idx, below_idx = _find_match(
            cs, ce, offset, gt_segs, perp_tol, parallel_tol_deg, claimed_gt
        )
        if above_idx is not None and below_idx is not None:
            tp += 1
            claimed_gt.add(above_idx)
            claimed_gt.add(below_idx)
        else:
            fp_walls.append({"segment_id": seg["segment_id"], "role": seg["semantic"]["functional_role"]})

    matched_gt = claimed_gt

    n_pred = len(pred["walls"])
    n_gt_segs = len(gt_segs)
    n_gt_walls_estimated = n_gt_segs // 2
    n_matched_gt = len(matched_gt)
    fp = n_pred - tp
    fn_walls_estimated = max(0, n_gt_walls_estimated - tp)
    precision = tp / n_pred if n_pred else 0.0
    recall_walls = tp / n_gt_walls_estimated if n_gt_walls_estimated else 0.0
    coverage_segs = n_matched_gt / n_gt_segs if n_gt_segs else 0.0
    f1 = 2 * precision * recall_walls / (precision + recall_walls) if (precision + recall_walls) else 0.0

    # SOFT metric: did the centerline end up near any GT line, regardless of pair match?
    # Useful when the pipeline has the right wall location but wrong thickness.
    soft_tp = 0
    soft_claimed: set[int] = set()
    for seg in sorted_preds:
        cs = np.array(seg["geometry"]["start"], dtype=float)
        ce = np.array(seg["geometry"]["end"], dtype=float)
        cm = (cs + ce) / 2.0
        cangle = _angle_deg(ce - cs)
        for idx, gtseg in enumerate(gt_segs):
            if idx in soft_claimed:
                continue
            if _angular_diff(_angle_deg(gtseg.b - gtseg.a), cangle) > parallel_tol_deg:
                continue
            if _seg_contains_point(gtseg, cm, perp_tol, parallel_tol_deg):
                soft_tp += 1
                soft_claimed.add(idx)
                break
    # soft_tp is bounded above by n_pred (each prediction claims at most one
    # GT line). Dividing by n_gt_segs would cap recall at 0.5 because GT
    # walls are drawn as two parallel segments. Use the per-wall estimate
    # instead so a soft_recall of 1.0 is theoretically reachable.
    soft_precision = soft_tp / n_pred if n_pred else 0.0
    soft_recall = soft_tp / n_gt_walls_estimated if n_gt_walls_estimated else 0.0
    soft_f1 = (
        2 * soft_precision * soft_recall / (soft_precision + soft_recall)
        if (soft_precision + soft_recall) else 0.0
    )

    return {
        "plan": source_pdf.name,
        "n_pred_walls": n_pred,
        "n_gt_red_segments": n_gt_segs,
        "n_gt_walls_estimated": n_gt_walls_estimated,
        "tp_walls": tp,
        "fp_walls": fp,
        "fn_walls_estimated": fn_walls_estimated,
        "matched_gt_segments": n_matched_gt,
        "precision": round(precision, 3),
        "recall_walls": round(recall_walls, 3),
        "coverage_gt_segments": round(coverage_segs, 3),
        "f1": round(f1, 3),
        # SOFT metrics (centerline near any GT line, no pair requirement).
        # Recall is computed per estimated wall, not per raw GT segment, so a
        # value of 1.0 is theoretically reachable with one prediction per wall.
        "soft_tp": soft_tp,
        "soft_precision": round(soft_precision, 3),
        "soft_recall_walls": round(soft_recall, 3),
        "soft_f1": round(soft_f1, 3),
        "perp_tolerance_pt": perp_tol,
        "parallel_tolerance_deg": parallel_tol_deg,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate pipeline output against a hand-marked correct.pdf.")
    parser.add_argument("pipeline_output", type=Path)
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("correct_pdf", type=Path)
    parser.add_argument("--perp-tol", type=float, default=6.0)
    parser.add_argument("--parallel-tol", type=float, default=5.0)
    args = parser.parse_args(argv)

    result = evaluate(
        args.pipeline_output, args.source_pdf, args.correct_pdf,
        perp_tol=args.perp_tol, parallel_tol_deg=args.parallel_tol,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
