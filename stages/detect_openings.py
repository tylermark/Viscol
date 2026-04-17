"""Stage 6 — Opening detection.

Doors are drawn as quarter-circle arcs (the "swing") connecting a door-edge
line to a perpendicular line. In vector PDFs, a door swing is typically a
cubic Bezier (``kind="arc"`` in our Stage 1 output) whose chord length is
close to the door width.

We detect openings by:
1. Finding arc primitives with chord length in [door_min, door_max].
2. Estimating the arc's "hinge" point (the vertex of the swing).
3. Locating the nearest wall centerline within ``door_to_wall_distance``.
4. Emitting an opening linked to that wall.

Window detection is deferred — windows are notationally diverse (break-lines,
multi-line symbols, or explicit W-1 callouts) and a separate rule set would
be needed per drawing convention. For Phase 1 we emit windows only if we
have obvious symbol evidence; otherwise we leave the count empty rather than
hallucinate.
"""

from __future__ import annotations

import math
import uuid

import networkx as nx
import numpy as np


def _arc_chord_length(path: dict) -> float:
    pts = path.get("points") or []
    if len(pts) < 2:
        return 0.0
    dx = float(pts[-1][0] - pts[0][0])
    dy = float(pts[-1][1] - pts[0][1])
    return math.hypot(dx, dy)


def _wall_midpoint(data: dict) -> tuple[float, float]:
    s = data["start"]
    e = data["end"]
    return ((s[0] + e[0]) / 2.0, (s[1] + e[1]) / 2.0)


def _closest_wall_to_point(
    point: tuple[float, float],
    walls: dict[str, dict],
    max_distance: float,
) -> str | None:
    px, py = point
    best_id = None
    best_dist = max_distance
    for segment_id, data in walls.items():
        s = np.asarray(data["start"], dtype=float)
        e = np.asarray(data["end"], dtype=float)
        ab = e - s
        ab_len2 = float(ab @ ab)
        if ab_len2 < 1e-9:
            continue
        t = float(((px - s[0]) * ab[0] + (py - s[1]) * ab[1]) / ab_len2)
        t = max(0.0, min(1.0, t))
        foot = s + t * ab
        d = math.hypot(px - foot[0], py - foot[1])
        if d < best_dist:
            best_dist = d
            best_id = segment_id
    return best_id


def detect_openings(
    extracted: dict,
    graph: nx.MultiGraph,
    rooms: list[dict],
    config: dict,
) -> list[dict]:
    """Detect door openings from arc primitives and link them to walls/rooms."""
    chord_min = float(config["door_arc_chord_min"])
    chord_max = float(config["door_arc_chord_max"])
    door_to_wall = float(config["door_to_wall_distance"])

    walls: dict[str, dict] = {}
    for _u, _v, key, data in graph.edges(keys=True, data=True):
        walls[key] = data
    if not walls:
        return []

    openings: list[dict] = []
    for path in extracted.get("paths") or []:
        if path.get("kind") != "arc":
            continue
        chord = _arc_chord_length(path)
        if chord < chord_min or chord > chord_max:
            continue
        pts = path.get("points") or []
        if len(pts) < 2:
            continue

        # Use the chord midpoint as the probe point for wall association.
        # This is not exactly the hinge — but it's robust and avoids needing
        # the full arc control points (which PyMuPDF's cubic Bezier gives us
        # but we simplify to start/end in Stage 1).
        mid = ((pts[0][0] + pts[-1][0]) / 2.0, (pts[0][1] + pts[-1][1]) / 2.0)
        wall_id = _closest_wall_to_point(mid, walls, door_to_wall)
        if wall_id is None:
            # Arc not near any wall — likely a decorative curve, symbol, or
            # standalone door swing we can't localize. Skip rather than fabricate.
            continue

        # Identify the two rooms this opening connects (walls bound up to 2 rooms)
        wall_data = walls[wall_id]
        connects = list(wall_data.get("adjacent_room_ids") or [])[:2]

        openings.append(
            {
                "opening_id": str(uuid.uuid4()),
                "type": "door",
                "position": [float(mid[0]), float(mid[1])],
                "width": float(chord),
                "swing_arc": [list(pts[0]), [float(mid[0]), float(mid[1])], list(pts[-1])],
                "wall_segment_id": wall_id,
                "connects_room_ids": connects if connects else None,
                "confidence": 0.6,
                "rule_triggered": "arc_chord_near_wall",
            }
        )
    return openings
