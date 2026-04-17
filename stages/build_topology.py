"""Stage 4 — Topology construction.

Clusters wall endpoints into junctions, splits walls whose body is touched by
another wall's endpoint, classifies junctions, and builds a NetworkX MultiGraph.
"""

from __future__ import annotations

import copy
import math
import uuid

import networkx as nx
import numpy as np


def _foot_parameter(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """Parameter t and foot point F on segment [a,b] closest to p.

    t=0 → a, t=1 → b. t may fall outside [0,1].
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return 0.0, a.copy()
    t = float(np.dot(p - a, ab) / denom)
    return t, a + t * ab


def _split_walls_on_endpoints(walls: list[dict], snap: float) -> list[dict]:
    """Split any wall whose body is touched by another wall's endpoint."""
    if not walls:
        return []
    splits: dict[str, list[float]] = {w["segment_id"]: [] for w in walls}
    interior_min = max(snap, 1e-6)

    wall_arrays = []
    for w in walls:
        a = np.asarray(w["start"], dtype=float)
        b = np.asarray(w["end"], dtype=float)
        length = float(np.linalg.norm(b - a))
        wall_arrays.append((a, b, length))

    for i, w_i in enumerate(walls):
        a_i, b_i, _ = wall_arrays[i]
        for j, w_j in enumerate(walls):
            if i == j:
                continue
            a_j, b_j, length_j = wall_arrays[j]
            if length_j < 2 * interior_min:
                continue
            t_min = interior_min / length_j
            t_max = 1.0 - t_min
            for ep in (a_i, b_i):
                t, foot = _foot_parameter(ep, a_j, b_j)
                if t < t_min or t > t_max:
                    continue
                if float(np.linalg.norm(ep - foot)) > snap:
                    continue
                splits[w_j["segment_id"]].append(t)

    new_walls: list[dict] = []
    for w, (a, b, length) in zip(walls, wall_arrays):
        ts = splits[w["segment_id"]]
        if not ts or length < 2 * interior_min:
            new_walls.append(w)
            continue
        bounds = sorted({0.0, 1.0, *[max(0.0, min(1.0, t)) for t in ts]})
        min_piece_param = interior_min / length
        merged: list[float] = []
        for t in bounds:
            if not merged or (t - merged[-1]) >= min_piece_param:
                merged.append(t)
            else:
                merged[-1] = t
        if merged[0] > 0.0:
            merged[0] = 0.0
        if merged[-1] < 1.0:
            merged[-1] = 1.0

        for k in range(len(merged) - 1):
            t0, t1 = merged[k], merged[k + 1]
            piece_start = a + t0 * (b - a)
            piece_end = a + t1 * (b - a)
            piece_len = float(np.linalg.norm(piece_end - piece_start))
            if piece_len < interior_min:
                continue
            piece = copy.deepcopy(w)
            piece["segment_id"] = str(uuid.uuid4())
            piece["start"] = [float(piece_start[0]), float(piece_start[1])]
            piece["end"] = [float(piece_end[0]), float(piece_end[1])]
            piece["length"] = piece_len
            piece["split_from"] = w["segment_id"]
            new_walls.append(piece)
    return new_walls


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def _cluster_endpoints(walls: list[dict], snap: float) -> tuple[list[int], list[np.ndarray]]:
    """Return (cluster_id_per_endpoint, cluster_positions).

    Endpoints are indexed as 2*i (start) and 2*i+1 (end) for wall i.
    """
    positions = []
    for w in walls:
        positions.append(np.asarray(w["start"], dtype=float))
        positions.append(np.asarray(w["end"], dtype=float))
    n = len(positions)
    uf = _UnionFind(n)
    if snap <= 0:
        return [uf.find(i) for i in range(n)], positions

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, p in enumerate(positions):
        key = (int(p[0] // snap), int(p[1] // snap))
        buckets.setdefault(key, []).append(idx)

    for (kx, ky), idxs in buckets.items():
        neighbors: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbors.extend(buckets.get((kx + dx, ky + dy), []))
        for i in idxs:
            for j in neighbors:
                if j <= i:
                    continue
                if float(np.linalg.norm(positions[i] - positions[j])) <= snap:
                    uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    root_to_cid: dict[int, int] = {}
    cluster_points: dict[int, list[np.ndarray]] = {}
    cid_per_endpoint: list[int] = []
    for idx, root in enumerate(roots):
        if root not in root_to_cid:
            root_to_cid[root] = len(root_to_cid)
            cluster_points[root] = []
        cluster_points[root].append(positions[idx])
        cid_per_endpoint.append(root_to_cid[root])

    centroids: list[np.ndarray] = [np.zeros(2)] * len(root_to_cid)
    for root, cid in root_to_cid.items():
        pts = cluster_points[root]
        centroids[cid] = np.mean(np.stack(pts), axis=0)
    return cid_per_endpoint, centroids


def _classify_junction(directions: list[np.ndarray]) -> str:
    n = len(directions)
    if n == 1:
        return "endpoint"
    if n == 2:
        d = float(np.dot(directions[0], directions[1]))
        if d < -0.99:
            return "endpoint"
        return "corner"
    if n == 3:
        return "t-junction"
    return "x-junction"


def build_topology(
    walls_in: list[dict], config: dict
) -> tuple[nx.MultiGraph, list[dict], list[dict]]:
    """Cluster wall endpoints into junctions and construct the topology graph.

    Returns (graph, junctions, dropped_walls). ``dropped_walls`` contains wall
    records removed by the isolation filter (both endpoints floating AND length
    below ``config['isolated_max_length']``). They are exposed to the caller so
    visualization / debugging can render them separately.
    """
    snap = float(config["junction_snap_distance"])
    isolated_max_length = float(config["isolated_max_length"])

    walls = _split_walls_on_endpoints(list(walls_in), snap)

    cid_per_ep, centroids = _cluster_endpoints(walls, snap)
    n_clusters = len(centroids)

    cluster_walls: list[list[tuple[int, str]]] = [[] for _ in range(n_clusters)]
    for i, _ in enumerate(walls):
        cluster_walls[cid_per_ep[2 * i]].append((i, "start"))
        cluster_walls[cid_per_ep[2 * i + 1]].append((i, "end"))

    cluster_types: list[str] = []
    for cid in range(n_clusters):
        directions: list[np.ndarray] = []
        here = centroids[cid]
        for wall_idx, side in cluster_walls[cid]:
            w = walls[wall_idx]
            other = np.asarray(w["end" if side == "start" else "start"], dtype=float)
            v = other - here
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                directions.append(v / n)
        cluster_types.append(_classify_junction(directions))

    cluster_junction_ids = [str(uuid.uuid4()) for _ in range(n_clusters)]

    surviving_walls: list[dict] = []
    surviving_ep_cids: list[tuple[int, int]] = []
    for i, w in enumerate(walls):
        sc = cid_per_ep[2 * i]
        ec = cid_per_ep[2 * i + 1]
        start_xy = [float(centroids[sc][0]), float(centroids[sc][1])]
        end_xy = [float(centroids[ec][0]), float(centroids[ec][1])]
        length = float(
            np.linalg.norm(np.asarray(end_xy) - np.asarray(start_xy))
        )
        if sc == ec or length < snap:
            continue
        w["start_junction_id"] = cluster_junction_ids[sc]
        w["end_junction_id"] = cluster_junction_ids[ec]
        w["start_junction_type"] = cluster_types[sc]
        w["end_junction_type"] = cluster_types[ec]
        w["start"] = start_xy
        w["end"] = end_xy
        w["length"] = length
        surviving_walls.append(w)
        surviving_ep_cids.append((sc, ec))
    walls = surviving_walls

    cluster_segment_ids: list[list[str]] = [[] for _ in range(n_clusters)]
    for w, (sc, ec) in zip(walls, surviving_ep_cids):
        cluster_segment_ids[sc].append(w["segment_id"])
        cluster_segment_ids[ec].append(w["segment_id"])

    for w, (sc, ec) in zip(walls, surviving_ep_cids):
        others = set(cluster_segment_ids[sc]) | set(cluster_segment_ids[ec])
        others.discard(w["segment_id"])
        w["connected_segment_ids"] = sorted(others)

    active_clusters = {sc for sc, _ in surviving_ep_cids} | {ec for _, ec in surviving_ep_cids}

    # Recompute junction types based on surviving walls, not originals
    final_types: dict[int, str] = {}
    for cid in active_clusters:
        dirs: list[np.ndarray] = []
        here = centroids[cid]
        for w, (sc, ec) in zip(walls, surviving_ep_cids):
            if sc == cid:
                other = np.asarray(w["end"], dtype=float)
            elif ec == cid:
                other = np.asarray(w["start"], dtype=float)
            else:
                continue
            v = other - here
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                dirs.append(v / n)
        final_types[cid] = _classify_junction(dirs)

    for w, (sc, ec) in zip(walls, surviving_ep_cids):
        w["start_junction_type"] = final_types[sc]
        w["end_junction_type"] = final_types[ec]

    # v0.2.0 isolation filter: drop walls where BOTH endpoints are floating
    # AND the wall is short. These are overwhelmingly door swings, dimension
    # ticks, and fixture outlines that Stage 3 spuriously paired into walls.
    dropped_walls: list[dict] = []
    kept_walls: list[dict] = []
    kept_ep_cids: list[tuple[int, int]] = []
    for w, (sc, ec) in zip(walls, surviving_ep_cids):
        both_floating = final_types[sc] == "endpoint" and final_types[ec] == "endpoint"
        if both_floating and float(w["length"]) < isolated_max_length:
            dropped_walls.append(w)
            continue
        kept_walls.append(w)
        kept_ep_cids.append((sc, ec))
    walls = kept_walls
    surviving_ep_cids = kept_ep_cids

    # Rebuild active_clusters, cluster_segment_ids, and connected_segment_ids
    # from the kept set so orphan junctions and stale references are purged.
    active_clusters = {sc for sc, _ in surviving_ep_cids} | {ec for _, ec in surviving_ep_cids}
    cluster_segment_ids = [[] for _ in range(n_clusters)]
    for w, (sc, ec) in zip(walls, surviving_ep_cids):
        cluster_segment_ids[sc].append(w["segment_id"])
        cluster_segment_ids[ec].append(w["segment_id"])
    for w, (sc, ec) in zip(walls, surviving_ep_cids):
        others = set(cluster_segment_ids[sc]) | set(cluster_segment_ids[ec])
        others.discard(w["segment_id"])
        w["connected_segment_ids"] = sorted(others)

    graph: nx.MultiGraph = nx.MultiGraph()
    for cid in active_clusters:
        graph.add_node(
            cluster_junction_ids[cid],
            position=[float(centroids[cid][0]), float(centroids[cid][1])],
            junction_type=final_types[cid],
            connected_segment_ids=list(cluster_segment_ids[cid]),
        )

    for w in walls:
        graph.add_edge(
            w["start_junction_id"],
            w["end_junction_id"],
            key=w["segment_id"],
            **{k: v for k, v in w.items() if k not in ("start_junction_id", "end_junction_id")},
        )

    junctions = [
        {
            "junction_id": cluster_junction_ids[cid],
            "position": [float(centroids[cid][0]), float(centroids[cid][1])],
            "junction_type": final_types[cid],
            "connected_segment_ids": list(cluster_segment_ids[cid]),
        }
        for cid in sorted(active_clusters)
    ]

    # v0.5.0 Path A: post-hoc gap closing. Merge clusters of degree-1 junctions
    # that are close enough to be the broken ends of the same continuous wall
    # (e.g., a single wall split by a door opening into two fragments).
    _close_endpoint_gaps(graph, junctions, config)

    return graph, junctions, dropped_walls


def _close_endpoint_gaps(graph: nx.MultiGraph, junctions: list[dict], config: dict) -> int:
    """Merge pairs of degree-1 junctions that are close enough to be broken ends of the same wall.

    Mutates ``graph`` and ``junctions`` in place. Returns the number of merges performed.

    A merge is performed when:
      - Both junctions have degree 1 (single-wall endpoints)
      - Their distance is less than ``gap_close_distance``
      - The outgoing directions of their walls don't severely diverge (so we
        don't merge two endpoints that clearly belong to different walls).
    """
    gap_dist = float(config.get("gap_close_distance", 0.0))
    if gap_dist <= 0:
        return 0
    max_drift = float(config.get("gap_close_max_angle_drift", 15.0))

    merges = 0
    # Iterate — each pass may enable further merges
    for _iteration in range(3):
        loose = [jid for jid in graph.nodes if graph.degree(jid) == 1]
        if len(loose) < 2:
            break
        positions = {jid: np.asarray(graph.nodes[jid]["position"], dtype=float) for jid in loose}
        # Build spatial buckets for O(n) pairwise search
        buckets: dict[tuple[int, int], list[str]] = {}
        for jid, p in positions.items():
            key = (int(p[0] // gap_dist), int(p[1] // gap_dist))
            buckets.setdefault(key, []).append(jid)

        merged_this_pass = set()
        pairs_to_merge: list[tuple[str, str, float]] = []
        for jid, p in positions.items():
            if jid in merged_this_pass:
                continue
            kx, ky = int(p[0] // gap_dist), int(p[1] // gap_dist)
            best_other, best_dist = None, gap_dist
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other_jid in buckets.get((kx + dx, ky + dy), []):
                        if other_jid == jid or other_jid in merged_this_pass:
                            continue
                        d = float(np.linalg.norm(p - positions[other_jid]))
                        if d < best_dist:
                            # Check angle continuation plausibility
                            if not _angle_plausible(graph, jid, other_jid, positions, max_drift):
                                continue
                            best_dist = d
                            best_other = other_jid
            if best_other is not None:
                pairs_to_merge.append((jid, best_other, best_dist))
                merged_this_pass.add(jid)
                merged_this_pass.add(best_other)

        if not pairs_to_merge:
            break
        for jid_a, jid_b, _d in pairs_to_merge:
            _perform_junction_merge(graph, junctions, jid_a, jid_b)
            merges += 1
    return merges


def _angle_plausible(
    graph: nx.MultiGraph,
    jid_a: str,
    jid_b: str,
    positions: dict[str, np.ndarray],
    max_drift: float,
) -> bool:
    """Two loose endpoints A and B are plausibly the same broken wall end if either
    (1) their walls are approximately colinear (continuation through a gap), or
    (2) the gap forms an L-corner: B lies roughly where A's wall would continue +/- 90°.

    For v0.5 we adopt the permissive test: accept if either the walls are colinear
    OR perpendicular. This is loose enough to catch most real gaps.
    """
    wall_a = next(iter(graph.edges(jid_a, data=True)), None)
    wall_b = next(iter(graph.edges(jid_b, data=True)), None)
    if wall_a is None or wall_b is None:
        return True  # Graph oddity; be permissive

    def outward_dir(wall: tuple, jid: str) -> np.ndarray:
        u, v, data = wall
        other = v if u == jid else u
        this_pos = positions.get(jid)
        other_pos = np.asarray(graph.nodes[other]["position"], dtype=float)
        if this_pos is None:
            return np.zeros(2)
        vec = other_pos - this_pos
        n = float(np.linalg.norm(vec))
        if n < 1e-9:
            return np.zeros(2)
        return vec / n

    dir_a = outward_dir(wall_a, jid_a)
    dir_b = outward_dir(wall_b, jid_b)
    if float(np.linalg.norm(dir_a)) < 1e-9 or float(np.linalg.norm(dir_b)) < 1e-9:
        return True

    # Colinear continuation: walls point AWAY from each other (dir_a . (B-A)/|B-A| > 0
    # and dir_b . (A-B)/|A-B| > 0, i.e., each wall continues toward the other).
    pa, pb = positions[jid_a], positions[jid_b]
    gap_vec = pb - pa
    gap_len = float(np.linalg.norm(gap_vec))
    if gap_len < 1e-9:
        return True
    gap_unit = gap_vec / gap_len

    # Wall A extends outward in dir_a. For continuation, dir_a should point OPPOSITE
    # to the gap direction (B is behind A, so the wall continues from B across the gap to A).
    # Actually we want: A's wall comes INTO A from the direction opposite dir_a;
    # if you continue through A you'd go in -dir_a. For this to cross to B, gap_unit should be ~-dir_a.
    colinear_a = float(np.dot(-dir_a, gap_unit))
    colinear_b = float(np.dot(dir_b, gap_unit))  # B's wall extends toward A
    # Accept if BOTH walls roughly continue toward the gap (both dot products near +1).
    # Allow up to max_drift degrees off-axis, so the threshold is cos(max_drift)
    # (≈0.97 for 15°), not cos(90-max_drift) which would let 75° off-axis through.
    cos_threshold = math.cos(math.radians(max_drift))
    if colinear_a > cos_threshold and colinear_b > cos_threshold:
        return True

    # Or accept if the walls are perpendicular (L-corner across a gap)
    perp = abs(float(np.dot(dir_a, dir_b)))
    if perp < math.sin(math.radians(max_drift)):
        return True

    return False


def _perform_junction_merge(
    graph: nx.MultiGraph,
    junctions: list[dict],
    jid_a: str,
    jid_b: str,
) -> None:
    """Merge junction ``jid_b`` into ``jid_a``: redirect B's edges to A, move A to midpoint, delete B."""
    if jid_a not in graph or jid_b not in graph:
        return
    pa = np.asarray(graph.nodes[jid_a]["position"], dtype=float)
    pb = np.asarray(graph.nodes[jid_b]["position"], dtype=float)
    midpoint = ((pa + pb) / 2.0).tolist()

    # Collect B's incident edges
    edges_from_b = []
    for u, v, key, data in list(graph.edges(jid_b, keys=True, data=True)):
        other = v if u == jid_b else u
        edges_from_b.append((other, key, data))

    # Remove B
    graph.remove_node(jid_b)

    # Re-add B's edges now pointing to A; snap the B-side endpoint to midpoint
    for other, key, data in edges_from_b:
        if other == jid_a:
            continue  # Don't create self-loops
        new_data = dict(data)
        # Update geometry: whichever endpoint was at B should become the midpoint; the A-endpoint already goes through jid_a
        if new_data.get("end_junction_id") == jid_b:
            new_data["end"] = midpoint
            new_data["end_junction_id"] = jid_a
        else:
            new_data["start"] = midpoint
            new_data["start_junction_id"] = jid_a
        # Recompute length
        s = np.asarray(new_data["start"], dtype=float)
        e = np.asarray(new_data["end"], dtype=float)
        new_data["length"] = float(np.linalg.norm(e - s))
        graph.add_edge(jid_a, other, key=key, **new_data)

    # Move A to midpoint and update all of A's incident edges
    graph.nodes[jid_a]["position"] = midpoint
    for u, v, key, data in list(graph.edges(jid_a, keys=True, data=True)):
        other = v if u == jid_a else u
        new_data = dict(data)
        if new_data.get("start_junction_id") == jid_a:
            new_data["start"] = midpoint
        if new_data.get("end_junction_id") == jid_a:
            new_data["end"] = midpoint
        s = np.asarray(new_data["start"], dtype=float)
        e = np.asarray(new_data["end"], dtype=float)
        new_data["length"] = float(np.linalg.norm(e - s))
        # Replace edge data by removing and re-adding (NetworkX doesn't have in-place edge attr replace for MultiGraph cleanly)
        graph.remove_edge(u, v, key=key)
        graph.add_edge(u, v, key=key, **new_data)

    # Update junction type on A based on new degree
    new_degree = graph.degree(jid_a)
    if new_degree == 1:
        new_type = "endpoint"
    elif new_degree == 2:
        new_type = "corner"
    elif new_degree == 3:
        new_type = "t-junction"
    else:
        new_type = "x-junction"
    graph.nodes[jid_a]["junction_type"] = new_type

    # Propagate the recomputed junction_type to every incident edge so that
    # start_junction_type/end_junction_type on those edges stay consistent
    # with the node (downstream semantic rules look at these edge fields).
    for u, v, key, data in list(graph.edges(jid_a, keys=True, data=True)):
        new_data = dict(data)
        changed = False
        if new_data.get("start_junction_id") == jid_a and new_data.get("start_junction_type") != new_type:
            new_data["start_junction_type"] = new_type
            changed = True
        if new_data.get("end_junction_id") == jid_a and new_data.get("end_junction_type") != new_type:
            new_data["end_junction_type"] = new_type
            changed = True
        if changed:
            graph.remove_edge(u, v, key=key)
            graph.add_edge(u, v, key=key, **new_data)

    # Rebuild junctions list entry for A; remove B's entry
    new_junctions = []
    for j in junctions:
        if j["junction_id"] == jid_b:
            continue
        if j["junction_id"] == jid_a:
            j = dict(j)
            j["position"] = midpoint
            j["junction_type"] = new_type
            j["connected_segment_ids"] = sorted(
                list({key for _u, _v, key in graph.edges(jid_a, keys=True)})
            )
        new_junctions.append(j)
    junctions[:] = new_junctions
