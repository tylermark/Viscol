"""Stage 4 — Topology construction.

Clusters wall endpoints into junctions, splits walls whose body is touched by
another wall's endpoint, classifies junctions, and builds a NetworkX MultiGraph.
"""

from __future__ import annotations

import copy
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
) -> tuple[nx.MultiGraph, list[dict]]:
    """Cluster wall endpoints into junctions and construct the topology graph."""
    snap = float(config["junction_snap_distance"])

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

    return graph, junctions
