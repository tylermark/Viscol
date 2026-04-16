"""Stage 5 — Semantic role assignment.

Applies rule-based classification to each wall edge in the topology graph.
Rule order is fixed: exterior → demising → wet_wall → bearing_wall → interior_partition.
Low-confidence results are downgraded to `unknown` while preserving the
`rule_triggered` signal for research.
"""

from __future__ import annotations

import networkx as nx
from shapely.geometry import MultiLineString, Point, MultiPolygon
from shapely.ops import polygonize, unary_union


CROSS_DOC_REQUIRED_ROLES = {"bearing_wall", "wet_wall"}


def _edge_midpoint(data: dict) -> tuple[float, float]:
    s = data["start"]
    e = data["end"]
    return (0.5 * (s[0] + e[0]), 0.5 * (s[1] + e[1]))


def _assign(data: dict, role: str, confidence: float, rule: str, config: dict) -> None:
    min_conf = float(config["semantic_min_confidence"])
    effective_role = role
    effective_rule = rule
    if confidence < min_conf and role != "exterior":
        effective_role = "unknown"
        effective_rule = f"{rule}_low_confidence"
    data["functional_role"] = effective_role
    data["confidence"] = float(confidence)
    data["rule_triggered"] = effective_rule
    data["requires_cross_document_validation"] = role in CROSS_DOC_REQUIRED_ROLES


def _exterior_rule(graph: nx.MultiGraph, config: dict) -> set[tuple]:
    snap = float(config["junction_snap_distance"])
    lines = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        s = data["start"]
        e = data["end"]
        if s != e:
            lines.append([(s[0], s[1]), (e[0], e[1])])
    if not lines:
        return set()
    merged = unary_union(MultiLineString(lines))
    polys = list(polygonize(merged))
    if not polys:
        return set()
    footprint = unary_union(polys)
    if isinstance(footprint, MultiPolygon):
        footprint = max(footprint.geoms, key=lambda p: p.area)
    boundary = footprint.exterior

    assigned: set[tuple] = set()
    for u, v, key, data in graph.edges(keys=True, data=True):
        mid = Point(*_edge_midpoint(data))
        if boundary.distance(mid) <= snap:
            _assign(data, "exterior", 0.9, "exterior_outer_loop", config)
            assigned.add((u, v, key))
    return assigned


def _demising_rule(
    graph: nx.MultiGraph, assigned: set[tuple], config: dict
) -> set[tuple]:
    min_len = float(config["demising_min_length"])
    newly: set[tuple] = set()
    through_types = {"t-junction", "x-junction"}
    for u, v, key, data in graph.edges(keys=True, data=True):
        if (u, v, key) in assigned:
            continue
        if float(data.get("length", 0.0)) <= min_len:
            continue
        if data.get("start_junction_type") not in through_types:
            continue
        if data.get("end_junction_type") not in through_types:
            continue
        _assign(data, "demising", 0.7, "demising_long_through", config)
        newly.add((u, v, key))
    return newly


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalpha())


def _wet_wall_rule(
    graph: nx.MultiGraph, text_blocks: list[dict], assigned: set[tuple], config: dict
) -> set[tuple]:
    labels = [str(lbl).upper() for lbl in (config.get("wet_zone_labels") or [])]
    if not labels or not text_blocks:
        return set()
    proximity = float(config["wet_zone_proximity"])

    wet_zones: list[tuple[float, float, float, float]] = []
    for tb in text_blocks:
        norm = _normalize(tb.get("text", ""))
        if not norm:
            continue
        if any(norm.startswith(lbl) or norm == lbl for lbl in labels):
            x0, y0, x1, y1 = tb["bbox"]
            wet_zones.append(
                (x0 - proximity, y0 - proximity, x1 + proximity, y1 + proximity)
            )
    if not wet_zones:
        return set()

    newly: set[tuple] = set()
    for u, v, key, data in graph.edges(keys=True, data=True):
        if (u, v, key) in assigned:
            continue
        mx, my = _edge_midpoint(data)
        for x0, y0, x1, y1 in wet_zones:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                _assign(data, "wet_wall", 0.5, "wet_wall_text_label_proximity", config)
                newly.add((u, v, key))
                break
    return newly


def _bearing_wall_rule(
    graph: nx.MultiGraph, assigned: set[tuple], config: dict
) -> set[tuple]:
    min_len = float(config["bearing_min_continuous_length"])
    through_types = {"corner", "t-junction", "x-junction"}

    eligible_edges: dict[tuple, dict] = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        if (u, v, key) in assigned:
            continue
        eligible_edges[(u, v, key)] = data

    used: set[tuple] = set()
    newly: set[tuple] = set()
    edge_list = sorted(
        eligible_edges.keys(),
        key=lambda k: eligible_edges[k].get("length", 0.0),
        reverse=True,
    )

    for start_key in edge_list:
        if start_key in used:
            continue
        path = _extend_linear_path(
            graph, eligible_edges, used, start_key, through_types
        )
        total_length = sum(eligible_edges[k].get("length", 0.0) for k in path)
        if total_length > min_len and len(path) >= 2:
            for k in path:
                data = eligible_edges[k]
                _assign(
                    data,
                    "bearing_wall",
                    0.4,
                    "bearing_continuous_path",
                    config,
                )
                used.add(k)
                newly.add(k)
    return newly


def _extend_linear_path(
    graph: nx.MultiGraph,
    eligible: dict,
    used: set[tuple],
    seed: tuple,
    through_types: set[str],
) -> list[tuple]:
    """Greedy extension through corner/T/X junctions only; one branch per junction."""
    if seed in used:
        return []
    path = [seed]
    path_set = {seed}

    def step(current: tuple, forward: bool) -> None:
        u, v, key = current
        node = v if forward else u
        while True:
            node_type = graph.nodes[node].get("junction_type")
            if node_type not in through_types:
                return
            candidates = []
            for nbr in graph.neighbors(node):
                for k in graph[node][nbr]:
                    ek = (node, nbr, k) if (node, nbr, k) in eligible else (nbr, node, k)
                    if ek == current or ek in path_set or ek in used:
                        continue
                    if ek in eligible:
                        candidates.append((ek, nbr))
            if not candidates:
                return
            ek, nbr = max(
                candidates, key=lambda c: eligible[c[0]].get("length", 0.0)
            )
            path.append(ek)
            path_set.add(ek)
            current = ek
            node = nbr

    step(seed, forward=True)
    step(seed, forward=False)
    return path


def _interior_partition_default(
    graph: nx.MultiGraph, assigned: set[tuple], config: dict
) -> None:
    for u, v, key, data in graph.edges(keys=True, data=True):
        if (u, v, key) in assigned:
            continue
        _assign(data, "interior_partition", 0.6, "default_interior", config)


def assign_semantics(
    graph: nx.MultiGraph, text_blocks: list[dict], config: dict
) -> None:
    """Assign a functional_role, confidence, rule_triggered, and cross-doc flag to every edge."""
    assigned: set[tuple] = set()
    assigned |= _exterior_rule(graph, config)
    assigned |= _demising_rule(graph, assigned, config)
    assigned |= _wet_wall_rule(graph, text_blocks, assigned, config)
    assigned |= _bearing_wall_rule(graph, assigned, config)
    _interior_partition_default(graph, assigned, config)
