"""Schema validator for pipeline output (v0.6.0 multi-entity graph).

Validates against the §4 schema in CLAUDE.md. Returns a list of error strings
(empty = valid). Every entity-type top-level key must be present (even if
empty list). Cross-references between entities (e.g., wall.adjacent_room_ids
pointing to real room_ids) are validated.
"""

from __future__ import annotations

import re


ALLOWED_FUNCTIONAL_ROLES = {
    "interior_partition",
    "bearing_wall",
    "wet_wall",
    "demising",
    "exterior",
    "unknown",
}
CROSS_DOC_REQUIRED_ROLES = {"bearing_wall", "wet_wall"}
ALLOWED_JUNCTION_TYPES = {"corner", "t-junction", "x-junction", "endpoint"}
ALLOWED_ROOM_TYPES = {
    "unit", "bathroom", "kitchen", "stair", "hallway", "mechanical",
    "laundry", "storage", "office", "unknown",
}
ALLOWED_OPENING_TYPES = {"door", "window", "unknown"}
ALLOWED_TEXT_CLASSIFICATIONS = {
    "room_label", "room_number", "sheet_callout", "dimension",
    "wall_schedule_tag", "note", "title", "grid_label", "unknown",
}
ALLOWED_GRID_AXES = {"horizontal", "vertical"}

TOP_LEVEL_KEYS = (
    "metadata", "rooms", "walls", "openings",
    "text_regions", "grid_lines", "junctions", "cross_references",
)

REQUIRED_METADATA_FIELDS = {
    "source_pdf": str,
    "extraction_date": str,
    "pipeline_version": str,
    "coordinate_system": str,
    "page_size": list,
    "entity_counts": dict,
}

REQUIRED_ROOM_FIELDS = (
    "room_id", "polygon", "area", "centroid",
    "bounding_walls", "text_labels",
    "room_name", "room_number", "room_type",
    "rule_triggered", "requires_cross_document_validation",
)

REQUIRED_WALL_FIELDS = ("segment_id", "geometry", "visual_properties", "topology", "semantic")
REQUIRED_GEOMETRY_FIELDS = ("start", "end", "centerline_length", "angle_degrees", "thickness")
REQUIRED_VISUAL_FIELDS = ("line_weight", "color_rgb", "is_dashed", "layer_name")
REQUIRED_TOPOLOGY_FIELDS = (
    "start_junction_id", "end_junction_id",
    "start_junction_type", "end_junction_type",
    "connected_segment_ids", "adjacent_room_ids",
)
REQUIRED_SEMANTIC_FIELDS = (
    "functional_role", "confidence", "rule_triggered",
    "requires_cross_document_validation",
)

REQUIRED_OPENING_FIELDS = (
    "opening_id", "type", "position", "width",
    "swing_arc", "wall_segment_id", "connects_room_ids",
    "confidence", "rule_triggered",
)

REQUIRED_TEXT_REGION_FIELDS = (
    "text_region_id", "text", "bbox", "classification",
    "references", "enclosing_room_id", "linked_entity_ids",
)

REQUIRED_GRID_FIELDS = ("grid_id", "axis", "label", "start", "end")

REQUIRED_JUNCTION_FIELDS = ("junction_id", "position", "junction_type", "connected_segment_ids")

REQUIRED_CROSS_REF_FIELDS = ("source_text_region_id", "target_sheet", "target_detail", "context")


def _require_keys(obj: dict, keys, path: str, errs: list) -> bool:
    if not isinstance(obj, dict):
        errs.append(f"{path}: expected object, got {type(obj).__name__}")
        return False
    for k in keys:
        if k not in obj:
            errs.append(f"{path}: missing required key '{k}'")
    return True


def _validate_xy(obj, path: str, errs: list) -> None:
    if not (isinstance(obj, (list, tuple)) and len(obj) == 2
            and all(isinstance(v, (int, float)) for v in obj)):
        errs.append(f"{path}: expected [x, y] numeric pair")


def _validate_room(i: int, room: dict, segment_ids: set, text_region_ids: set, errs: list) -> str | None:
    p = f"rooms[{i}]"
    if not _require_keys(room, REQUIRED_ROOM_FIELDS, p, errs):
        return None
    rid = room.get("room_id")
    if not isinstance(rid, str) or not rid:
        errs.append(f"{p}.room_id: must be non-empty string")
        return None
    if room.get("room_type") not in ALLOWED_ROOM_TYPES:
        errs.append(f"{p}.room_type: '{room.get('room_type')}' not in {sorted(ALLOWED_ROOM_TYPES)}")
    poly = room.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        errs.append(f"{p}.polygon: expected list of >=3 [x,y] pairs")
    else:
        for j, pt in enumerate(poly):
            _validate_xy(pt, f"{p}.polygon[{j}]", errs)
    _validate_xy(room.get("centroid"), f"{p}.centroid", errs)
    if not isinstance(room.get("area"), (int, float)):
        errs.append(f"{p}.area: expected number")
    if not isinstance(room.get("bounding_walls"), list):
        errs.append(f"{p}.bounding_walls: expected list")
    else:
        for sid in room["bounding_walls"]:
            if sid not in segment_ids:
                errs.append(f"{p}.bounding_walls: unknown segment_id '{sid}'")
    if not isinstance(room.get("text_labels"), list):
        errs.append(f"{p}.text_labels: expected list")
    else:
        for tid in room["text_labels"]:
            if tid and tid not in text_region_ids:
                errs.append(f"{p}.text_labels: unknown text_region_id '{tid}'")
    return rid


def _validate_wall(i: int, seg: dict, junction_ids: set, errs: list) -> str | None:
    p = f"walls[{i}]"
    if not _require_keys(seg, REQUIRED_WALL_FIELDS, p, errs):
        return None
    sid = seg.get("segment_id")
    if not isinstance(sid, str) or not sid:
        errs.append(f"{p}.segment_id: must be non-empty string")
        return None
    geom = seg.get("geometry", {})
    _require_keys(geom, REQUIRED_GEOMETRY_FIELDS, f"{p}.geometry", errs)
    _validate_xy(geom.get("start"), f"{p}.geometry.start", errs)
    _validate_xy(geom.get("end"), f"{p}.geometry.end", errs)
    vis = seg.get("visual_properties", {})
    _require_keys(vis, REQUIRED_VISUAL_FIELDS, f"{p}.visual_properties", errs)
    rgb = vis.get("color_rgb")
    if rgb is not None and not (
        isinstance(rgb, (list, tuple)) and len(rgb) == 3
        and all(isinstance(v, (int, float)) for v in rgb)
    ):
        errs.append(f"{p}.visual_properties.color_rgb: expected [r,g,b] or null")
    top = seg.get("topology", {})
    _require_keys(top, REQUIRED_TOPOLOGY_FIELDS, f"{p}.topology", errs)
    for key in ("start_junction_type", "end_junction_type"):
        v = top.get(key)
        if v is not None and v not in ALLOWED_JUNCTION_TYPES:
            errs.append(f"{p}.topology.{key}: '{v}' not in {sorted(ALLOWED_JUNCTION_TYPES)}")
    for ref_key in ("start_junction_id", "end_junction_id"):
        ref = top.get(ref_key)
        if ref is not None and ref not in junction_ids:
            errs.append(f"{p}.topology.{ref_key}: unknown junction_id '{ref}'")
    if not isinstance(top.get("connected_segment_ids"), list):
        errs.append(f"{p}.topology.connected_segment_ids: expected list")
    if not isinstance(top.get("adjacent_room_ids"), list):
        errs.append(f"{p}.topology.adjacent_room_ids: expected list")
    sem = seg.get("semantic", {})
    _require_keys(sem, REQUIRED_SEMANTIC_FIELDS, f"{p}.semantic", errs)
    role = sem.get("functional_role")
    if role not in ALLOWED_FUNCTIONAL_ROLES:
        errs.append(f"{p}.semantic.functional_role: '{role}' not in {sorted(ALLOWED_FUNCTIONAL_ROLES)}")
    if role in CROSS_DOC_REQUIRED_ROLES and sem.get("requires_cross_document_validation") is not True:
        errs.append(f"{p}.semantic: role '{role}' requires requires_cross_document_validation=true")
    conf = sem.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        errs.append(f"{p}.semantic.confidence: expected number in [0,1]")
    return sid


def _validate_opening(i: int, op: dict, segment_ids: set, room_ids: set, errs: list) -> str | None:
    p = f"openings[{i}]"
    if not _require_keys(op, REQUIRED_OPENING_FIELDS, p, errs):
        return None
    oid = op.get("opening_id")
    if not isinstance(oid, str) or not oid:
        errs.append(f"{p}.opening_id: must be non-empty string")
        return None
    if op.get("type") not in ALLOWED_OPENING_TYPES:
        errs.append(f"{p}.type: '{op.get('type')}' not in {sorted(ALLOWED_OPENING_TYPES)}")
    _validate_xy(op.get("position"), f"{p}.position", errs)
    wsid = op.get("wall_segment_id")
    if wsid is not None and wsid not in segment_ids:
        errs.append(f"{p}.wall_segment_id: unknown segment_id '{wsid}'")
    crid = op.get("connects_room_ids")
    if crid is not None:
        if not isinstance(crid, list):
            errs.append(f"{p}.connects_room_ids: expected list or null")
        else:
            for rid in crid:
                if rid not in room_ids:
                    errs.append(f"{p}.connects_room_ids: unknown room_id '{rid}'")
    conf = op.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        errs.append(f"{p}.confidence: expected number in [0,1]")
    return oid


def _validate_text_region(
    i: int, region: dict, room_ids: set, segment_ids: set, grid_ids: set, errs: list,
) -> str | None:
    p = f"text_regions[{i}]"
    if not _require_keys(region, REQUIRED_TEXT_REGION_FIELDS, p, errs):
        return None
    tid = region.get("text_region_id")
    if not isinstance(tid, str) or not tid:
        errs.append(f"{p}.text_region_id: must be non-empty string")
        return None
    if region.get("classification") not in ALLOWED_TEXT_CLASSIFICATIONS:
        errs.append(f"{p}.classification: '{region.get('classification')}' not in {sorted(ALLOWED_TEXT_CLASSIFICATIONS)}")
    bbox = region.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)):
        errs.append(f"{p}.bbox: expected [x0,y0,x1,y1] numeric")
    if not isinstance(region.get("references"), list):
        errs.append(f"{p}.references: expected list")
    # enclosing_room_id and linked_entity_ids are validated in post-pass
    # after all IDs (rooms, grid_lines) are collected.
    if not isinstance(region.get("linked_entity_ids"), list):
        errs.append(f"{p}.linked_entity_ids: expected list")
    return tid


def _validate_grid_line(i: int, g: dict, errs: list) -> str | None:
    p = f"grid_lines[{i}]"
    if not _require_keys(g, REQUIRED_GRID_FIELDS, p, errs):
        return None
    gid = g.get("grid_id")
    if not isinstance(gid, str) or not gid:
        errs.append(f"{p}.grid_id: must be non-empty string")
        return None
    if g.get("axis") not in ALLOWED_GRID_AXES:
        errs.append(f"{p}.axis: '{g.get('axis')}' not in {sorted(ALLOWED_GRID_AXES)}")
    _validate_xy(g.get("start"), f"{p}.start", errs)
    _validate_xy(g.get("end"), f"{p}.end", errs)
    return gid


def _validate_junction(i: int, j: dict, errs: list) -> str | None:
    p = f"junctions[{i}]"
    if not _require_keys(j, REQUIRED_JUNCTION_FIELDS, p, errs):
        return None
    jid = j.get("junction_id")
    if not isinstance(jid, str) or not jid:
        errs.append(f"{p}.junction_id: must be non-empty string")
        return None
    if j.get("junction_type") not in ALLOWED_JUNCTION_TYPES:
        errs.append(f"{p}.junction_type: '{j.get('junction_type')}' not in {sorted(ALLOWED_JUNCTION_TYPES)}")
    _validate_xy(j.get("position"), f"{p}.position", errs)
    if not isinstance(j.get("connected_segment_ids"), list):
        errs.append(f"{p}.connected_segment_ids: expected list")
    return jid


def _validate_cross_ref(i: int, cr: dict, text_region_ids: set, errs: list) -> None:
    p = f"cross_references[{i}]"
    if not _require_keys(cr, REQUIRED_CROSS_REF_FIELDS, p, errs):
        return
    src = cr.get("source_text_region_id")
    if src not in text_region_ids:
        errs.append(f"{p}.source_text_region_id: unknown text_region_id '{src}'")


def validate(doc: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(doc, dict):
        return [f"root: expected object, got {type(doc).__name__}"]
    for k in TOP_LEVEL_KEYS:
        if k not in doc:
            errs.append(f"root: missing required key '{k}'")
    if errs:
        return errs

    meta = doc["metadata"]
    if _require_keys(meta, REQUIRED_METADATA_FIELDS.keys(), "metadata", errs):
        for k, t in REQUIRED_METADATA_FIELDS.items():
            if k in meta and not isinstance(meta[k], t):
                errs.append(f"metadata.{k}: expected {t.__name__}, got {type(meta[k]).__name__}")

    if not isinstance(doc["walls"], list):
        errs.append("walls: expected list")
        return errs
    if not isinstance(doc["junctions"], list):
        errs.append("junctions: expected list")
        return errs
    if not isinstance(doc["rooms"], list):
        errs.append("rooms: expected list")
        return errs
    if not isinstance(doc["openings"], list):
        errs.append("openings: expected list")
        return errs
    if not isinstance(doc["text_regions"], list):
        errs.append("text_regions: expected list")
        return errs
    if not isinstance(doc["grid_lines"], list):
        errs.append("grid_lines: expected list")
        return errs
    if not isinstance(doc["cross_references"], list):
        errs.append("cross_references: expected list")
        return errs

    # Gather IDs first so cross-references validate
    junction_ids: set[str] = set()
    for i, j in enumerate(doc["junctions"]):
        jid = _validate_junction(i, j, errs)
        if jid:
            if jid in junction_ids:
                errs.append(f"junctions[{i}].junction_id: duplicate '{jid}'")
            junction_ids.add(jid)

    segment_ids: set[str] = set()
    for i, seg in enumerate(doc["walls"]):
        sid = _validate_wall(i, seg, junction_ids, errs)
        if sid:
            if sid in segment_ids:
                errs.append(f"walls[{i}].segment_id: duplicate '{sid}'")
            segment_ids.add(sid)

    room_ids: set[str] = set()
    text_region_ids: set[str] = set()
    # Text regions first so rooms can reference them
    grid_ids_tmp: set[str] = set()
    for i, region in enumerate(doc["text_regions"]):
        # We don't know room_ids yet; we'll re-validate enclosing_room_id against the final set after rooms are collected.
        tid = _validate_text_region(i, region, set(), segment_ids, grid_ids_tmp, errs)
        if tid:
            if tid in text_region_ids:
                errs.append(f"text_regions[{i}].text_region_id: duplicate '{tid}'")
            text_region_ids.add(tid)
    for i, room in enumerate(doc["rooms"]):
        rid = _validate_room(i, room, segment_ids, text_region_ids, errs)
        if rid:
            if rid in room_ids:
                errs.append(f"rooms[{i}].room_id: duplicate '{rid}'")
            room_ids.add(rid)
    grid_ids: set[str] = set()
    for i, g in enumerate(doc["grid_lines"]):
        gid = _validate_grid_line(i, g, errs)
        if gid:
            if gid in grid_ids:
                errs.append(f"grid_lines[{i}].grid_id: duplicate '{gid}'")
            grid_ids.add(gid)

    # Post-pass: check cross-references now that all ID sets are populated
    known_linkable = segment_ids | grid_ids | room_ids
    for i, region in enumerate(doc["text_regions"]):
        erid = region.get("enclosing_room_id")
        if erid is not None and erid not in room_ids:
            errs.append(f"text_regions[{i}].enclosing_room_id: unknown room_id '{erid}'")
        linked = region.get("linked_entity_ids") or []
        for eid in linked:
            if eid and eid not in known_linkable:
                errs.append(f"text_regions[{i}].linked_entity_ids: unknown entity_id '{eid}'")

    # Wall adjacent_room_ids and connected_segment_ids reference IDs we only
    # know after all entities are collected, so they're validated here.
    for i, seg in enumerate(doc["walls"]):
        top = seg.get("topology", {})
        adj = top.get("adjacent_room_ids")
        if isinstance(adj, list):
            for rid in adj:
                if rid not in room_ids:
                    errs.append(
                        f"walls[{i}].topology.adjacent_room_ids: unknown room_id '{rid}'"
                    )
        csids = top.get("connected_segment_ids")
        if isinstance(csids, list):
            for sid in csids:
                if sid not in segment_ids:
                    errs.append(
                        f"walls[{i}].topology.connected_segment_ids: unknown segment_id '{sid}'"
                    )

    for i, j in enumerate(doc["junctions"]):
        csids = j.get("connected_segment_ids")
        if isinstance(csids, list):
            for sid in csids:
                if sid not in segment_ids:
                    errs.append(
                        f"junctions[{i}].connected_segment_ids: unknown segment_id '{sid}'"
                    )

    opening_ids: set[str] = set()
    for i, op in enumerate(doc["openings"]):
        oid = _validate_opening(i, op, segment_ids, room_ids, errs)
        if oid:
            if oid in opening_ids:
                errs.append(f"openings[{i}].opening_id: duplicate '{oid}'")
            opening_ids.add(oid)

    for i, cr in enumerate(doc["cross_references"]):
        _validate_cross_ref(i, cr, text_region_ids, errs)

    # Entity counts consistency
    ec = doc["metadata"].get("entity_counts", {})
    if isinstance(ec, dict):
        expected = {
            "rooms": len(doc["rooms"]),
            "walls": len(doc["walls"]),
            "openings": len(doc["openings"]),
            "text_regions": len(doc["text_regions"]),
            "grid_lines": len(doc["grid_lines"]),
            "junctions": len(doc["junctions"]),
            "cross_references": len(doc["cross_references"]),
        }
        for k, v in expected.items():
            if ec.get(k) != v:
                errs.append(f"metadata.entity_counts.{k}: {ec.get(k)} != len({k})={v}")

    return errs
