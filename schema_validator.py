"""Schema validator for pipeline output. Custom (no jsonschema dep)."""

ALLOWED_FUNCTIONAL_ROLES = {
    "interior_partition",
    "bearing_wall",
    "wet_wall",
    "demising",
    "exterior",
    "unknown",
}

ALLOWED_JUNCTION_TYPES = {
    "corner",
    "t-junction",
    "x-junction",
    "endpoint",
}

CROSS_DOC_REQUIRED_ROLES = {"bearing_wall", "wet_wall"}

REQUIRED_METADATA_FIELDS = {
    "source_pdf": str,
    "extraction_date": str,
    "pipeline_version": str,
    "total_segments": int,
    "coordinate_system": str,
}

REQUIRED_SEGMENT_FIELDS = ("segment_id", "geometry", "visual_properties", "topology", "semantic")
REQUIRED_GEOMETRY_FIELDS = ("start", "end", "centerline_length", "angle_degrees", "thickness")
REQUIRED_VISUAL_FIELDS = ("line_weight", "color_rgb", "is_dashed", "layer_name")
REQUIRED_TOPOLOGY_FIELDS = (
    "start_junction_id",
    "end_junction_id",
    "start_junction_type",
    "end_junction_type",
    "connected_segment_ids",
)
REQUIRED_SEMANTIC_FIELDS = (
    "functional_role",
    "confidence",
    "rule_triggered",
    "requires_cross_document_validation",
)
REQUIRED_JUNCTION_FIELDS = ("junction_id", "position", "junction_type", "connected_segment_ids")


def _require_keys(obj: dict, keys, path: str, errs: list):
    if not isinstance(obj, dict):
        errs.append(f"{path}: expected object, got {type(obj).__name__}")
        return False
    for k in keys:
        if k not in obj:
            errs.append(f"{path}: missing required key '{k}'")
    return True


def validate(doc: dict) -> list[str]:
    """Return a list of validation error strings. Empty list means valid."""
    errs: list[str] = []

    if not isinstance(doc, dict):
        return [f"root: expected object, got {type(doc).__name__}"]

    for top in ("metadata", "segments", "junctions"):
        if top not in doc:
            errs.append(f"root: missing required key '{top}'")
    if errs:
        return errs

    meta = doc["metadata"]
    if _require_keys(meta, REQUIRED_METADATA_FIELDS.keys(), "metadata", errs):
        for k, t in REQUIRED_METADATA_FIELDS.items():
            if k in meta and not isinstance(meta[k], t):
                errs.append(f"metadata.{k}: expected {t.__name__}, got {type(meta[k]).__name__}")

    segments = doc["segments"]
    junctions = doc["junctions"]
    if not isinstance(segments, list):
        errs.append(f"segments: expected list, got {type(segments).__name__}")
        return errs
    if not isinstance(junctions, list):
        errs.append(f"junctions: expected list, got {type(junctions).__name__}")
        return errs

    segment_ids: set[str] = set()
    junction_ids: set[str] = set()

    for i, seg in enumerate(segments):
        p = f"segments[{i}]"
        if not _require_keys(seg, REQUIRED_SEGMENT_FIELDS, p, errs):
            continue
        sid = seg.get("segment_id")
        if not isinstance(sid, str) or not sid:
            errs.append(f"{p}.segment_id: must be non-empty string")
        elif sid in segment_ids:
            errs.append(f"{p}.segment_id: duplicate id '{sid}'")
        else:
            segment_ids.add(sid)

        geom = seg.get("geometry", {})
        _require_keys(geom, REQUIRED_GEOMETRY_FIELDS, f"{p}.geometry", errs)
        for pt_key in ("start", "end"):
            pt = geom.get(pt_key)
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
                errs.append(f"{p}.geometry.{pt_key}: expected [x, y] numeric pair")

        vis = seg.get("visual_properties", {})
        _require_keys(vis, REQUIRED_VISUAL_FIELDS, f"{p}.visual_properties", errs)
        rgb = vis.get("color_rgb")
        if rgb is not None:
            if not (isinstance(rgb, (list, tuple)) and len(rgb) == 3 and all(isinstance(v, (int, float)) for v in rgb)):
                errs.append(f"{p}.visual_properties.color_rgb: expected [r, g, b] numeric triple or null")

        top = seg.get("topology", {})
        _require_keys(top, REQUIRED_TOPOLOGY_FIELDS, f"{p}.topology", errs)
        for jt_key in ("start_junction_type", "end_junction_type"):
            v = top.get(jt_key)
            if v is not None and v not in ALLOWED_JUNCTION_TYPES:
                errs.append(f"{p}.topology.{jt_key}: '{v}' not in {sorted(ALLOWED_JUNCTION_TYPES)}")
        csids = top.get("connected_segment_ids")
        if not isinstance(csids, list):
            errs.append(f"{p}.topology.connected_segment_ids: expected list")

        sem = seg.get("semantic", {})
        _require_keys(sem, REQUIRED_SEMANTIC_FIELDS, f"{p}.semantic", errs)
        role = sem.get("functional_role")
        if role not in ALLOWED_FUNCTIONAL_ROLES:
            errs.append(f"{p}.semantic.functional_role: '{role}' not in {sorted(ALLOWED_FUNCTIONAL_ROLES)}")
        cross = sem.get("requires_cross_document_validation")
        if role in CROSS_DOC_REQUIRED_ROLES and cross is not True:
            errs.append(f"{p}.semantic: role '{role}' requires requires_cross_document_validation=true")
        conf = sem.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            errs.append(f"{p}.semantic.confidence: expected number in [0, 1]")

    for i, j in enumerate(junctions):
        p = f"junctions[{i}]"
        if not _require_keys(j, REQUIRED_JUNCTION_FIELDS, p, errs):
            continue
        jid = j.get("junction_id")
        if not isinstance(jid, str) or not jid:
            errs.append(f"{p}.junction_id: must be non-empty string")
        elif jid in junction_ids:
            errs.append(f"{p}.junction_id: duplicate id '{jid}'")
        else:
            junction_ids.add(jid)
        jt = j.get("junction_type")
        if jt not in ALLOWED_JUNCTION_TYPES:
            errs.append(f"{p}.junction_type: '{jt}' not in {sorted(ALLOWED_JUNCTION_TYPES)}")
        pos = j.get("position")
        if not (isinstance(pos, (list, tuple)) and len(pos) == 2 and all(isinstance(v, (int, float)) for v in pos)):
            errs.append(f"{p}.position: expected [x, y] numeric pair")
        csids = j.get("connected_segment_ids")
        if not isinstance(csids, list):
            errs.append(f"{p}.connected_segment_ids: expected list")
        else:
            for ref in csids:
                if ref not in segment_ids:
                    errs.append(f"{p}.connected_segment_ids: references unknown segment_id '{ref}'")

    for i, seg in enumerate(segments):
        p = f"segments[{i}]"
        top = seg.get("topology", {})
        for ref_key in ("start_junction_id", "end_junction_id"):
            ref = top.get(ref_key)
            if ref is not None and ref not in junction_ids:
                errs.append(f"{p}.topology.{ref_key}: references unknown junction_id '{ref}'")
        csids = top.get("connected_segment_ids")
        if isinstance(csids, list):
            for ref in csids:
                if ref not in segment_ids:
                    errs.append(f"{p}.topology.connected_segment_ids: references unknown segment_id '{ref}'")

    meta_total = doc["metadata"].get("total_segments")
    if isinstance(meta_total, int) and meta_total != len(segments):
        errs.append(f"metadata.total_segments ({meta_total}) != len(segments) ({len(segments)})")

    return errs
