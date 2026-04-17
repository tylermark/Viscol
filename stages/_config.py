"""Config loader. Single entry point so no .py file hardcodes thresholds."""

from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "coordinate_flip_y",
    "wall_min_line_weight",
    "annotation_max_line_weight",
    "dimension_text_proximity",
    "wall_color_max_darkness",
    "parallel_angle_tolerance",
    "orthogonal_angle_tolerance",
    "wall_min_thickness",
    "wall_max_thickness",
    "wall_min_overlap_ratio",
    "thickness_cluster_bin_width",
    "thickness_cluster_min_support",
    "thickness_cluster_tolerance",
    "thickness_cluster_max_modes",
    "thickness_cluster_min_candidates",
    "thickness_cluster_mode_max_realistic",
    "wall_candidate_min_count",
    "wall_fallback_min_paths",
    "wall_fallback_min_length",
    "wall_fallback_max_candidates",
    "junction_snap_distance",
    "isolated_max_length",
    "gap_close_distance",
    "gap_close_max_angle_drift",
    "junction_anchor_enabled",
    "junction_anchor_tag_only",
    "junction_anchor_min_segments",
    "junction_anchor_min_cluster_size",
    "junction_anchor_snap_tolerance",
    "junction_raw_snap_radius",
    "demising_min_length",
    "semantic_min_confidence",
    "wet_zone_proximity",
    "bearing_min_continuous_length",
    "wet_zone_labels",
    "evaluation_match_proximity",
    # v0.6.0 — multi-entity extraction
    "room_min_area",
    "room_max_aspect_ratio",
    "room_min_short_dimension",
    "room_boundary_tolerance",
    "room_gap_close_enabled",
    "room_gap_close_max_distance",
    "room_gap_close_max_angle_drift_deg",
    "room_type_label_patterns",
    "door_arc_span_min_deg",
    "door_arc_span_max_deg",
    "door_arc_chord_min",
    "door_arc_chord_max",
    "door_to_wall_distance",
    "exterior_tolerance",
    "grid_line_axis_tolerance_deg",
    "text_room_number_pattern",
    "text_sheet_callout_pattern",
    "text_dimension_pattern",
    "text_grid_label_pattern",
    "text_grid_label_blocklist",
    "grid_line_min_length",
    "grid_label_proximity",
    "grid_endpoint_near_boundary",
}


def load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got {type(config).__name__}")
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")
    return config
