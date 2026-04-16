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
    "junction_snap_distance",
    "demising_min_length",
    "semantic_min_confidence",
    "wet_zone_proximity",
    "bearing_min_continuous_length",
    "wet_zone_labels",
    "evaluation_match_proximity",
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
