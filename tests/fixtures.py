"""Synthetic test fixtures — small hand-built inputs for stage tests."""

from __future__ import annotations

import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def default_config() -> dict:
    """Load the real config.yaml so tests share thresholds with production."""
    import yaml

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_path(
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    stroke_width: float = 1.0,
    stroke_rgb: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    is_dashed: bool = False,
    dash_pattern=None,
    layer_name: str | None = None,
    page_num: int = 0,
    kind: str = "line",
    path_id: str | None = None,
) -> dict:
    return {
        "id": path_id or str(uuid.uuid4()),
        "kind": kind,
        "points": [list(p1), list(p2)],
        "stroke_width": stroke_width,
        "stroke_rgb": list(stroke_rgb) if stroke_rgb is not None else None,
        "fill_rgb": None,
        "dash_pattern": dash_pattern,
        "is_dashed": is_dashed,
        "layer_name": layer_name,
        "page_num": page_num,
    }


def parallel_wall_pair(
    start: tuple[float, float],
    length: float,
    thickness: float,
    *,
    horizontal: bool = True,
    stroke_width: float = 1.0,
) -> list[dict]:
    """Two parallel line paths forming a wall of the given length and thickness."""
    if horizontal:
        p1 = make_path(
            start,
            (start[0] + length, start[1]),
            stroke_width=stroke_width,
        )
        p2 = make_path(
            (start[0], start[1] + thickness),
            (start[0] + length, start[1] + thickness),
            stroke_width=stroke_width,
        )
    else:
        p1 = make_path(
            start,
            (start[0], start[1] + length),
            stroke_width=stroke_width,
        )
        p2 = make_path(
            (start[0] + thickness, start[1]),
            (start[0] + thickness, start[1] + length),
            stroke_width=stroke_width,
        )
    return [p1, p2]


def make_wall_record(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    thickness: float = 4.0,
    stroke_width: float = 1.0,
    stroke_rgb=(0.0, 0.0, 0.0),
) -> dict:
    """A post-Stage-3 wall record used by Stage 4 / Stage 5 tests."""
    import math

    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy)
    angle = math.degrees(math.atan2(ey - sy, ex - sx)) % 180.0
    return {
        "segment_id": str(uuid.uuid4()),
        "start": [float(sx), float(sy)],
        "end": [float(ex), float(ey)],
        "length": float(length),
        "angle_degrees": float(angle),
        "thickness": float(thickness),
        "stroke_width": float(stroke_width),
        "color_rgb": list(stroke_rgb),
        "is_dashed": False,
        "layer_name": None,
        "source_path_ids": [],
        "rules_passed": ["parallel_pairing", "thickness_in_range", "orthogonality"],
        "rules_failed": [],
    }
