"""Floor plan semantic extraction pipeline stages.

Submodules:
    extract_paths       — Stage 1: vector PDF → raw primitives
    classify_properties — Stage 2: primitives → element type candidates
    detect_walls        — Stage 3: wall candidates → centerline segments
    build_topology      — Stage 4: segments → junctions + graph
    assign_semantics    — Stage 5: graph → functional roles

Import the public functions from the submodules, not from the package root, so
test code can still reference the submodules themselves (e.g. to monkeypatch fitz).
"""
