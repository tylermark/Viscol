# Construction Intelligence — Phase 1 Extraction Infrastructure

## 1. What this project is

This is **Phase 1 data infrastructure** for a construction foundation model (H0). Not a wall detector. Not a polished utility. **An extractor that turns vector AEC drawings into structured, FM-ready training signal at scale**, plus the measurement apparatus that lets us evaluate whether downstream foundation models reach senior-coordinator-level understanding.

Read this file at the start of every session. Re-read it when scope questions arise.

## 2. Research frame — why this pipeline exists

The program's north-star hypothesis is **H0 (Construction Foundation Model Hypothesis)**: a foundation model trained on a sufficiently large and diverse corpus of multi-format AEC documents can develop expert-level semantic understanding equivalent to a senior coordinator. H1, H2, H3 are sub-hypotheses that test *how* that understanding is realized (explicit semantic layer? multi-document encoder? knowledge graph?).

This pipeline exists because the FM needs training signal. Drawings in the wild are PDFs with vector primitives but no structured annotations. Human coordinators extract rooms, walls, openings, dimensions, cross-references, and grid systems mentally when they read a drawing. Phase 1's job is to **extract that structure programmatically at scale so a Phase 2 foundation model can train on it**.

Update the wiki path below to match your local LLM-Wiki folder location.

```python
WIKI_PATH = r"C:\Users\tyler\LLM-Wiki"
```

### Research context (read these before major decisions)

| File | Purpose |
|------|---------|
| `[WIKI_PATH]/wiki/hypotheses/construction-foundation-model-hypothesis.md` | **H0 — THE MAIN HYPOTHESIS.** Every scope decision in this pipeline should serve H0. |
| `[WIKI_PATH]/wiki/hypotheses/semantic-layer-hypothesis.md` | H1 — geometry→role bridge. Motivates *why* we extract roles, not just geometry. |
| `[WIKI_PATH]/wiki/hypotheses/multi-document-inference-hypothesis.md` | H2 — single doc insufficient. Motivates cross-sheet reference extraction. |
| `[WIKI_PATH]/wiki/hypotheses/knowledge-graph-superiority-hypothesis.md` | H3 — graph > embedding. Motivates the graph output schema. |

### What we measure against

Phase 1 is **not measured by wall F1**. Prior iterations discovered wall F1 ≈ 0.45 on real drawings is near the rule-based ceiling; this is recorded as a baseline (see §10 Legacy baseline) but is **not the primary metric going forward**. The primary metrics for Phase 1 are:

1. **Entity coverage**: fraction of drawings for which we extract rooms, walls, openings, text regions, grid, cross-references
2. **Scale**: number of drawings processed to a usable format
3. **Format stability**: schema compliance across diverse drawing sources (archcad400k, MLSTRUCT-FP, real-world plans)
4. **Coordinator-task evaluations**: derived benchmarks (e.g., "which rooms are bathrooms?", "what is on the other side of wall X?", "which walls reference schedule line W-3?")

Wall F1 against hand-marked ground truth is a secondary diagnostic, not an optimization target.

## 3. Pipeline architecture

```text
Vector PDF (single page)
    │
    ▼
Stage 1: Path Extraction               (unchanged)
    PyMuPDF → raw primitives: lines, polylines, arcs, fills, text-with-position
    │
    ▼
Stage 2: Primitive Classification      (refactored)
    Classify each primitive into: wall_candidate | annotation | dimension | hatch |
                                   text_label | arc_symbol | grid_line | schedule_note |
                                   fixture | unknown
    │
    ▼
Stage 3: Wall Detection                (unchanged)
    HFV2013-style parallel pair detection + length-weighted thickness clustering +
    fallback for uniform stroke drawings
    │
    ▼
Stage 4: Topology                      (unchanged)
    Junction clustering + endpoint-on-body split + graph construction
    │
    ▼
Stage 5: Room Detection                (NEW)
    Polygonize wall graph → candidate rooms
    Text-label-in-polygon matching → room names + types
    │
    ▼
Stage 6: Opening Detection             (NEW)
    Arc primitives ± 90°±tol → door swings → opening positions
    Windows via symbol-pair or break-in-wall detection (best-effort)
    │
    ▼
Stage 7: Text Region Classification    (NEW)
    Classify each text block as: room_label | room_number | sheet_callout |
                                  dimension | wall_schedule_tag | note | title |
                                  grid_label | unknown
    Cross-reference resolution (e.g., "A302" → sheet A302 reference)
    │
    ▼
Stage 8: Grid Detection                (NEW)
    Long near-axis-aligned lines that terminate at letter/number callouts →
    structural/architectural grid system
    │
    ▼
Stage 9: Semantic Annotation           (refactored — no longer final stage)
    Per-wall role assignment using room context (wet_wall ← bounds BATH room, etc.)
    Per-room type assignment using fixture proximity and label matching
    All assignments carry rule_triggered and requires_cross_document_validation flags
    │
    ▼
Stage 10: Graph Assembly               (NEW)
    Emit the structured graph: nodes = {rooms, walls, openings, text_regions, grid_lines},
    edges = {bounds, opens_through, labeled_by, references, on_grid}
    │
    ▼
Structured graph JSON (one per drawing)
```

## 4. Output schema

The canonical output is a **structured graph per drawing**, FM-training-ready. Every field present on every entity; unknown values get `null`, not a missing key.

```jsonc
{
  "metadata": {
    "source_pdf": "filename.pdf",
    "extraction_date": "YYYY-MM-DD",
    "pipeline_version": "0.6.0",
    "coordinate_system": "pdf_points_bottom_left_origin",
    "page_size": [width, height],
    "entity_counts": {
      "rooms": 0,
      "walls": 0,
      "openings": 0,
      "text_regions": 0,
      "grid_lines": 0,
      "junctions": 0,
      "cross_references": 0
    }
  },
  "rooms": [
    {
      "room_id": "uuid-v4",
      "polygon": [[x, y], ...],
      "area": 0.0,
      "centroid": [x, y],
      "bounding_walls": ["segment_id", ...],
      "text_labels": ["text_region_id", ...],
      "room_name": "Studio Unit 1.1a" | null,
      "room_number": "203" | null,
      "room_type": "unit | bathroom | kitchen | stair | hallway | mechanical | unknown",
      "requires_cross_document_validation": false
    }
  ],
  "walls": [
    {
      "segment_id": "uuid-v4",
      "geometry": {
        "start": [x, y],
        "end": [x, y],
        "centerline_length": 0.0,
        "angle_degrees": 0.0,
        "thickness": 0.0
      },
      "visual_properties": {
        "line_weight": 0.0,
        "color_rgb": [0, 0, 0],
        "is_dashed": false,
        "layer_name": null
      },
      "topology": {
        "start_junction_id": "uuid-v4",
        "end_junction_id": "uuid-v4",
        "start_junction_type": "corner | t-junction | x-junction | endpoint",
        "end_junction_type": "corner | t-junction | x-junction | endpoint",
        "connected_segment_ids": [],
        "adjacent_room_ids": []
      },
      "semantic": {
        "functional_role": "interior_partition | bearing_wall | wet_wall | demising | exterior | unknown",
        "confidence": 0.0,
        "rule_triggered": "string or null",
        "requires_cross_document_validation": false
      }
    }
  ],
  "openings": [
    {
      "opening_id": "uuid-v4",
      "type": "door | window | unknown",
      "position": [x, y],
      "width": 0.0,
      "swing_arc": [[x, y], [x, y], [x, y]] | null,
      "wall_segment_id": "uuid-v4" | null,
      "connects_room_ids": ["room_id", "room_id"] | null,
      "confidence": 0.0,
      "rule_triggered": "string or null"
    }
  ],
  "text_regions": [
    {
      "text_region_id": "uuid-v4",
      "text": "Studio Unit 1.1a",
      "bbox": [x0, y0, x1, y1],
      "classification": "room_label | room_number | sheet_callout | dimension | wall_schedule_tag | note | title | grid_label | unknown",
      "references": ["sheet_id_or_null"],
      "enclosing_room_id": "room_id or null",
      "linked_entity_ids": ["wall_or_grid_id"],
      "rule_triggered": "string",
      "requires_cross_document_validation": false
    }
  ],
  "grid_lines": [
    {
      "grid_id": "uuid-v4",
      "axis": "horizontal | vertical",
      "label": "A" | "1" | null,
      "start": [x, y],
      "end": [x, y],
      "rule_triggered": "string"
    }
  ],
  "junctions": [
    {
      "junction_id": "uuid-v4",
      "position": [x, y],
      "junction_type": "corner | t-junction | x-junction | endpoint",
      "connected_segment_ids": []
    }
  ],
  "cross_references": [
    {
      "cross_reference_id": "uuid-v4",
      "source_text_region_id": "text_region_id",
      "target_sheet": "A302",
      "target_detail": "1" | null,
      "context": "room_label | sheet_callout | note",
      "rule_triggered": "string",
      "requires_cross_document_validation": true
    }
  ]
}
```

### Schema rules
- All entity-type top-level keys always present, even if empty list.
- Every entity has a UUID-v4 `*_id` unique across the document.
- All geometry in PDF points, bottom-left origin (Y-flipped from PyMuPDF native).
- `requires_cross_document_validation: true` on any role/classification that depends on non-architectural information (bearing_wall, wet_wall, and any cross_reference).

## 5. What's in scope for Phase 1

### In scope
1. **Rooms**: detected by polygonizing the wall graph, named by text labels, typed by fixture/label proximity
2. **Walls**: centerlines + thickness + role (geometric roles only, per H1 we cannot get functional roles without cross-doc)
3. **Openings**: doors via arc detection, windows on a best-effort basis
4. **Text regions**: classified and linked to spatial entities
5. **Grid lines**: structural/architectural grid with letter/number labels
6. **Cross-references**: in-drawing callouts like "A302", "see A7.1" parsed into structured link records

### Out of scope for Phase 1
1. **Multi-document fusion** (H2 territory — single drawing at a time)
2. **ML / neural models** (rule-based only — see §8)
3. **Functional roles requiring non-visual info** (wet_wall confirmation needs MEP plan; we flag `requires_cross_document_validation: true` and stop)
4. **IFC / DXF ingestion** (PDF only for Phase 1)
5. **Drawing-level semantics** (floor level, phase, discipline — these come from sheet metadata, Phase 2)
6. **Polished per-drawing accuracy** (we care about scale and structure, not nail-in-the-coffin F1)

## 6. Measurement protocol

### Primary metrics (Phase 1 success)
1. **Entity coverage per drawing** — fraction of drawings where each entity type is non-empty
2. **Schema validity rate** — fraction that pass `schema_validator.validate()`
3. **Process throughput** — wall-clock seconds per drawing (<60s target, <300s hard cap)
4. **Batch scale** — demonstrated on >=100 diverse drawings end-to-end

### Secondary metrics (diagnostics, not targets)
1. **Wall F1** against hand-marked ground truth (legacy; v0.4.1 = 0.45 across 11 plans)
2. **Room naming precision** — fraction of detected rooms whose matched text label is inside the polygon (not a nearby mis-match)
3. **Opening-on-wall consistency** — fraction of openings whose `wall_segment_id` is non-null
4. **Cross-reference resolution rate** — fraction of detected cross-references that point to plausible sheet IDs

### Coordinator-task benchmarks (the real Phase 1 deliverable)
These are H0-relevant. Built over batch extracted output:
1. "Which rooms are bathrooms/kitchens?" (from `room_type`)
2. "What rooms are adjacent to room X?" (from `bounding_walls` ∩)
3. "What's on the other side of wall X?" (from `adjacent_room_ids`)
4. "Which walls are referenced by schedule note W-3?" (from `cross_references`)
5. "Which rooms require cross-document validation for wet-wall classification?" (from the flag)

Each coordinator-task benchmark has a precision/recall/F1 computed against a manually-audited subset. These numbers — not wall F1 — are what Phase 1 is accountable to.

## 7. Project structure

```text
project-root/
├── CLAUDE.md                  ← this file
├── config.yaml                ← all tunable thresholds
├── pipeline.py                ← entry point: python pipeline.py <pdf>
├── schema_validator.py        ← validates against §4 schema
├── stages/
│   ├── __init__.py
│   ├── _config.py             ← config loader
│   ├── extract_paths.py       ← Stage 1
│   ├── classify_properties.py ← Stage 2
│   ├── detect_walls.py        ← Stage 3
│   ├── build_topology.py      ← Stage 4
│   ├── detect_rooms.py        ← Stage 5 (NEW)
│   ├── detect_openings.py     ← Stage 6 (NEW)
│   ├── classify_text.py       ← Stage 7 (NEW)
│   ├── detect_grid.py         ← Stage 8 (NEW)
│   ├── assign_semantics.py    ← Stage 9 (room-aware)
│   └── assemble_graph.py      ← Stage 10 (NEW)
├── evaluate.py                ← wall F1 diagnostic (legacy)
├── evaluate_correct.py        ← wall F1 vs hand-marked (legacy)
├── benchmark.py               ← cross-drawing stability + per-entity coverage
├── benchmark_gt.py            ← GT wall F1 diagnostic (legacy)
├── visualize.py               ← annotated PDF renderer
├── output/                    ← pipeline JSON outputs (gitignored)
├── evaluation/
│   ├── ground_truth/          ← manually labeled JSONs
│   └── results/               ← evaluation run outputs
└── tests/
```

## 8. Conventions

- **Language**: Python 3.10+
- **Core dependencies**: PyMuPDF, Shapely, NetworkX, NumPy, PyYAML
- **No ML model dependencies in Phase 1.** The explicit-rule baseline is the contrast Phase 2 (FM) must outperform. Without a clean baseline, H1/H3 ablations are unmeasurable.
- **All thresholds in `config.yaml`.** No hardcoded numbers in `.py` files.
- **Coordinates**: PDF points, origin at bottom-left.
- **One test per pipeline stage minimum.** Tests run with `pytest`.
- **Fail loudly** on unexpected geometry (negative thickness, zero-length segment, disconnected graph) — raise with a clear message.

## 9. Principles

- **Infrastructure first, accuracy second.** A messy graph extracted from 1000 drawings is more valuable to H0 than a pristine extraction from 10. Scale and structure matter more than per-drawing F1.
- **Every entity names its source rule.** `rule_triggered` on every classification. Black-box outputs don't support H1/H3 research.
- **H2 honesty flag.** `requires_cross_document_validation: true` on any claim that can't be verified from one drawing alone. This is the pipeline's empirical footprint of H2.
- **Rule transparency over cleverness.** When in doubt, pick the simpler rule and mark confidence lower. Don't build opaque heuristics that hide which signal fired.
- **Domain knowledge is authoritative.** When wiki domain pages conflict with geometric heuristics, domain rules win.

## 10. Legacy baseline (for H1/H3 comparison)

v0.4.1 of the wall-only pipeline achieved:
- Mean wall F1 = 0.45 across 11 hand-marked plans (perp_tol=12pt, parallel_tol=10°)
- Median F1 = 0.49, range 0.25–0.62
- Mean precision = 0.45, mean recall = 0.54

This is the rule-based geometry-only baseline a future FM must exceed. Do not re-optimize this number; it's a diagnostic artifact, not a target. The v0.4.1 code is preserved in Stages 1–4 (wall detection) and in the legacy evaluators (`evaluate.py`, `evaluate_correct.py`).

## 11. What done looks like for Phase 1

Phase 1 concludes when all of the following are true:
1. Pipeline extracts the full §4 schema (rooms, walls, openings, text_regions, grid, cross_references) without erroring on >=90% of a 100-drawing sample from archcad400k or similar
2. Output validates against `schema_validator.validate()` on every successful run
3. Coverage metrics (§6) are reported over the batch
4. A short write-up exists describing: what the pipeline reliably extracts, what fails, which entity types have the strongest signal, and what architecture decisions this implies for Phase 2 (FM training)

After that we start Phase 2: use the Phase 1 corpus to train an FM and evaluate it against coordinator-task benchmarks and the H0 falsification criteria.
