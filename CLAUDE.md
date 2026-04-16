# Floor Plan Semantic Extraction Engine — Operating Manual

You are building the Phase 1 semantic extraction pipeline for a construction intelligence research program. This file is your operating manual. Read it at the start of every session.

---

## 1. Purpose

This pipeline ingests **vector PDF floor plans** and outputs **structured JSON** where every detected wall segment carries geometry, visual properties, topology context, and a preliminary semantic functional role.

This is a **measurement instrument first, a utility second.** Every pipeline run against labeled floor plans is a data point for testing the Semantic Layer Hypothesis (H1): the claim that geometry alone is insufficient to reliably infer functional role, and that an explicit semantic layer is required to bridge drawing geometry and construction intent.

The wet-wall binary classification experiment is the **first benchmark probe** — not the end goal. The end goal is a construction foundation model that understands building elements at the level of a senior coordinator.

---

## 2. Research Context

This project sits downstream of the LLM-Wiki research reference. Before making any significant architectural decision, consult the relevant wiki pages.

**Update the path below to match your local LLM-Wiki folder location.**

```
WIKI_PATH = "C:\Users\tyler\LLM-Wiki"
```

### Hypotheses (evaluation framework)

| File | What it tells you |
|------|-------------------|
| `[WIKI_PATH]/wiki/hypotheses/semantic-layer-hypothesis.md` | H1 — the core question this pipeline probes. Read the Refinement v2 section carefully. |
| `[WIKI_PATH]/wiki/hypotheses/multi-document-inference-hypothesis.md` | H2 — why single-document extraction is only the first step. |
| `[WIKI_PATH]/wiki/hypotheses/knowledge-graph-superiority-hypothesis.md` | H3 — informs the graph construction design in Stage 4. |
| `[WIKI_PATH]/wiki/hypotheses/construction-foundation-model-hypothesis.md` | H0 — the north star this pipeline feeds into. |

### Domain knowledge (read before touching Stage 5)

These pages encode practitioner expertise that cannot be sourced from papers. Never override them without explicit instruction.

- `[WIKI_PATH]/wiki/domains/wet-wall.md`
- `[WIKI_PATH]/wiki/domains/bearing-wall.md`
- `[WIKI_PATH]/wiki/domains/cfs-panel.md`
- `[WIKI_PATH]/wiki/domains/panelization-rules.md`
- `[WIKI_PATH]/wiki/domains/bim-to-fabrication-pipeline.md`

### Key papers (read before touching each stage)

| Paper | Stage it informs |
|-------|-----------------|
| `[WIKI_PATH]/wiki/papers/unsupervised-wall-detector-floorplans.md` | Stage 3 — HFV2013's six geometric wall assumptions are the rule basis |
| `[WIKI_PATH]/wiki/papers/raster-to-vector-floorplan.md` | Stage 4 — junction-first topology approach |
| `[WIKI_PATH]/wiki/papers/floorplancad-panoptic-symbol-spotting.md` | Stage 3 — vector CAD benchmark context |
| `[WIKI_PATH]/wiki/papers/shortcut-learning.md` | Stage 5 — why rule-based priors resist notation shortcuts |
| `[WIKI_PATH]/wiki/papers/texture-vs-shape-bias.md` | Stage 5 — why VLMs latch onto drafting conventions over functional role |

---

## 3. Pipeline Architecture

```
Vector PDF
    │
    ▼
Stage 1: Path Extraction
    PyMuPDF → raw geometric primitives
    (lines, polylines, arcs, filled regions, text with positions)
    │
    ▼
Stage 2: Property Classification
    Group by line weight / color / dash pattern
    → element type candidates (wall / annotation / dimension / hatch)
    │
    ▼
Stage 3: Wall Candidate Detection
    Apply HFV2013 geometric rules:
    parallel pairs, consistent thickness, orthogonal orientation
    → centerline segments with thickness
    │
    ▼
Stage 4: Topology Construction
    Junction detection (corner / T / X / endpoint)
    → NetworkX graph: nodes = junctions, edges = wall segments
    │
    ▼
Stage 5: Semantic Role Assignment
    Rule-based first pass using domain knowledge from wiki/domains/
    → functional role + confidence + rule_triggered
    │
    ▼
Structured JSON output
```

---

## 4. Output Schema

Every wall segment in the output JSON must carry all of the following fields. Do not omit any field — unknown values get `null`, not a missing key.

```json
{
  "metadata": {
    "source_pdf": "filename.pdf",
    "extraction_date": "YYYY-MM-DD",
    "pipeline_version": "0.1.0",
    "total_segments": 0,
    "coordinate_system": "pdf_points_bottom_left_origin"
  },
  "segments": [
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
        "connected_segment_ids": []
      },
      "semantic": {
        "functional_role": "interior_partition | bearing_wall | wet_wall | demising | exterior | unknown",
        "confidence": 0.0,
        "rule_triggered": "rule name or null",
        "requires_cross_document_validation": false
      }
    }
  ],
  "junctions": [
    {
      "junction_id": "uuid-v4",
      "position": [x, y],
      "junction_type": "corner | t-junction | x-junction | endpoint",
      "connected_segment_ids": []
    }
  ]
}
```

### Semantic role definitions

These map directly to the wiki domain pages. When in doubt, read the domain page.

| Role | Rule basis | Wiki page |
|------|-----------|-----------|
| `exterior` | Longest continuous walls forming the building perimeter | `wiki/domains/bearing-wall.md` |
| `bearing_wall` | Structurally continuous — flag for cross-document validation | `wiki/domains/bearing-wall.md` |
| `wet_wall` | Bounds bathroom/kitchen zone, or contains plumbing symbol adjacency | `wiki/domains/wet-wall.md` |
| `demising` | Full-length continuous wall separating units or tenancies | `wiki/domains/panelization-rules.md` |
| `interior_partition` | Default — does not match any above rule | — |
| `unknown` | Geometry detected but rules insufficient to classify | — |

`requires_cross_document_validation: true` must be set for any `bearing_wall` or `wet_wall` assignment — these cannot be confirmed from a single architectural drawing alone (this is H2's empirical footprint in the pipeline).

---

## 5. Evaluation Protocol

**The pipeline is not done when it runs. It is done when it has been validated.**

### Ground truth format

Ground truth files live in `evaluation/ground_truth/`. One JSON file per floor plan, same schema as pipeline output but with human-verified `functional_role` values and `confidence: 1.0`.

### Evaluation script

`evaluate.py` takes pipeline output + ground truth and reports:

- Per-class precision, recall, F1
- Overall accuracy
- Confusion matrix (saved as CSV)
- Failure mode summary: which rules fired incorrectly and why

Results saved to `evaluation/results/YYYY-MM-DD_<filename>.json`.

### What the numbers mean for H1

| Result | H1 implication |
|--------|---------------|
| High accuracy across all classes | H1 may be weaker than expected for geometry-amenable classes |
| High accuracy on exterior/partition, low on wet_wall/bearing_wall | H1 confirmed — functional role requires non-visual information |
| Systematic confusion (e.g., wet_wall → partition) | H1 confirmed — and defines the exact failure mode |
| Random errors across all classes | Pipeline bug, not an H1 result |

---

## 6. Project Structure

```
project-root/
├── CLAUDE.md                  ← this file
├── config.yaml                ← all tunable thresholds
├── pipeline.py                ← entry point: python pipeline.py <pdf_path>
├── stages/
│   ├── __init__.py
│   ├── extract_paths.py       ← Stage 1
│   ├── classify_properties.py ← Stage 2
│   ├── detect_walls.py        ← Stage 3
│   ├── build_topology.py      ← Stage 4
│   └── assign_semantics.py    ← Stage 5
├── evaluate.py
├── output/                    ← pipeline JSON outputs (gitignored if large)
├── evaluation/
│   ├── ground_truth/          ← manually labeled JSONs
│   └── results/               ← evaluation run outputs
└── tests/
    ├── test_extract.py
    ├── test_classify.py
    ├── test_walls.py
    ├── test_topology.py
    └── test_semantics.py
```

---

## 7. Conventions

- **Language**: Python 3.10+
- **Core dependencies**: PyMuPDF (`fitz`), Shapely, NetworkX, NumPy, PyYAML
- **No ML model dependencies in Phase 1.** Rule-based only. This is intentional — establish and measure the geometric baseline before introducing learned components.
- **All thresholds in `config.yaml`**, never hardcoded. Thresholds that vary per drawing set must be tunable without touching code.
- **Coordinates**: PDF points, origin at bottom-left.
- **One test per pipeline stage minimum.** Tests run with `pytest`.
- **Fail loudly.** If a stage produces unexpected geometry (negative thickness, zero-length segment, disconnected graph), raise an exception with a clear message rather than silently continuing.

---

## 8. Principles

- **Geometry first, semantics second.** Get wall detection geometrically correct before adding semantic rules. A wrong geometry with a right label is worse than a right geometry with an unknown label.
- **Rule transparency.** Every semantic assignment must name the rule that fired. Black-box outputs are not useful for research.
- **H1 visibility.** The `requires_cross_document_validation` flag is the pipeline's acknowledgment that it cannot answer certain questions from a single drawing. Preserve this signal — do not paper over it with a confident guess.
- **Domain knowledge is authoritative.** The wiki domain pages encode practitioner expertise. When a domain rule conflicts with a geometric heuristic, the domain rule wins.
- **Measure everything.** No phase is complete without a benchmark result on held-out floor plans.
