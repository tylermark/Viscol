# Phase 1 Summary — Construction Intelligence Pipeline v0.6.0

**Date:** 2026-04-17
**Corpus:** 36 real-world architectural PDFs (`C:\Users\tyler\autospec-research\data\plans`)
**Pipeline version:** v0.6.0 (H0-driven multi-entity extraction infrastructure)

Per CLAUDE.md §11, Phase 1 concludes when a short write-up exists describing:
what the pipeline reliably extracts, what fails, which entity types have the
strongest signal, and what architecture decisions this implies for Phase 2
(FM training).

---

## Headline numbers

**Batch processing:** 36 plans → 31 successful (86%). 5 timed out at the 300s
hard cap (per CLAUDE.md §6); **0 errors**. Pipeline is stable on real drawings.

**Per-plan throughput:** min=2.1s, median=35.1s, p90=154.8s, max=253.6s.
Median is well under the 60s target; tail plans approach but mostly stay under
the 300s cap.

**Total entities extracted across 31 ok plans:**

| Entity | Total | Median/plan | Coverage (plans w/ ≥1) |
|---|---:|---:|---:|
| Walls | 13,635 | ~300 | **97%** |
| Junctions | 13,138 | — | — |
| Text regions | 11,947 | ~380 | **100%** |
| Cross-references | 1,436 | ~46 | **94%** |
| Openings | 275 | 0 | 26% |
| Grid lines | 168 | 2 | 77% |
| Rooms | 106 | 0 | 35% |

## What reliably extracts — the strong signals

### 1. Text regions (100% coverage, 11,947 instances)

Every plan produces text region records. The Stage 7 classifier assigns one
of nine `classification` values (`room_label`, `room_number`, `sheet_callout`,
`dimension`, `wall_schedule_tag`, `note`, `title`, `grid_label`, `unknown`) to
each region. Every record carries:

- `bbox`: `[x0, y0, x1, y1]` bounding box in PDF points (bottom-left origin)
- `rule_triggered`: string naming the matching heuristic (e.g.
  `room_number_pattern`, `sheet_callout_pattern`, `multiword_note_heuristic`)
- `requires_cross_document_validation`: boolean, set to `true` for any region
  whose classification depends on another sheet (sheet_callout or any region
  with non-empty `references`)

**This is the richest signal Phase 2 has access to.**

### 2. Walls (97% coverage, 13,635 instances)

Centerlines + thickness + functional role (interior_partition, bearing_wall,
wet_wall, demising, exterior, unknown). Role distribution is plan-dependent
but role assignment fires on every wall.

One plan (`23PR03_EHC Athens Extract[9]`) produced 0 walls — a Stage 3 edge
case worth investigating separately.

### 3. Cross-references (94% coverage, 1,436 records across 31 plans)

This is the unexpected finding of Phase 1. The cross-reference extractor
produced **245 unique target identifiers** across the corpus, most of which
are legitimate cross-document references:

- **Architectural sheet IDs:** `A302`, `A401`, `S1`, `L7` — standard sheet callouts
- **Wall schedule tags:** `W1`, `W2`, …, `W7` — 93 refs to `W2`, 65 to `W1`; walls reference wall-type schedule tables
- **Unit type tags:** `U1`–`U5.1` — spaces reference unit-type-definition tables
- **Door type tags:** `D01`, `D03`, `D04` — doors reference door-schedule tables

**Plausible-sheet precision:** 96.2% (1,382/1,436 records point to well-formed
identifiers matching architectural naming conventions).

**Cross-plan commonality:** Tags like `W1`, `W2`, `A302`, `A401` appear across
multiple plans — these are the exact candidates for Phase 2 multi-document
inference training (H2 territory).

See `evaluation/results/cross_reference_graph_2026-04-17.md` for the full
coordinator-task evaluation report.

### 4. Grid lines (77% coverage, 168 instances)

Where drafted with standard letter/number bubble callouts, grid detection
works. Median ~2 grid lines per plan — consistent with small plans that only
have a few primary grid lines drafted.

## What fails — the weak signals

### Room detection

**Coverage:** 35% of plans produce ≥1 room.
**Typing:** **Of 106 detected rooms, only 8 (7.5%)** received a non-`unknown`
room_type (3 office, 2 mechanical, 2 laundry, 1 unit).

**Why it's weak:** `shapely.ops.polygonize` requires closed loops, but real
floor plans have door openings, arches, and drafting gaps everywhere. Closure
is rare. When it does close, it closes on the wrong features:
- Wall-thickness voids (narrow cavities between parallel wall centerlines)
- Aggregate regions spanning multiple actual rooms

Additionally, even when real-size polygons are produced, the labels inside
them are often fixture callouts (`F10.`, `BOX 2B-1a`) rather than room-type
words.

**v0.6 mitigations that did help:**
- Sliver filter (aspect > 5 or short-side < 15pt): cuts wall-thickness voids
- Virtual gap closure: infrastructure ready but rarely fires on real plans
  because endpoint pairs aren't at door gaps

**Architectural implication for Phase 2:** Rule-based polygonization of a
noisy wall graph is the wrong tool for room extraction. A foundation model
trained on the extracted walls + text regions + fixtures can likely learn
room boundaries directly, using the same visual cues coordinators use (walls
that bound a space + labels inside that name it). This is H1 (semantic layer)
territory — not a Phase 1 deliverable.

### Opening detection

**Coverage:** 26% of plans produce ≥1 opening.

Bimodal distribution — 23 plans get 0 openings, some get 30+. The arc-based
heuristic assumes every door is drawn as a quarter-circle swing arc, which
varies too much by drafting convention.

**Architectural implication:** Door detection is the right place for a small
trained classifier in Phase 2. Rule-based arc detection can't generalize
across drafting conventions.

## Concrete defects found during analysis (worth fixing in v0.6.1)

1. **`text_sheet_callout_pattern` over-matches.** The regex
   `^[A-Z][0-9]+[.]?[0-9]*[a-z]?$` allows a trailing `.` with nothing after it,
   letting fixture tags like `F10.` through as sheet callouts. Tightening to
   `^[A-Z][0-9]+(\.[0-9]+)?[a-z]?$` would reject them.
2. **Single plan with 0 walls.** `23PR03_EHC Athens Extract[9]` produced 0
   walls and 0 junctions. Stage 3 edge case.
3. **5 plans exceed 300s timeout.** Complex residential plans with high primitive
   counts. Performance optimization at scale.
4. **Duplicate rooms from inner/outer loops.** On Jefferson, `area=261,047` and
   `area=260,791` are the same room's outer-boundary and inner-boundary loops
   both emitted as separate rooms. Deduplicate by polygon overlap.

## What this means for Phase 2 architecture

1. **Text regions + cross-references are the deterministic input layer for
   Phase 2.** Small classifier refinements can boost precision cheaply.
2. **Walls are reliably extracted.** Functional role assignment is the H1
   semantic layer an FM will need to learn beyond the geometric rules.
3. **Rooms are NOT reliably extracted by rule-based methods.** Phase 2 should
   treat room derivation as a learned task over walls + openings + text, not
   as input ground truth from Phase 1.
4. **Openings need ML.** Rule-based arc detection doesn't generalize.
5. **The cross-reference graph is a rich latent H2 signal.** 1,436 records
   across 31 plans, with many targets shared across plans — exactly the raw
   material Phase 2 multi-document inference can train on.

## Follow-ups

See individual items in §"Concrete defects" above. Biggest remaining research
question: why 28% of walls on complex plans have loose endpoints (Stage 3/4
hygiene investigation, time-boxed).
