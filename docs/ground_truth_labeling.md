# Ground-truth labeling workflow

The pipeline reports aggregate entity counts across 36 plans. That tells us
coverage but not correctness: *of the 106 rooms we found, how many are real?
of the 8 we called `bathroom`, how many actually are?* The coordinator-task
evaluation answers those questions.

This doc walks through the labeling loop end-to-end.

## TL;DR

```bash
# 1. Generate a pre-filled YAML template for a plan you've already extracted
python scripts/generate_labeling_template.py output/benchmark/<plan>.json

# 2. Open the annotated PDF for visual reference (requires feat/v06-visualizer)
#    and open the YAML in your editor
python visualize.py output/benchmark/<plan>.json <plan>.pdf --out <plan>_annotated.pdf

# 3. Fill in the YAML — mostly editing `correct_type` and `valid_targets`
#    (see "What to edit" below)

# 4. Run the evaluation
python scripts/eval_coordinator_tasks.py \
    evaluation/labels/<plan>.labels.yaml \
    output/benchmark/<plan>.json
```

## Why this design

Most labeling tools require drawing polygons from scratch. That's expensive —
hours per plan. Instead we exploit the fact that the pipeline already extracts
rooms, walls, and cross-references with stable UUIDs. The user's job reduces
to **reviewing** the extraction and **editing** it in-place:

- *Is room `r-abc123` really a room? If yes, what type?*
- *Is `F10.` a real sheet reference, or did we misclassify a fixture tag?*
- *Did we miss any obvious rooms?*

Per-plan labeling time drops from hours to roughly 15–30 minutes.

## The labeling template

`evaluation/labels/<stem>.labels.yaml` has three sections:

### Section 1 — `rooms`

One entry per room the pipeline detected, pre-filled with context (centroid,
area, detected type, labels inside the polygon, and current room_name /
room_number if any).

**What to edit:** the `correct_type` field on each entry. Write one of:

- `unit`, `bathroom`, `kitchen`, `stair`, `hallway`, `mechanical`,
  `laundry`, `storage`, `office` — if the room is real and has that type
- `unknown` — if the room is real but you can't tell the type from the drawing
- `wrong_detection` — if this isn't actually a room (sliver, aggregate across
  multiple rooms, duplicate inner/outer loop, etc.)

### Section 2 — `missed_rooms`

Rooms you see on the drawing that the pipeline didn't detect. Leave empty if
coverage is complete on this plan; otherwise add `{centroid: [x, y],
correct_type: <type>}` entries. (Centroid doesn't need to be exact — the
evaluator uses it only for reporting context.)

### Section 3 — `cross_references`

- `detected_targets`: read-only list of every target the pipeline emitted.
- `valid_targets`: starts as a copy of `detected_targets`. **Remove items**
  that aren't real sheet references (e.g., fixture callouts like `F10.` or
  dimension strings that slipped through).
- `missed_targets`: **add** any sheet references you see on the drawing that
  aren't in `detected_targets`.

## Running the evaluation

```bash
python scripts/eval_coordinator_tasks.py \
    evaluation/labels/<stem>.labels.yaml \
    output/benchmark/<stem>.json
```

Output goes to `evaluation/results/coordinator_eval_<stem>_<date>.md` and
`.json`. The markdown report lists precision/recall for each task plus any
type-level mismatches for easy review.

The evaluator **refuses to run** if any `correct_type` field still says
`TODO` — this prevents silently reporting garbage metrics from a
half-finished template.

## Scaling this

For v1 we ship templates for three plans of different character:

| Template | Character |
|---|---|
| `2274_shattuck_2.labels.yaml` | Dense residential, 47 detected rooms |
| `2100_jefferson.labels.yaml` | Large plan with inner/outer-loop duplicate rooms |
| `capsule_crate.labels.yaml` | No detected rooms but many cross-references |

Once these three are labeled, you have per-plan precision/recall you can
trend against every pipeline change. Adding more plans is a matter of
`python scripts/generate_labeling_template.py <new plan>.json` → label → run
the eval.

## Deferred

Wall-to-schedule-tag assignment (task 3 from the original proposal) is not
yet in the template. It needs per-wall labeling machinery that would make
this template considerably more complex. We'll add it when tasks 1 and 2
are producing signal worth acting on.
