"""Coordinator-task evaluation: sheet-to-sheet cross-reference graph.

Per CLAUDE.md §6 (coordinator-task benchmarks), a senior coordinator reading
a drawing mentally tracks what OTHER drawings are called out: "A302 for
wall section", "1/A7.1 for stair detail", etc. This script measures
whether Phase 1's extracted cross_references capture that navigation
signal.

For each pipeline output JSON:
  - Enumerate cross_references, grouping by target_sheet
  - Distinguish real sheet_callout classifications from the noisy "everything
    matches the callout pattern" cases (e.g., "F10." fixture tags mis-
    classified as sheet_callouts)

Across plans:
  - Total unique sheets referenced
  - Per-plan reference count distribution (coverage metric)
  - Plausible-sheet precision proxy: fraction of target_sheet strings that
    look like real architectural sheet IDs (A-prefix + digits, E-prefix, etc.)

Output: evaluation/results/cross_reference_graph_<date>.md and .json.

Usage:
    python scripts/eval_cross_reference_graph.py <output_dir> [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


# "Real" architectural sheet IDs look like: A302, A7.1, E-101, M2.5a, S1.01
_PLAUSIBLE_SHEET_RE = re.compile(r"^[A-Z][A-Z\-]?\d+(\.\d+[a-z]?)?$")
# Obvious false positives we see a lot: fixture callouts ending in a period
# (e.g. "F10.", "F09."), column grid references (A-B-C single letters), etc.
_FIXTURE_CALLOUT_RE = re.compile(r"^[FPM]\d+\.$")  # F10., P12., M3.


def _classify_target(target: str) -> str:
    if _FIXTURE_CALLOUT_RE.match(target):
        return "fixture_callout_misclassified"
    if _PLAUSIBLE_SHEET_RE.match(target):
        return "plausible_sheet_id"
    if re.match(r"^[A-Z]\d$", target):
        # "A2", "B6" — could be grid callouts, probably not sheets
        return "grid_or_short_tag"
    return "other"


def _summarize_plan(doc: dict) -> dict:
    xrefs = doc.get("cross_references") or []
    target_counts = Counter(x.get("target_sheet") for x in xrefs)
    category_counts: Counter = Counter()
    for t in target_counts:
        category_counts[_classify_target(str(t))] += 1

    # Text-region reverse lookup for each xref
    tid_to_text = {t["text_region_id"]: t for t in (doc.get("text_regions") or [])}
    enriched: list[dict] = []
    for x in xrefs:
        src = tid_to_text.get(x.get("source_text_region_id") or "", {})
        enriched.append({
            "target_sheet": x.get("target_sheet"),
            "target_detail": x.get("target_detail"),
            "source_text": src.get("text"),
            "source_classification": src.get("classification"),
            "category": _classify_target(str(x.get("target_sheet") or "")),
        })

    return {
        "source_pdf": doc.get("metadata", {}).get("source_pdf"),
        "total_xrefs": len(xrefs),
        "unique_targets": len(target_counts),
        "top_targets": target_counts.most_common(10),
        "category_breakdown": dict(category_counts),
        "enriched_xrefs": enriched,
    }


def evaluate(output_dir: Path) -> dict:
    plan_summaries: list[dict] = []
    global_target_counts: Counter = Counter()
    plans_per_target: dict[str, set] = defaultdict(set)
    category_totals: Counter = Counter()

    for p in sorted(output_dir.glob("*.json")):
        if "_dropped" in p.name or ".invalid" in p.name:
            continue
        with p.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        summary = _summarize_plan(doc)
        plan_summaries.append(summary)
        source = summary.get("source_pdf") or p.stem
        for x in summary["enriched_xrefs"]:
            t = x["target_sheet"]
            if t:
                global_target_counts[t] += 1
                plans_per_target[t].add(source)
            category_totals[x["category"]] += 1

    total_xrefs = sum(s["total_xrefs"] for s in plan_summaries)
    unique_targets = len(global_target_counts)
    plausible = category_totals.get("plausible_sheet_id", 0)
    plausible_precision = plausible / total_xrefs if total_xrefs else 0.0

    # Cross-plan reference commonality
    shared_targets = {t: sorted(s) for t, s in plans_per_target.items() if len(s) > 1}

    return {
        "n_plans": len(plan_summaries),
        "total_xrefs": total_xrefs,
        "unique_targets_across_plans": unique_targets,
        "plausible_sheet_precision": round(plausible_precision, 3),
        "category_totals_by_unique_target": dict(category_totals),
        "top_global_targets": global_target_counts.most_common(20),
        "shared_targets": {t: v for t, v in sorted(shared_targets.items(), key=lambda kv: -len(kv[1]))[:20]},
        "per_plan": plan_summaries,
    }


def _render_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append("# Coordinator-task evaluation — cross-reference graph")
    lines.append("")
    lines.append(f"- Plans analyzed: {result['n_plans']}")
    lines.append(f"- Total cross_reference records: {result['total_xrefs']:,}")
    lines.append(f"- Unique target sheets across corpus: {result['unique_targets_across_plans']:,}")
    lines.append(f"- Plausible-sheet precision proxy: **{result['plausible_sheet_precision']:.1%}**")
    lines.append("")
    lines.append("## Target categorization (by xref instance)")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for cat, count in sorted(result["category_totals_by_unique_target"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cat} | {count:,} |")
    lines.append("")
    lines.append("## Top 20 globally-referenced target sheets")
    lines.append("")
    lines.append("| Target | Refs |")
    lines.append("|---|---:|")
    for t, n in result["top_global_targets"]:
        lines.append(f"| `{t}` | {n} |")
    lines.append("")
    lines.append("## Targets referenced by multiple plans (cross-doc candidates)")
    lines.append("")
    if not result["shared_targets"]:
        lines.append("_None — every target is unique to a single plan._")
    else:
        lines.append("| Target | Plans |")
        lines.append("|---|---|")
        for t, plans in result["shared_targets"].items():
            lines.append(f"| `{t}` | {', '.join(pl[:40] for pl in plans[:3])}{'…' if len(plans) > 3 else ''} |")
    lines.append("")
    lines.append("## Per-plan summary")
    lines.append("")
    lines.append("| Plan | Total xrefs | Unique targets | Plausible | Fixture-misclass | Other |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in result["per_plan"]:
        pdf = s["source_pdf"] or "?"
        if len(pdf) > 55:
            pdf = pdf[:52] + "…"
        cb = s["category_breakdown"]
        lines.append(
            f"| {pdf} | {s['total_xrefs']} | {s['unique_targets']} | "
            f"{cb.get('plausible_sheet_id', 0)} | {cb.get('fixture_callout_misclassified', 0)} | "
            f"{cb.get('other', 0) + cb.get('grid_or_short_tag', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation") / "results" / f"cross_reference_graph_{date.today().isoformat()}.md",
    )
    args = parser.parse_args(argv)

    result = evaluate(args.output_dir)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.report.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md = _render_markdown(result)
    with args.report.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"Report (markdown): {args.report}")
    print(f"Report (json):     {json_path}")
    print()
    print(f"Plans: {result['n_plans']}")
    print(f"Total xrefs: {result['total_xrefs']:,}")
    print(f"Unique targets: {result['unique_targets_across_plans']:,}")
    print(f"Plausible-sheet precision: {result['plausible_sheet_precision']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
