"""Focused benchmark: run pipeline + evaluate on the plans that have a `*-correct.pdf` ground truth.

Faster than the full 36-plan benchmark; gives real precision/recall numbers.

Usage:
    python benchmark_gt.py <plans_dir> <ground_truth_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pipeline
import evaluate_correct


def _gt_match(plan_pdf: Path, gt_dir: Path) -> Path | None:
    candidate = gt_dir / f"{plan_pdf.stem}-correct.pdf"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plans_dir", type=Path)
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark_gt"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--perp-tol", type=float, default=12.0)
    parser.add_argument("--parallel-tol", type=float, default=10.0)
    parser.add_argument("--report", type=Path, default=Path("evaluation/results/gt_eval.md"))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plans = sorted(p for p in args.plans_dir.iterdir() if p.suffix.lower() == ".pdf")
    plans_with_gt = [(p, _gt_match(p, args.ground_truth_dir)) for p in plans]
    plans_with_gt = [(p, gt) for p, gt in plans_with_gt if gt is not None]
    print(f"Found {len(plans_with_gt)} plans with ground truth")

    results = []
    for i, (plan, gt) in enumerate(plans_with_gt, 1):
        print(f"\n[{i}/{len(plans_with_gt)}] {plan.name}", flush=True)
        t0 = time.time()
        try:
            pipeline.run(plan, page_num=None, config_path=args.config, output_dir=args.output_dir)
            out_json = args.output_dir / f"{plan.stem}.json"
            r = evaluate_correct.evaluate(
                out_json, plan, gt,
                perp_tol=args.perp_tol,
                parallel_tol_deg=args.parallel_tol,
            )
            r["wall_time_seconds"] = round(time.time() - t0, 2)
            print(
                f"   P={r['precision']:.2f}  R={r['recall_walls']:.2f}  F1={r['f1']:.2f}  "
                f"(soft P={r['soft_precision']:.2f}, R={r['soft_recall_segments']:.2f})  "
                f"({r['tp_walls']} TP / {r['n_pred_walls']} pred / {r['n_gt_walls_estimated']} gt)"
            )
            results.append(r)
        except Exception as exc:
            print(f"   ERROR: {type(exc).__name__}: {exc}")
            results.append({"plan": plan.name, "error": str(exc)})

    valid = [r for r in results if "error" not in r]
    if valid:
        avg_p = sum(r["precision"] for r in valid) / len(valid)
        avg_r = sum(r["recall_walls"] for r in valid) / len(valid)
        avg_f = sum(r["f1"] for r in valid) / len(valid)
        avg_sp = sum(r["soft_precision"] for r in valid) / len(valid)
        avg_sr = sum(r["soft_recall_segments"] for r in valid) / len(valid)
        avg_sf = sum(r["soft_f1"] for r in valid) / len(valid)
        print(f"\n========== AGGREGATE ({len(valid)} plans) ==========")
        print(f"Strict (pair-match): P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f:.3f}")
        print(f"Soft (centerline-near-GT): P={avg_sp:.3f}  R={avg_sr:.3f}  F1={avg_sf:.3f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    md = [
        f"# GT-evaluated benchmark — v{pipeline.PIPELINE_VERSION}",
        f"- Date: {date.today().isoformat()}",
        f"- Plans: {len(results)} ({len(valid)} ok)",
        f"- Tolerances: perp={args.perp_tol}pt, parallel={args.parallel_tol}°",
        "",
    ]
    if valid:
        md += [
            f"## Aggregate ({len(valid)} plans)",
            "",
            f"- **Strict P/R/F1**: {avg_p:.3f} / {avg_r:.3f} / **{avg_f:.3f}**",
            f"- **Soft P/R/F1**: {avg_sp:.3f} / {avg_sr:.3f} / **{avg_sf:.3f}**",
            "",
        ]
    md += [
        "## Per-plan",
        "",
        "| Plan | Pred | GT | TP | P | R | F1 | softP | softR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "error" in r:
            md.append(f"| {r['plan'][:50]} | — | — | — | ERROR | | | | |")
            continue
        md.append(
            f"| {r['plan'][:50]} | {r['n_pred_walls']} | {r['n_gt_walls_estimated']} | "
            f"{r['tp_walls']} | {r['precision']:.2f} | {r['recall_walls']:.2f} | "
            f"{r['f1']:.2f} | {r['soft_precision']:.2f} | {r['soft_recall_segments']:.2f} |"
        )
    args.report.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport: {args.report}")

    json_path = args.report.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"date": date.today().isoformat(), "tolerances": {"perp": args.perp_tol, "parallel": args.parallel_tol}, "results": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
