"""Batch evaluator: run evaluate_correct on every plan that has a `*-correct.pdf` ground truth.

Usage:
    python evaluate_all_correct.py <output_dir> <plans_dir> <ground_truth_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import evaluate_correct


def _gt_match(plan_pdf: Path, gt_dir: Path) -> Path | None:
    candidate = gt_dir / f"{plan_pdf.stem}-correct.pdf"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="Where pipeline output JSONs live (e.g. output/benchmark)")
    parser.add_argument("plans_dir", type=Path)
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument("--perp-tol", type=float, default=6.0)
    parser.add_argument("--parallel-tol", type=float, default=5.0)
    parser.add_argument("--report", type=Path, default=Path("evaluation/results/correct_eval.md"))
    args = parser.parse_args(argv)

    pdfs = sorted(p for p in args.plans_dir.iterdir() if p.suffix.lower() == ".pdf")
    rows: list[dict] = []
    for pdf in pdfs:
        gt = _gt_match(pdf, args.ground_truth_dir)
        if gt is None:
            continue
        out_json = args.output_dir / f"{pdf.stem}.json"
        if not out_json.exists():
            continue
        try:
            r = evaluate_correct.evaluate(
                out_json, pdf, gt,
                perp_tol=args.perp_tol,
                parallel_tol_deg=args.parallel_tol,
            )
            rows.append(r)
            print(
                f"{pdf.stem[:55]:<55}  P={r['precision']:.2f}  R={r['recall_walls']:.2f}  "
                f"F1={r['f1']:.2f}  ({r['tp_walls']} TP / {r['n_pred_walls']} pred / "
                f"{r['n_gt_walls_estimated']} gt)"
            )
        except Exception as exc:
            print(f"{pdf.stem[:55]:<55}  ERROR: {exc}")
            rows.append({"plan": pdf.name, "error": str(exc)})

    if not rows:
        print("No plans found with matching ground truth.")
        return 1

    valid = [r for r in rows if "error" not in r]
    if not valid:
        print(f"\nERROR: all {len(rows)} matched plans failed to evaluate.")
        return 1

    avg_p = sum(r["precision"] for r in valid) / len(valid)
    avg_r = sum(r["recall_walls"] for r in valid) / len(valid)
    avg_f = sum(r["f1"] for r in valid) / len(valid)
    print()
    print(f"AGGREGATE across {len(valid)} plans:")
    print(f"  Mean precision: {avg_p:.3f}")
    print(f"  Mean recall:    {avg_r:.3f}")
    print(f"  Mean F1:        {avg_f:.3f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Per-plan evaluation against `*-correct.pdf` ground truth", ""]
    md.append(f"- Tolerances: perp={args.perp_tol}pt, parallel={args.parallel_tol}°")
    md.append(f"- Plans evaluated: {len(rows)}")
    md.append(f"- **Mean precision: {avg_p:.3f}**")
    md.append(f"- **Mean recall: {avg_r:.3f}**")
    md.append(f"- **Mean F1: {avg_f:.3f}**")
    md.append("")
    md.append("| Plan | Pred | GT | TP | FP | FN | P | R | F1 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if "error" in r:
            md.append(f"| {r['plan'][:55]} | — | — | — | — | — | ERROR | | |")
            continue
        name = r["plan"][:55]
        md.append(
            f"| {name} | {r['n_pred_walls']} | {r['n_gt_walls_estimated']} | "
            f"{r['tp_walls']} | {r['fp_walls']} | {r['fn_walls_estimated']} | "
            f"{r['precision']:.2f} | {r['recall_walls']:.2f} | {r['f1']:.2f} |"
        )
    args.report.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport: {args.report}")

    json_path = args.report.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"tolerances": {"perp_tol": args.perp_tol, "parallel_tol_deg": args.parallel_tol}, "results": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
