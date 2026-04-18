"""Regenerate the benchmark markdown/JSON report from existing output JSONs
without re-running the pipeline. Useful after the benchmark was interrupted.

Usage:
    python scripts/regen_report.py <output_dir> <plans_dir> [--ground-truth DIR]
                                   [--report PATH] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark  # noqa: E402
import pipeline  # noqa: E402
from stages._config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate benchmark report from existing output JSONs.")
    parser.add_argument("output_dir", type=Path, help="Directory with per-plan *.json outputs.")
    parser.add_argument("plans_dir", type=Path, help="Original plans directory (for filename matching).")
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation") / "results" / f"benchmark_{date.today().isoformat()}.md",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pdfs = sorted(p for p in args.plans_dir.iterdir() if p.suffix.lower() == ".pdf")
    results: list[dict] = []
    for pdf in pdfs:
        out = args.output_dir / f"{pdf.stem}.json"
        plan_result: dict = {
            "plan": pdf.name,
            "stem": pdf.stem,
            "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2),
        }
        if not out.exists():
            plan_result["pipeline_status"] = "not_processed"
            results.append(plan_result)
            continue
        plan_result["pipeline_status"] = "ok"
        summary = benchmark._summarize_run(pdf, args.output_dir, config)
        plan_result.update({k: v for k, v in summary.items() if k != "plan"})

        if args.ground_truth is not None:
            gt_csv = benchmark._match_ground_truth(pdf, args.ground_truth)
            if gt_csv is not None:
                plan_result["ground_truth_csv"] = gt_csv.name
                plan_result.update(benchmark._ground_truth_stats(gt_csv))
                if plan_result.get("walls") is not None and plan_result.get("estimated_gt_walls"):
                    plan_result["detection_rate"] = round(
                        plan_result["walls"] / max(plan_result["estimated_gt_walls"], 1), 3
                    )
        results.append(plan_result)

    from collections import Counter
    status_counts: Counter = Counter()
    for r in results:
        raw = r.get("pipeline_status", "")
        bucket = raw.split(":", 1)[0] if raw else "unknown"
        status_counts[bucket] += 1

    report = {
        "benchmark_date": date.today().isoformat(),
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "plans_dir": str(args.plans_dir),
        "ground_truth_dir": str(args.ground_truth) if args.ground_truth else None,
        "n_plans": len(results),
        "n_ok": status_counts.get("ok", 0),
        "n_errors": status_counts.get("error", 0),
        "n_not_processed": status_counts.get("not_processed", 0),
        "status_counts": dict(status_counts),
        "results": results,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.report.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = benchmark._render_markdown(report)
    with args.report.open("w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report (markdown): {args.report}")
    print(f"Report (json):     {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
