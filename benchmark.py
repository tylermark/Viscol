"""Cross-drawing benchmark harness.

Runs the pipeline across a directory of floor-plan PDFs, collects per-plan
stats, and emits a consolidated report. Where ground-truth CSV files exist
with segment-level ``is_wall`` labels, computes an aggregate detection rate.

This is the v0.4.0-small generalization-measurement tool. It does NOT match
individual segments across extraction tools (that's a larger undertaking).
It measures cross-drawing STABILITY of the pipeline's outputs.

Usage:
    python benchmark.py <plans_dir> [--ground-truth <csv_dir>] [--out <report.md>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date
from pathlib import Path

import pipeline
from stages._config import load_config


def _run_pipeline_subprocess(pdf_path: str, config_path: str, output_dir: str) -> None:
    """Top-level function so ProcessPoolExecutor can pickle it."""
    pipeline.run(
        pdf_path=Path(pdf_path),
        page_num=None,
        config_path=Path(config_path),
        output_dir=Path(output_dir),
    )


def _run_with_timeout(pdf: Path, config_path: Path, output_dir: Path, timeout_s: float) -> tuple[str, str | None]:
    """Run the pipeline on one PDF in a subprocess, killing it on timeout.

    Returns (status, error_message_or_None). A timeout returns ("timeout", message).
    """
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _run_pipeline_subprocess, str(pdf), str(config_path), str(output_dir)
        )
        try:
            future.result(timeout=timeout_s)
            return "ok", None
        except FuturesTimeoutError:
            # Best-effort cancellation; kill any worker processes still running.
            for proc in pool._processes.values():  # type: ignore[attr-defined]
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            return "timeout", f"exceeded {timeout_s:.0f}s cap"
        except Exception as exc:
            return "error", f"{type(exc).__name__}: {exc}"


def _find_pdfs(plans_dir: Path) -> list[Path]:
    return sorted(p for p in plans_dir.iterdir() if p.suffix.lower() == ".pdf")


def _match_ground_truth(plan_pdf: Path, gt_dir: Path) -> Path | None:
    """Find the latest CSV whose stem matches the plan's stem.

    Convention: `<stem>_segments_v4.csv` beats `_segments_v3.csv` beats `_segments_v2.csv`.
    """
    stem = plan_pdf.stem
    candidates = sorted(gt_dir.glob(f"{stem}_segments*.csv"))
    if not candidates:
        return None
    # Prefer higher "vN" suffix; fall back to lex order
    best = None
    best_rank = -1
    for c in candidates:
        name = c.stem
        suffix = name.rsplit("_", 1)[-1]
        rank = 0
        if suffix.startswith("v"):
            try:
                rank = int("".join(ch for ch in suffix[1:] if ch.isdigit()) or "0")
            except ValueError:
                rank = 0
        if rank > best_rank:
            best, best_rank = c, rank
    return best


def _ground_truth_stats(csv_path: Path) -> dict:
    """Return {gt_wall_segments, gt_non_wall_segments, estimated_gt_walls}."""
    wall_segs = 0
    non_wall_segs = 0
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = (row.get("is_wall") or "").strip()
            if v == "1":
                wall_segs += 1
            elif v == "0":
                non_wall_segs += 1
    # Walls are pairs of line segments, so divide by 2 for a rough pair-count estimate.
    estimated_walls = wall_segs // 2
    return {
        "gt_wall_segments": wall_segs,
        "gt_non_wall_segments": non_wall_segs,
        "estimated_gt_walls": estimated_walls,
    }


def _summarize_run(plan: Path, output_dir: Path, config: dict) -> dict:
    """Read the pipeline's output JSON for this plan and extract summary fields."""
    out = output_dir / f"{plan.stem}.json"
    dropped = output_dir / f"{plan.stem}_dropped.json"
    result: dict = {"plan": plan.name, "status": "unknown"}
    if not out.exists():
        result["status"] = "no_output"
        return result

    with out.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    role_counts = Counter(s["semantic"]["functional_role"] for s in doc["walls"])
    thicknesses = [s["geometry"]["thickness"] for s in doc["walls"]]
    lengths = [s["geometry"]["centerline_length"] for s in doc["walls"]]
    room_type_counts = Counter(r.get("room_type") or "unknown" for r in doc.get("rooms", []))
    text_class_counts = Counter(
        t.get("classification") or "unknown" for t in doc.get("text_regions", [])
    )

    result.update(
        {
            "status": "ok",
            "walls": len(doc["walls"]),
            "junctions": len(doc["junctions"]),
            "rooms": len(doc.get("rooms", [])),
            "openings": len(doc.get("openings", [])),
            "text_regions": len(doc.get("text_regions", [])),
            "grid_lines": len(doc.get("grid_lines", [])),
            "cross_references": len(doc.get("cross_references", [])),
            "role_counts": dict(role_counts),
            "room_type_counts": dict(room_type_counts),
            "text_class_counts": dict(text_class_counts),
            "thickness_median": _median(thicknesses) if thicknesses else None,
            "thickness_min": min(thicknesses) if thicknesses else None,
            "thickness_max": max(thicknesses) if thicknesses else None,
            "length_median": _median(lengths) if lengths else None,
        }
    )

    if dropped.exists():
        with dropped.open("r", encoding="utf-8") as f:
            dropped_doc = json.load(f)
        reason_counts = Counter(s.get("reason", "unknown") for s in dropped_doc.get("segments", []))
        result["dropped_total"] = len(dropped_doc.get("segments", []))
        result["dropped_by_reason"] = dict(reason_counts)
    return result


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float(0.5 * (s[n // 2 - 1] + s[n // 2]))


def run_benchmark(
    plans_dir: Path,
    gt_dir: Path | None,
    output_dir: Path,
    report_path: Path,
    config_path: Path,
    max_plans: int | None = None,
    per_plan_timeout_s: float = 300.0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _find_pdfs(plans_dir)
    if max_plans is not None:
        pdfs = pdfs[:max_plans]

    results: list[dict] = []
    print(f"Benchmarking {len(pdfs)} plans…")
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}", flush=True)
        t0 = time.time()
        plan_result: dict = {
            "plan": pdf.name,
            "stem": pdf.stem,
            "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2),
        }
        status, err = _run_with_timeout(pdf, config_path, output_dir, per_plan_timeout_s)
        if status == "ok":
            plan_result["pipeline_status"] = "ok"
        elif status == "timeout":
            plan_result["pipeline_status"] = f"timeout: {err}"
        else:
            plan_result["pipeline_status"] = f"error: {err}"
        plan_result["wall_time_seconds"] = round(time.time() - t0, 2)

        # Only summarize on a fresh successful run — otherwise stale output JSONs
        # from a previous benchmark would be silently counted as this run's result.
        if plan_result["pipeline_status"] == "ok":
            summary = _summarize_run(pdf, output_dir, load_config(config_path))
            plan_result.update({k: v for k, v in summary.items() if k != "plan"})
        else:
            plan_result["status"] = "error"

        if gt_dir is not None:
            gt_csv = _match_ground_truth(pdf, gt_dir)
            if gt_csv is not None:
                plan_result["ground_truth_csv"] = gt_csv.name
                plan_result.update(_ground_truth_stats(gt_csv))
                if plan_result.get("walls") is not None and plan_result.get("estimated_gt_walls"):
                    plan_result["detection_rate"] = round(
                        plan_result["walls"] / max(plan_result["estimated_gt_walls"], 1),
                        3,
                    )

        results.append(plan_result)

    report = {
        "benchmark_date": date.today().isoformat(),
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "plans_dir": str(plans_dir),
        "ground_truth_dir": str(gt_dir) if gt_dir else None,
        "n_plans": len(results),
        "n_ok": sum(1 for r in results if r.get("pipeline_status") == "ok"),
        "n_errors": sum(1 for r in results if r.get("pipeline_status", "").startswith("error")),
        "results": results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = report_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = _render_markdown(report)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport (markdown): {report_path}")
    print(f"Report (json):     {json_path}")
    return report


def _render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark report — v{report['pipeline_version']}")
    lines.append("")
    lines.append(f"- **Date:** {report['benchmark_date']}")
    lines.append(f"- **Plans:** {report['n_plans']} ({report['n_ok']} ok, {report['n_errors']} errors)")
    lines.append(f"- **Plans dir:** `{report['plans_dir']}`")
    if report["ground_truth_dir"]:
        lines.append(f"- **Ground truth dir:** `{report['ground_truth_dir']}`")
    lines.append("")

    ok = [r for r in report["results"] if r.get("pipeline_status") == "ok"]
    if ok:
        walls = [r.get("walls", 0) or 0 for r in ok]
        dropped = [r.get("dropped_total", 0) or 0 for r in ok]
        thick = [r.get("thickness_median") for r in ok if r.get("thickness_median")]
        lines.append("## Aggregate across ok plans")
        lines.append("")
        lines.append(f"- Walls detected: min={min(walls)}, median={_median(walls):.0f}, max={max(walls)}")
        lines.append(f"- Dropped (total): min={min(dropped)}, median={_median(dropped):.0f}, max={max(dropped)}")
        if thick:
            lines.append(
                f"- Median centerline thickness per plan: min={min(thick):.1f} pt, "
                f"median={_median(thick):.1f} pt, max={max(thick):.1f} pt"
            )
        with_gt = [r for r in ok if "detection_rate" in r]
        if with_gt:
            rates = [r["detection_rate"] for r in with_gt]
            lines.append(
                f"- Detection rate (walls / estimated_gt_walls) across {len(with_gt)} plans with GT: "
                f"min={min(rates):.2f}, median={_median(rates):.2f}, max={max(rates):.2f}"
            )
        lines.append("")

    # ----- v0.6 entity-coverage aggregate -----
    if ok:
        def _frac(entity: str) -> float:
            return sum(1 for r in ok if (r.get(entity) or 0) > 0) / len(ok)

        lines.append("## Entity coverage (v0.6) — fraction of plans with ≥1 entity of each type")
        lines.append("")
        lines.append(f"- Walls:            {_frac('walls'):.0%} ({len(ok)} plans)")
        lines.append(f"- Rooms:            {_frac('rooms'):.0%}")
        lines.append(f"- Openings:         {_frac('openings'):.0%}")
        lines.append(f"- Text regions:     {_frac('text_regions'):.0%}")
        lines.append(f"- Grid lines:       {_frac('grid_lines'):.0%}")
        lines.append(f"- Cross-references: {_frac('cross_references'):.0%}")
        lines.append("")
        # Aggregated room-type distribution
        rt_total: Counter = Counter()
        for r in ok:
            rt_total.update(r.get("room_type_counts") or {})
        if rt_total:
            lines.append(f"- Room types (across all plans): {dict(rt_total)}")
        lines.append("")

    lines.append("## Per-plan v0.6 entity counts")
    lines.append("")
    lines.append(
        "| Plan | Status | Walls | Rooms | Open | Text | Grid | XRef | GT walls* | Det. rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in report["results"]:
        status = r.get("pipeline_status", "?")
        status_cell = "✓" if status == "ok" else f"✗ {status[:40]}"
        walls = r.get("walls", "—")
        rooms = r.get("rooms", "—")
        openings = r.get("openings", "—")
        text_r = r.get("text_regions", "—")
        grid = r.get("grid_lines", "—")
        xref = r.get("cross_references", "—")
        gt_walls = r.get("estimated_gt_walls", "—")
        det_rate = r.get("detection_rate", "—")
        det_rate_cell = f"{det_rate:.2f}" if isinstance(det_rate, (int, float)) else "—"
        name = r.get("plan", "")
        if len(name) > 55:
            name = name[:52] + "…"
        lines.append(
            f"| {name} | {status_cell} | {walls} | {rooms} | {openings} | "
            f"{text_r} | {grid} | {xref} | {gt_walls} | {det_rate_cell} |"
        )

    # Secondary wall-detail table (thickness, dropped, role breakdown)
    lines.append("")
    lines.append("## Per-plan wall details")
    lines.append("")
    lines.append(
        "| Plan | Junct. | Ext. | Dropped (thick/iso) | Thk(med) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for r in report["results"]:
        if r.get("pipeline_status") != "ok":
            continue
        junct = r.get("junctions", "—")
        roles = r.get("role_counts", {})
        ext = roles.get("exterior", 0) if isinstance(roles, dict) else "—"
        drop_reasons = r.get("dropped_by_reason", {})
        drop_thick = drop_reasons.get("thickness_cluster_outlier", 0)
        drop_iso = drop_reasons.get("isolated_short_segment", 0)
        dropped_cell = f"{drop_thick}/{drop_iso}" if r.get("dropped_total") else "—"
        thk = r.get("thickness_median")
        thk_cell = f"{thk:.1f}" if thk else "—"
        name = r.get("plan", "")
        if len(name) > 55:
            name = name[:52] + "…"
        lines.append(
            f"| {name} | {junct} | {ext} | {dropped_cell} | {thk_cell} |"
        )

    lines.append("")
    lines.append("\\* *GT walls = (# CSV segments labeled `is_wall=1`) / 2, rough pair-count estimate.*")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-drawing benchmark for the wall extraction pipeline.")
    parser.add_argument("plans_dir", type=Path, help="Directory containing floor-plan PDFs.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional directory of CSV ground-truth files.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--report", type=Path, default=None, help="Output markdown report path.")
    parser.add_argument("--max-plans", type=int, default=None)
    parser.add_argument(
        "--per-plan-timeout",
        type=float,
        default=300.0,
        help="Per-plan wall-clock cap in seconds (CLAUDE.md §6 hard cap). Default 300.",
    )
    args = parser.parse_args(argv)

    if args.report is None:
        args.report = Path("evaluation") / "results" / f"benchmark_{date.today().isoformat()}.md"

    run_benchmark(
        plans_dir=args.plans_dir,
        gt_dir=args.ground_truth,
        output_dir=args.output_dir,
        report_path=args.report,
        config_path=args.config,
        max_plans=args.max_plans,
        per_plan_timeout_s=args.per_plan_timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
