"""Evaluation harness.

Usage:
    python evaluate.py <pipeline_output.json> <ground_truth.json>

Compares a pipeline output against a human-labeled ground truth file of the
same schema and reports precision/recall/F1, a confusion matrix, and a
failure summary. Saves full results to `evaluation/results/YYYY-MM-DD_<stem>.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from stages._config import load_config


CLASSES = [
    "exterior",
    "demising",
    "interior_partition",
    "wet_wall",
    "bearing_wall",
    "unknown",
]


def _midpoint(seg: dict) -> tuple[float, float]:
    s = seg["geometry"]["start"]
    e = seg["geometry"]["end"]
    return (0.5 * (s[0] + e[0]), 0.5 * (s[1] + e[1]))


def _match_segments(
    pred_segments: list[dict],
    truth_segments: list[dict],
    proximity: float,
) -> list[tuple[dict | None, dict | None]]:
    """Return (pred, truth) pairs. Unmatched on either side are paired with None."""
    pred_by_id = {s["segment_id"]: s for s in pred_segments}
    truth_by_id = {s["segment_id"]: s for s in truth_segments}
    shared_ids = set(pred_by_id) & set(truth_by_id)

    pairs: list[tuple[dict | None, dict | None]] = []
    if len(shared_ids) >= max(1, min(len(pred_segments), len(truth_segments)) // 2):
        for sid in shared_ids:
            pairs.append((pred_by_id[sid], truth_by_id[sid]))
        for sid in set(pred_by_id) - shared_ids:
            pairs.append((pred_by_id[sid], None))
        for sid in set(truth_by_id) - shared_ids:
            pairs.append((None, truth_by_id[sid]))
        return pairs

    used_truth: set[int] = set()
    for p in pred_segments:
        pm = _midpoint(p)
        best_j = -1
        best_d = proximity
        for j, t in enumerate(truth_segments):
            if j in used_truth:
                continue
            tm = _midpoint(t)
            d = math.hypot(pm[0] - tm[0], pm[1] - tm[1])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used_truth.add(best_j)
            pairs.append((p, truth_segments[best_j]))
        else:
            pairs.append((p, None))
    for j, t in enumerate(truth_segments):
        if j not in used_truth:
            pairs.append((None, t))
    return pairs


def _confusion_matrix(pairs: list[tuple[dict | None, dict | None]]) -> np.ndarray:
    """Rows = truth, cols = predicted. Extra 'unmatched' row/col at index len(CLASSES)."""
    size = len(CLASSES) + 1
    matrix = np.zeros((size, size), dtype=int)
    for pred, truth in pairs:
        t_idx = (
            CLASSES.index(truth["semantic"]["functional_role"])
            if truth is not None
            else len(CLASSES)
        )
        p_idx = (
            CLASSES.index(pred["semantic"]["functional_role"])
            if pred is not None
            else len(CLASSES)
        )
        matrix[t_idx, p_idx] += 1
    return matrix


def _per_class_metrics(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    n = len(CLASSES)
    for i, cls in enumerate(CLASSES):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum()) - tp
        fn = int(matrix[i, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    diag = int(np.trace(matrix[:n, :n]))
    total = int(matrix[:n, :n].sum())
    metrics["__overall__"] = {
        "accuracy": diag / total if total > 0 else 0.0,
        "matched_total": total,
        "unmatched_predictions": int(matrix[n, :n].sum()),
        "unmatched_truth": int(matrix[:n, n].sum()),
    }
    return metrics


def _print_confusion_matrix(matrix: np.ndarray) -> None:
    labels = CLASSES + ["UNMATCHED"]
    col_w = max(max(len(l) for l in labels), 6)
    print("\nConfusion matrix (rows=truth, cols=predicted):")
    header = " " * (col_w + 3) + " ".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = " ".join(f"{int(matrix[i, j]):>{col_w}}" for j in range(len(labels)))
        print(f"{row_label:<{col_w + 2}} {row}")


def _save_confusion_csv(matrix: np.ndarray, path: Path) -> None:
    labels = CLASSES + ["UNMATCHED"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, row_label in enumerate(labels):
            writer.writerow([row_label] + [int(matrix[i, j]) for j in range(len(labels))])


def _failure_summary(pairs: list[tuple[dict | None, dict | None]]) -> dict:
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pred, truth in pairs:
        if pred is None or truth is None:
            continue
        tr = truth["semantic"]["functional_role"]
        pr = pred["semantic"]["functional_role"]
        if tr == pr:
            continue
        by_pair[(tr, pr)].append(
            {
                "segment_id": pred.get("segment_id"),
                "predicted_rule": pred["semantic"].get("rule_triggered"),
                "truth_rule": truth["semantic"].get("rule_triggered"),
            }
        )
    summary = {}
    for (tr, pr), items in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        summary[f"{tr} -> {pr}"] = {
            "count": len(items),
            "examples": items[:5],
        }
    return summary


def _print_failure_summary(summary: dict) -> None:
    print("\nFailure summary (truth -> predicted):")
    if not summary:
        print("  (no misclassifications among matched segments)")
        return
    for key, info in summary.items():
        print(f"  {key}: {info['count']} case(s)")
        for ex in info["examples"]:
            print(
                f"    - segment_id={ex['segment_id']} "
                f"pred_rule={ex['predicted_rule']} truth_rule={ex['truth_rule']}"
            )


def run(
    pipeline_output: Path,
    ground_truth: Path,
    config_path: str | Path = "config.yaml",
    results_dir: Path = Path("evaluation/results"),
) -> Path:
    config = load_config(config_path)
    proximity = float(config.get("evaluation_match_proximity", config["junction_snap_distance"]))

    with pipeline_output.open("r", encoding="utf-8") as f:
        pred_doc = json.load(f)
    with ground_truth.open("r", encoding="utf-8") as f:
        truth_doc = json.load(f)

    pairs = _match_segments(pred_doc["walls"], truth_doc["walls"], proximity)
    matrix = _confusion_matrix(pairs)
    metrics = _per_class_metrics(matrix)
    summary = _failure_summary(pairs)

    _print_confusion_matrix(matrix)

    print("\nPer-class metrics:")
    print(f"  {'class':<20} {'precision':>10} {'recall':>10} {'f1':>10}  tp  fp  fn")
    for cls in CLASSES:
        m = metrics[cls]
        print(
            f"  {cls:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}  "
            f"{m['tp']:>3} {m['fp']:>3} {m['fn']:>3}"
        )
    overall = metrics["__overall__"]
    print(f"\nOverall accuracy: {overall['accuracy']:.3f}  "
          f"(matched={overall['matched_total']}, "
          f"unmatched_pred={overall['unmatched_predictions']}, "
          f"unmatched_truth={overall['unmatched_truth']})")

    _print_failure_summary(summary)

    results_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    stem = pipeline_output.stem
    csv_path = results_dir / f"{today}_{stem}_confusion.csv"
    json_path = results_dir / f"{today}_{stem}.json"
    _save_confusion_csv(matrix, csv_path)

    result_doc = {
        "pipeline_output": str(pipeline_output),
        "ground_truth": str(ground_truth),
        "date": today,
        "pipeline_version": pred_doc.get("metadata", {}).get("pipeline_version"),
        "metrics": metrics,
        "confusion_matrix": {
            "labels": CLASSES + ["UNMATCHED"],
            "rows_truth_cols_predicted": matrix.tolist(),
        },
        "failures": summary,
        "role_counts_predicted": dict(
            Counter(s["semantic"]["functional_role"] for s in pred_doc["walls"])
        ),
        "role_counts_truth": dict(
            Counter(s["semantic"]["functional_role"] for s in truth_doc["walls"])
        ),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)

    print(f"\nCSV written: {csv_path}")
    print(f"Results written: {json_path}")
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate pipeline output against ground truth.")
    parser.add_argument("pipeline_output", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    run(args.pipeline_output, args.ground_truth, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
