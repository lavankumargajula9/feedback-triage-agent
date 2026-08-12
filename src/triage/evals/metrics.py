"""Honest classification metrics over completed eval runs (R12, R27, KTD6, AE6).

Everything here is deterministic arithmetic over the results file the runner
writes for 100%-complete runs (the ONLY metrics input, R32) plus the gold
labels:

- Per step (categorize / route / escalate), per system (pipeline / baseline):
  accuracy, macro-averaged precision/recall, and a full per-class table with
  support (R12, KTD6). Macro averages run over the gold classes; a class the
  system never predicts gets precision 0.0 — the documented zero-division
  convention behind every hand-checkable number.
- A step whose output is a typed failure predicts the ``OUTPUT_FAILURE``
  sentinel: it matches no gold label, so it is scored INCORRECT and stays in
  every denominator (R27, AE6), and each system's output-failure rate is its
  own reported line.
- Confusion matrix for categorization; paired bootstrap confidence intervals
  (seeded, ~1000 resamples over per-thread prediction pairs) on the
  pipeline-vs-baseline accuracy delta; variance-flip candidates; per-thread
  token/cost/latency aggregation when checkpoints carry a ``usage`` block
  (KTD6).

Draft quality is deliberately NOT here: the judge score is a distinct number
reported alongside, never blended into, these metrics (R13).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from statistics import mean
from typing import Any

from triage.evals.runner import EvalError, GoldLabel

# The sentinel "label" a failed output predicts: off every taxonomy, so it can
# never match a gold label (R27) — failures are wrong, never excluded (AE6).
OUTPUT_FAILURE = "<output-failure>"

STEPS = ("categorize", "route", "escalate")
SYSTEMS = ("pipeline", "baseline")

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 1729

# Result-payload field carrying each step's predicted label.
_PIPELINE_FIELDS = {"categorize": "label", "route": "queue", "escalate": "escalate"}
_BASELINE_FIELDS = {"categorize": "category", "route": "queue", "escalate": "escalate"}


def _normalize(label: Any) -> str:
    """Labels as strings throughout (escalate booleans become 'true'/'false')."""
    if isinstance(label, bool):
        return "true" if label else "false"
    return label


def gold_by_step(gold: Mapping[int, GoldLabel]) -> dict[str, dict[int, str]]:
    """The gold labels reshaped to per-step {thread_id: label} maps."""
    return {
        "categorize": {tid: g.category for tid, g in gold.items()},
        "route": {tid: g.queue for tid, g in gold.items()},
        "escalate": {tid: _normalize(g.escalate) for tid, g in gold.items()},
    }


def _pipeline_prediction(entry: Mapping[str, Any], step: str) -> str:
    payload = entry["pipeline"]["steps"].get(step)
    if not payload or "failure" in payload:
        return OUTPUT_FAILURE
    return _normalize(payload[_PIPELINE_FIELDS[step]])


def _baseline_prediction(entry: Mapping[str, Any], step: str) -> str:
    payload = entry["baseline"]
    if not payload or "failure" in payload:
        # ONE failed baseline call means no usable output for ANY step.
        return OUTPUT_FAILURE
    return _normalize(payload[_BASELINE_FIELDS[step]])


def collect_predictions(results: Mapping[str, Any]) -> dict[str, dict[str, dict[int, str]]]:
    """Per-system, per-step {thread_id: predicted label} from a results file."""
    predictions: dict[str, dict[str, dict[int, str]]] = {
        system: {step: {} for step in STEPS} for system in SYSTEMS
    }
    for entry in results["threads"]:
        tid = entry["thread_id"]
        for step in STEPS:
            predictions["pipeline"][step][tid] = _pipeline_prediction(entry, step)
            predictions["baseline"][step][tid] = _baseline_prediction(entry, step)
    return predictions


def accuracy(gold: Mapping[int, str], pred: Mapping[int, str]) -> float:
    """Fraction correct over ALL gold threads; a missing prediction is wrong."""
    return sum(pred.get(tid, OUTPUT_FAILURE) == label for tid, label in gold.items()) / len(gold)


def per_class_table(
    gold: Mapping[int, str], pred: Mapping[int, str]
) -> dict[str, dict[str, float | int]]:
    """Precision/recall/support per gold class (KTD6).

    Classes are the labels with gold support; a class the system never
    predicts gets precision 0.0 (zero-division convention). The failure
    sentinel has no gold support so it never gets a row — it drags the recall
    of true classes and is surfaced by :func:`failure_stats` instead (R27).
    """
    table: dict[str, dict[str, float | int]] = {}
    for cls in sorted(set(gold.values())):
        true_positives = sum(
            1 for tid, label in gold.items() if label == cls and pred.get(tid) == cls
        )
        predicted = sum(1 for tid in gold if pred.get(tid) == cls)
        support = sum(1 for label in gold.values() if label == cls)
        table[cls] = {
            "precision": true_positives / predicted if predicted else 0.0,
            "recall": true_positives / support,
            "support": support,
        }
    return table


def confusion_matrix(
    gold: Mapping[int, str], pred: Mapping[int, str]
) -> dict[str, dict[str, int]]:
    """Gold-label -> predicted-label counts; failures appear as the sentinel."""
    matrix: dict[str, dict[str, int]] = {}
    for tid, label in sorted(gold.items()):
        row = matrix.setdefault(label, {})
        predicted = pred.get(tid, OUTPUT_FAILURE)
        row[predicted] = row.get(predicted, 0) + 1
    return matrix


def failure_stats(pred: Mapping[int, str]) -> dict[str, float | int]:
    """Output-failure count and rate over one prediction set (R27, AE6)."""
    total = len(pred)
    failures = sum(1 for label in pred.values() if label == OUTPUT_FAILURE)
    return {"failures": failures, "total": total, "rate": failures / total if total else 0.0}


def paired_bootstrap_delta(
    gold: Mapping[int, str],
    pred_a: Mapping[int, str],
    pred_b: Mapping[int, str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile CI on the paired accuracy delta (a - b) over cached
    predictions (KTD6): threads are resampled with replacement, keeping each
    thread's two predictions paired; the seeded RNG makes reruns identical."""
    thread_ids = sorted(gold)
    a_correct = [1 if pred_a.get(tid, OUTPUT_FAILURE) == gold[tid] else 0 for tid in thread_ids]
    b_correct = [1 if pred_b.get(tid, OUTPUT_FAILURE) == gold[tid] else 0 for tid in thread_ids]
    n = len(thread_ids)
    observed = (sum(a_correct) - sum(b_correct)) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a_correct[i] - b_correct[i] for i in indices) / n)
    deltas.sort()
    low = deltas[int((alpha / 2) * resamples)]
    high = deltas[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {
        "delta": observed,
        "ci_low": low,
        "ci_high": high,
        "resamples": resamples,
        "seed": seed,
    }


def variance_flip_candidates(
    gold: Mapping[int, str], pipeline_pred: Mapping[int, str], baseline_pred: Mapping[int, str]
) -> dict[str, list[int]]:
    """Threads one arm gets right and the other wrong (KTD6 error-analysis feed)."""
    pipeline_only, baseline_only = [], []
    for tid in sorted(gold):
        pipeline_ok = pipeline_pred.get(tid) == gold[tid]
        baseline_ok = baseline_pred.get(tid) == gold[tid]
        if pipeline_ok and not baseline_ok:
            pipeline_only.append(tid)
        elif baseline_ok and not pipeline_ok:
            baseline_only.append(tid)
    return {"pipeline_only_correct": pipeline_only, "baseline_only_correct": baseline_only}


def aggregate_usage(entries: list[Mapping[str, Any]], system: str) -> dict[str, Any] | None:
    """Sum/mean the per-thread ``usage`` numbers for one arm, when present.

    Checkpoints may carry ``entry["usage"][system]`` dicts of numeric fields
    (tokens/cost/latency, KTD6); returns None when no thread has any.
    """
    rows = [
        entry["usage"][system]
        for entry in entries
        if isinstance(entry.get("usage"), Mapping) and isinstance(entry["usage"].get(system), Mapping)
    ]
    if not rows:
        return None
    fields = sorted(
        {key for row in rows for key, value in row.items() if isinstance(value, (int, float))}
    )
    totals = {field: sum(row.get(field, 0) for row in rows) for field in fields}
    return {
        "threads": len(rows),
        "totals": totals,
        "means": {field: totals[field] / len(rows) for field in fields},
    }


def compute_report(results: Mapping[str, Any], gold: Mapping[int, GoldLabel]) -> dict[str, Any]:
    """The full classification-metrics report for one completed run (R12, KTD6).

    ``results`` is a results file the runner wrote (100%-complete only, R32);
    ``gold`` is the validated gold-labels map. The thread sets must match.
    """
    result_ids = {entry["thread_id"] for entry in results["threads"]}
    if result_ids != set(gold):
        raise EvalError(
            f"results file and gold labels describe different thread sets "
            f"(results-only: {sorted(result_ids - set(gold))[:5]}, "
            f"gold-only: {sorted(set(gold) - result_ids)[:5]}); metrics are computed "
            "only over the gold-labeled eval set"
        )
    golds = gold_by_step(gold)
    predictions = collect_predictions(results)
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        steps: dict[str, Any] = {}
        for step in STEPS:
            table = per_class_table(golds[step], predictions[system][step])
            steps[step] = {
                "accuracy": accuracy(golds[step], predictions[system][step]),
                "macro_precision": mean(row["precision"] for row in table.values()),
                "macro_recall": mean(row["recall"] for row in table.values()),
                "per_class": table,
                "output_failures": failure_stats(predictions[system][step]),
            }
        # R27: the system's output-failure rate is its own line, over every
        # step-level prediction the system made.
        all_predictions = Counter()
        for step in STEPS:
            stats = steps[step]["output_failures"]
            all_predictions["failures"] += stats["failures"]
            all_predictions["total"] += stats["total"]
        systems[system] = {
            "steps": steps,
            "output_failures": {
                "failures": all_predictions["failures"],
                "total": all_predictions["total"],
                "rate": (
                    all_predictions["failures"] / all_predictions["total"]
                    if all_predictions["total"]
                    else 0.0
                ),
            },
        }
    return {
        "n_threads": len(gold),
        "systems": systems,
        "confusion": {
            system: confusion_matrix(golds["categorize"], predictions[system]["categorize"])
            for system in SYSTEMS
        },
        "deltas": {
            step: paired_bootstrap_delta(
                golds[step], predictions["pipeline"][step], predictions["baseline"][step]
            )
            for step in STEPS
        },
        "flips": {
            step: variance_flip_candidates(
                golds[step], predictions["pipeline"][step], predictions["baseline"][step]
            )
            for step in STEPS
        },
        "usage": {system: aggregate_usage(results["threads"], system) for system in SYSTEMS},
    }
