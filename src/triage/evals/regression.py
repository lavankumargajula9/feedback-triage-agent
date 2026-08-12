"""Regression gate over a recorded reference run (R15, KTD10, AE4).

The gated set is exactly R15's: per-step accuracy, per-step macro-averaged
precision/recall (pipeline arm — the system under guard), and mean draft
score. Per-class numbers are reported in the metrics report but NEVER gated —
a per-class dip alone cannot fail the check (AE4).

Tolerances (R15):
- percentage-scale metrics: max(2x the observed run-to-run spread, 2
  percentage points);
- mean draft score: 2x its own observed spread from the judge-scored variance
  passes on the 1-5 scale — no fixed floor.

Spreads are supplied in the reference artifact, which pins model IDs, prompt
hashes, the eval-set hash, and per-thread outputs (KTD10). When the current
run's pins differ from the reference the check emits a HARD environment-drift
warning, distinct from a metric failure. A reference is recorded ONLY via the
explicit record-reference command — never automatically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triage.evals.runner import write_json_atomic

DRAFT_SCORE_METRIC = "mean_draft_score"
GATED_STEPS = ("categorize", "route", "escalate")
GATED_STEP_METRICS = ("accuracy", "macro_precision", "macro_recall")
# R15: percentage-scale metrics never get a band tighter than 2 points.
PERCENTAGE_FLOOR = 0.02

# The determinism pins compared for environment drift (KTD10).
ENVIRONMENT_KEYS = ("models", "prompt_hashes", "eval_set_hash", "thread_digest")

REFERENCE_REQUIRED_KEYS = (
    "models",
    "prompt_hashes",
    "eval_set_hash",
    "thread_digest",
    "metrics",
    "spreads",
    "per_thread",
)


class RegressionError(Exception):
    """Raised for unusable reference artifacts."""


def gated_metrics(
    report: Mapping[str, Any], *, mean_draft_score: float | None = None
) -> dict[str, float]:
    """Extract R15's gated set from a metrics report (pipeline arm).

    Per-class numbers deliberately never appear here (AE4). ``mean_draft_score``
    joins the gate only when a judge-scored value is supplied.
    """
    steps = report["systems"]["pipeline"]["steps"]
    metrics = {
        f"{step}_{metric}": steps[step][metric]
        for step in GATED_STEPS
        for metric in GATED_STEP_METRICS
    }
    if mean_draft_score is not None:
        metrics[DRAFT_SCORE_METRIC] = mean_draft_score
    return metrics


def tolerance_for(name: str, spread: float) -> float:
    """The R15 noise tolerance for one gated metric.

    Percentage-scale metrics: max(2x observed spread, 2pp). Mean draft score:
    2x its own observed spread — no fixed floor.
    """
    if name == DRAFT_SCORE_METRIC:
        return 2.0 * spread
    return max(2.0 * spread, PERCENTAGE_FLOOR)


def check_regression(
    current_metrics: Mapping[str, float],
    reference: Mapping[str, Any],
    *,
    current_env: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the current run's gated metrics against the reference (R15).

    Returns ``failures`` (each naming its metric), ``drift`` (hard
    environment-drift warnings, distinct from metric failures), the per-metric
    ``checked`` detail, and ``ok`` — true only with no failures AND no drift.
    """
    drift = [
        (
            f"environment drift: {key} differs from the reference "
            f"(reference {reference.get(key)!r}, current {current_env.get(key)!r}); "
            "the comparison is not like-for-like — this is NOT a metric failure"
        )
        for key in ENVIRONMENT_KEYS
        if reference.get(key) != current_env.get(key)
    ]
    spreads = reference.get("spreads", {})
    failures: list[dict[str, Any]] = []
    checked: dict[str, Any] = {}
    for name, ref_value in reference["metrics"].items():
        tolerance = tolerance_for(name, float(spreads.get(name, 0.0)))
        current = current_metrics.get(name)
        if current is None:
            failures.append(
                {
                    "metric": name,
                    "message": (
                        f"regression: gated metric {name!r} is missing from the current "
                        f"run (reference {ref_value:.4f})"
                    ),
                }
            )
            continue
        drop = ref_value - current
        checked[name] = {
            "reference": ref_value,
            "current": current,
            "drop": drop,
            "tolerance": tolerance,
        }
        if drop > tolerance:
            failures.append(
                {
                    "metric": name,
                    "message": (
                        f"regression: {name} dropped to {current:.4f} from reference "
                        f"{ref_value:.4f} (drop {drop:.4f} > tolerance {tolerance:.4f})"
                    ),
                }
            )
    return {
        "ok": not failures and not drift,
        "failures": failures,
        "drift": drift,
        "checked": checked,
    }


def environment_of(results: Mapping[str, Any]) -> dict[str, Any]:
    """The current run's determinism pins, from its results file (KTD10)."""
    return {key: results[key] for key in ENVIRONMENT_KEYS}


def record_reference(
    results: Mapping[str, Any],
    metrics: Mapping[str, float],
    *,
    spreads: Mapping[str, float] | None = None,
    path: Path | str,
) -> Path:
    """Write the reference artifact — EXPLICIT-only (KTD10).

    Only the ``triage eval --record-reference`` command (or an equally explicit
    caller) reaches this; nothing records a reference automatically. Pins
    model IDs, prompt hashes, the eval-set hash, and per-thread outputs, plus
    the gated metrics and their observed variance-pass spreads.

    A reference without ``mean_draft_score`` or without measured spreads is
    refused rather than written: both defects make the gate look armed while
    it silently is not (R15, R21).
    """
    if DRAFT_SCORE_METRIC not in metrics:
        raise RegressionError(
            f"refusing to record a reference without the {DRAFT_SCORE_METRIC!r} metric: "
            "the draft-quality gate would silently not exist. Re-run with judging "
            "enabled (`triage eval --judge`, R13)."
        )
    if not spreads:
        raise RegressionError(
            "refusing to record a reference with no measured spreads: every "
            "percentage tolerance would collapse to the 2-point floor and the "
            f"{DRAFT_SCORE_METRIC!r} tolerance to zero (R15). Supply the spreads "
            "measured by the variance passes (R21) with `--spreads` or "
            "`--variance-results`."
        )
    reference = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_id": results.get("run_id", ""),
        "profile": results.get("profile", ""),
        "models": results["models"],
        "prompt_hashes": results["prompt_hashes"],
        "eval_set_hash": results["eval_set_hash"],
        "thread_digest": results["thread_digest"],
        "metrics": dict(metrics),
        "spreads": dict(spreads or {}),
        "per_thread": results["threads"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, reference)
    return path


def load_reference(path: Path | str) -> dict[str, Any]:
    """Load and validate a reference artifact; every defect is a RegressionError."""
    path = Path(path)
    if not path.is_file():
        raise RegressionError(
            f"no reference artifact at {path}: record one explicitly with "
            f"`triage eval --record-reference {path}` (KTD10; never recorded automatically)"
        )
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegressionError(f"reference artifact {path} is not valid JSON: {exc}") from None
    missing = [key for key in REFERENCE_REQUIRED_KEYS if key not in reference]
    if missing:
        raise RegressionError(
            f"reference artifact {path} is missing key(s): {', '.join(missing)}; "
            "re-record it with `triage eval --record-reference`"
        )
    if DRAFT_SCORE_METRIC not in reference["metrics"]:
        raise RegressionError(
            f"reference artifact {path} has no {DRAFT_SCORE_METRIC!r} metric, so it "
            "cannot gate draft quality (R15); re-record it with judging enabled "
            "(`triage eval --judge --record-reference`)"
        )
    return reference
