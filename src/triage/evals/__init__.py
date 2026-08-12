"""Eval harness (U7): cache-first LLM cache + resumable runner (R8, R30, R32, KTD5).

- :mod:`triage.evals.cache` — SQLite call cache keyed by hash of
  (model, params, full prompt, schema name, run-id salt), integrated at the
  LLM-wrapper seam so both eval arms share it (KTD5).
- :mod:`triage.evals.runner` — gold-labels loader, the single-prompt baseline
  arm assembled from the shared fragments (R8), and the checkpointed,
  resumable per-thread runner whose results files record model ids, prompt
  hashes, and the eval-set hash (R32, KTD5).
"""

from __future__ import annotations

from triage.evals.cache import CachingClient, CallCache, cache_key
from triage.evals.runner import (
    BaselineResult,
    EvalError,
    GoldLabel,
    baseline_system,
    baseline_user_text,
    load_gold_labels,
    run_baseline,
    run_eval,
    write_results,
)

__all__ = [
    "BaselineResult",
    "CachingClient",
    "CallCache",
    "EvalError",
    "GoldLabel",
    "baseline_system",
    "baseline_user_text",
    "cache_key",
    "load_gold_labels",
    "run_baseline",
    "run_eval",
    "write_results",
]
