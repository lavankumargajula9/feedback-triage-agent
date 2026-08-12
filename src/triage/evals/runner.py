"""Eval runner: checkpointed, resumable execution of both arms (R8, R30, R32, KTD5).

Three responsibilities:

- **Gold-labels loader** — validates the eval-set CSV (schema, label-set
  membership against the single-sourced taxonomy, no duplicate thread ids)
  before any call is planned; U6's data integrity is enforced here.
- **Baseline arm (R8)** — ONE LLM call producing all four outputs, its prompt
  assembled from the SAME shared fragments the pipeline steps use, so the
  before/after comparison is equal-information.
- **Runner (R32, KTD5)** — executes the pipeline arm and the baseline arm per
  thread through the shared call cache, checkpointing each per-thread result
  to disk as it completes. Resume skips completed threads; the results file —
  the only input metrics may use — is written exclusively for 100%-complete
  runs and records model ids, prompt hashes, and the eval-set hash.

Execution is serial (backoff on transient errors lives in the cache wrapper);
typed OutputFailures are recorded in checkpoints, never raised (R27).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from triage.evals.cache import CachingClient, CallCache
from triage.ingest.reconstruct import Thread
from triage.pipeline import result_dict, run_pipeline
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import build_user_text, degenerate_reason
from triage.tools.schemas import NEVER_SENT, CategoryLabel, OutputFailure, QueueLabel


class EvalError(Exception):
    """Raised for invalid eval inputs or incomplete runs (R32)."""


# ---------------------------------------------------------------------------
# Gold-labels loader (U6 data integrity, enforced here).
# ---------------------------------------------------------------------------

GOLD_COLUMNS = ("thread_id", "category", "queue", "escalate")
_TRUE_VALUES = frozenset({"true", "1", "yes"})
_FALSE_VALUES = frozenset({"false", "0", "no"})


@dataclass(frozen=True)
class GoldLabel:
    """One gold-labeled eval thread: category, queue, and escalation truth."""

    thread_id: int
    category: str
    queue: str
    escalate: bool


def load_gold_labels(path: Path | str) -> dict[int, GoldLabel]:
    """Load and validate the gold-labels CSV; every defect is an EvalError.

    Validates the column schema, label-set membership against the
    single-sourced taxonomy (KTD7), the escalate boolean, and duplicate
    thread ids — bad eval data never reaches a paid call.
    """
    path = Path(path)
    if not path.is_file():
        raise EvalError(
            f"no gold-labels file at {path}: the eval set is produced by the U6 "
            f"labeling pass as a CSV with columns: {', '.join(GOLD_COLUMNS)}"
        )
    labels: dict[int, GoldLabel] = {}
    # utf-8-sig: labeling tools on Windows (Excel, PowerShell) write a BOM.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(GOLD_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise EvalError(
                f"gold-labels file {path} is missing column(s): {', '.join(sorted(missing))}; "
                f"expected columns: {', '.join(GOLD_COLUMNS)}"
            )
        for line_no, row in enumerate(reader, start=2):
            labels_entry = _parse_gold_row(row, path=path, line_no=line_no)
            if labels_entry.thread_id in labels:
                raise EvalError(
                    f"duplicate thread id {labels_entry.thread_id} in {path} (line {line_no})"
                )
            labels[labels_entry.thread_id] = labels_entry
    if not labels:
        raise EvalError(f"gold-labels file {path} has a header but no rows")
    return labels


def _parse_gold_row(row: dict[str, Any], *, path: Path, line_no: int) -> GoldLabel:
    try:
        thread_id = int(row["thread_id"])
    except (TypeError, ValueError):
        raise EvalError(
            f"bad thread_id {row.get('thread_id')!r} in {path} (line {line_no})"
        ) from None
    category = (row.get("category") or "").strip()
    if category not in fragments.CATEGORY_LABELS:
        raise EvalError(
            f"off-taxonomy category {category!r} for thread {thread_id} in {path} "
            f"(line {line_no}); allowed: {', '.join(fragments.CATEGORY_LABELS)}"
        )
    queue = (row.get("queue") or "").strip()
    if queue not in fragments.QUEUE_LABELS:
        raise EvalError(
            f"off-taxonomy queue {queue!r} for thread {thread_id} in {path} "
            f"(line {line_no}); allowed: {', '.join(fragments.QUEUE_LABELS)}"
        )
    escalate_raw = (row.get("escalate") or "").strip().lower()
    if escalate_raw not in _TRUE_VALUES | _FALSE_VALUES:
        raise EvalError(
            f"bad escalate value {row.get('escalate')!r} for thread {thread_id} in {path} "
            f"(line {line_no}); expected true or false"
        )
    return GoldLabel(
        thread_id=thread_id,
        category=category,
        queue=queue,
        escalate=escalate_raw in _TRUE_VALUES,
    )


# ---------------------------------------------------------------------------
# Single-prompt baseline arm (R8).
# ---------------------------------------------------------------------------

BASELINE_STEP_NAME = "baseline"

BASELINE_TASK_HEADER = (
    "Perform ALL FOUR triage tasks below on this one thread and return a single "
    "JSON object containing every output."
)


class BaselineResult(BaseModel):
    """All four triage outputs from ONE call (R8); labels restricted to taxonomy."""

    category: CategoryLabel = Field(description="Exactly one category label from the label set.")
    queue: QueueLabel = Field(description="Exactly one support-queue label from the label set.")
    escalate: bool = Field(description="True when this thread needs human attention.")
    escalate_reason: str = Field(description="The stated reason for the escalation decision.")
    draft: str = Field(description="The reply draft, in the brand's public support voice.")
    status: Literal["never_sent"] = Field(
        default=NEVER_SENT,
        description="Always 'never_sent': drafts exist only for human review.",
    )


def baseline_system() -> str:
    """The baseline system prompt, assembled from the SAME shared fragments
    the pipeline steps use (R8) — same preamble, same taxonomy text, same
    four step instructions, in step order."""
    return (
        f"{fragments.SYSTEM_PREAMBLE}\n\n"
        f"{fragments.taxonomy_block()}\n\n"
        f"{BASELINE_TASK_HEADER}\n\n"
        f"{fragments.CATEGORIZE_INSTRUCTIONS}\n\n"
        f"{fragments.ROUTE_INSTRUCTIONS}\n\n"
        f"{fragments.ESCALATE_INSTRUCTIONS}\n\n"
        f"{fragments.DRAFT_INSTRUCTIONS}"
    )


def baseline_user_text(thread: Thread) -> str:
    """Identical to the thread text the pipeline's first step sees (R8)."""
    return build_user_text(thread)


def run_baseline(
    thread: Thread, *, model: str, client: Any = None
) -> BaselineResult | OutputFailure:
    """The single-prompt baseline: same model and settings as the pipeline (R8).

    Degenerate threads short-circuit to the same General Inquiry + escalated
    result the pipeline steps produce, before any LLM call (R28).
    """
    reason = degenerate_reason(thread)
    if reason is not None:
        return BaselineResult(
            category=fragments.GENERAL_INQUIRY,
            queue=fragments.GENERAL_INQUIRY,
            escalate=True,
            escalate_reason=fragments.insufficient_content_reason(reason),
            draft=fragments.DEGENERATE_DRAFT,
        )
    return call_with_schema(
        BASELINE_STEP_NAME,
        model=model,
        system=baseline_system(),
        user_text=baseline_user_text(thread),
        schema=BaselineResult,
        client=client,
    )


# ---------------------------------------------------------------------------
# Call/cost planning (R32 preview).
# ---------------------------------------------------------------------------

PIPELINE_CALLS_PER_THREAD = 4
BASELINE_CALLS_PER_THREAD = 1
CALLS_PER_THREAD = PIPELINE_CALLS_PER_THREAD + BASELINE_CALLS_PER_THREAD

# Rough per-call token budget from the plan; cost preview only, never billing.
TOKENS_IN_PER_CALL = 1500
TOKENS_OUT_PER_CALL = 300
# (input $/MTok, output $/MTok) per model id used by draft roles.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def cost_per_call(model: str) -> float | None:
    """Rough dollars per call, or None when the model has no price on file."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (TOKENS_IN_PER_CALL * in_price + TOKENS_OUT_PER_CALL * out_price) / 1_000_000


def format_preview(
    remaining: int,
    total: int,
    *,
    pipeline_model: str,
    baseline_model: str,
    profile: str,
    run_id: str = "",
) -> str:
    """The planned-call-count and rough-cost preview printed BEFORE executing (R32)."""
    lines = [
        (
            f"Eval plan (R32): {remaining} of {total} threads to run "
            f"({total - remaining} already checkpointed)"
        ),
        (
            f"Planned calls: {remaining} threads x {CALLS_PER_THREAD} calls/thread "
            f"({PIPELINE_CALLS_PER_THREAD} pipeline + {BASELINE_CALLS_PER_THREAD} baseline) "
            f"= {remaining * CALLS_PER_THREAD} calls"
        ),
        (
            f"Models (profile {profile!r}, R30): pipeline={pipeline_model}, "
            f"baseline={baseline_model}"
        ),
        _cost_line(
            remaining * PIPELINE_CALLS_PER_THREAD,
            pipeline_model,
            remaining * BASELINE_CALLS_PER_THREAD,
            baseline_model,
        ),
    ]
    if run_id:
        lines.append(
            f"Run-id salt {run_id!r}: prior cache entries are bypassed for this run (KTD5)."
        )
    return "\n".join(lines)


def _cost_line(
    pipeline_calls: int, pipeline_model: str, baseline_calls: int, baseline_model: str
) -> str:
    pipeline_cost = cost_per_call(pipeline_model)
    baseline_cost = cost_per_call(baseline_model)
    if pipeline_cost is None or baseline_cost is None:
        unknown = pipeline_model if pipeline_cost is None else baseline_model
        return f"Rough cost: unknown (no pricing on file for {unknown!r})"
    total = pipeline_calls * pipeline_cost + baseline_calls * baseline_cost
    return (
        f"Rough cost: {pipeline_calls} pipeline calls x ~${pipeline_cost:.4f} + "
        f"{baseline_calls} baseline calls x ~${baseline_cost:.4f} "
        f"(~{TOKENS_IN_PER_CALL} in / ~{TOKENS_OUT_PER_CALL} out tokens per call) "
        f"~= ${total:.2f}; cache hits cost $0"
    )


# ---------------------------------------------------------------------------
# Checkpointed, resumable execution (R32).
# ---------------------------------------------------------------------------

RESULTS_FILENAME = "results.json"


def checkpoint_dir(out_dir: Path | str) -> Path:
    return Path(out_dir) / "threads"


def checkpoint_path(out_dir: Path | str, thread_id: int) -> Path:
    return checkpoint_dir(out_dir) / f"{thread_id}.json"


def completed_thread_ids(out_dir: Path | str) -> set[int]:
    """Thread ids with a finished checkpoint on disk (resume skips these)."""
    directory = checkpoint_dir(out_dir)
    if not directory.is_dir():
        return set()
    return {int(p.stem) for p in directory.glob("*.json") if p.stem.isdigit()}


def resume_instruction(out_dir: Path | str) -> str:
    """How to resume a partial run (printed on every nonzero partial exit, R32)."""
    return (
        "re-run the same `triage eval` command to resume; completed threads are "
        f"skipped via their checkpoints in {checkpoint_dir(out_dir)}"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write-then-rename so an interrupted write never leaves a torn checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _baseline_payload(result: BaselineResult | OutputFailure) -> dict[str, Any]:
    """JSON-ready baseline entry; typed failures are recorded, never raised (R27)."""
    if isinstance(result, OutputFailure):
        return {"failure": result.model_dump()}
    return result.model_dump()


def run_eval(
    threads: dict[int, Thread],
    *,
    out_dir: Path | str,
    cache_path: Path | str,
    pipeline_model: str,
    baseline_model: str,
    run_id: str = "",
    client: Any = None,
) -> dict[str, int]:
    """Execute both arms per thread with per-thread checkpointing (R32, KTD5).

    Threads already checkpointed are skipped; each completed thread is written
    to disk before the next begins, so an interruption at any point leaves a
    resumable run. Both arms call through the shared SQLite cache (KTD5), so a
    re-run of an interrupted thread replays its completed calls for free. Typed
    OutputFailures are recorded in checkpoints, never raised (R27); only
    transport-level exceptions propagate (the CLI turns them into a resume
    instruction).
    """
    out_dir = Path(out_dir)
    checkpoint_dir(out_dir).mkdir(parents=True, exist_ok=True)
    done = completed_thread_ids(out_dir)
    cache = CallCache(cache_path)
    try:
        caching_client = CachingClient(cache, inner=client, run_id=run_id)
        for thread_id in sorted(threads):
            if thread_id in done:
                continue
            thread = threads[thread_id]
            pipeline_result = result_dict(
                run_pipeline(thread, model=pipeline_model, client=caching_client)
            )
            baseline = _baseline_payload(
                run_baseline(thread, model=baseline_model, client=caching_client)
            )
            entry = {
                "thread_id": thread_id,
                "run_id": run_id,
                "pipeline": pipeline_result,
                "baseline": baseline,
                "ok": pipeline_result["ok"] and "failure" not in baseline,
            }
            _write_json_atomic(checkpoint_path(out_dir, thread_id), entry)
            done.add(thread_id)
    finally:
        cache.close()
    return {"completed": len(done & set(threads)), "total": len(threads)}


# ---------------------------------------------------------------------------
# Results file — only ever written for 100%-complete runs (R32, KTD5).
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_hashes() -> dict[str, str]:
    """Frozen-prompt hashes recorded in every results file (KTD5).

    Sampling parameters are not accepted on the chosen models, so determinism
    control is pinned model ids + these hashes + measured variance.
    """
    return {
        "categorize": _sha256_text(fragments.step_system(fragments.CATEGORIZE_INSTRUCTIONS)),
        "route": _sha256_text(fragments.step_system(fragments.ROUTE_INSTRUCTIONS)),
        "escalate": _sha256_text(fragments.step_system(fragments.ESCALATE_INSTRUCTIONS)),
        "draft": _sha256_text(fragments.step_system(fragments.DRAFT_INSTRUCTIONS)),
        "baseline": _sha256_text(baseline_system()),
    }


def eval_set_hash(labels_path: Path | str) -> str:
    """SHA-256 of the gold-labels file, recorded in every results file (KTD5)."""
    return hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()


def write_results(
    out_dir: Path | str,
    thread_ids: list[int],
    *,
    labels_path: Path | str,
    profile: str,
    pipeline_model: str,
    baseline_model: str,
    run_id: str = "",
) -> Path:
    """Assemble the run's results file from checkpoints — 100%-complete only (R32).

    A partial run raises EvalError carrying the resume instruction; metrics and
    regression comparisons are computed only from files this function wrote.
    """
    out_dir = Path(out_dir)
    thread_ids = sorted(thread_ids)
    missing = [tid for tid in thread_ids if not checkpoint_path(out_dir, tid).is_file()]
    if missing:
        done = len(thread_ids) - len(missing)
        shown = ", ".join(str(tid) for tid in missing[:5])
        suffix = ", ..." if len(missing) > 5 else ""
        raise EvalError(
            f"eval run incomplete: {done}/{len(thread_ids)} threads checkpointed "
            f"(missing thread ids: {shown}{suffix}); metrics are computed only from "
            f"100%-complete runs (R32) — {resume_instruction(out_dir)}"
        )
    entries = [
        json.loads(checkpoint_path(out_dir, tid).read_text(encoding="utf-8"))
        for tid in thread_ids
    ]
    results = {
        "run_id": run_id,
        "profile": profile,
        "models": {"pipeline": pipeline_model, "baseline": baseline_model},
        "prompt_hashes": prompt_hashes(),
        "eval_set_hash": eval_set_hash(labels_path),
        "determinism": (
            "pinned model ids + frozen prompt hashes above + run-id salt for "
            "variance passes (KTD5); the chosen models accept no sampling parameters"
        ),
        "calls_per_thread": CALLS_PER_THREAD,
        "complete": True,
        "threads": entries,
    }
    path = out_dir / RESULTS_FILENAME
    _write_json_atomic(path, results)
    return path
