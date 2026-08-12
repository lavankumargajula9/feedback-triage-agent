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
  runs and records model ids, prompt hashes, the eval-set hash, and a digest of
  the rendered thread text. Every checkpoint is pinned to the models, prompts,
  and thread text that produced it, and carries what each arm spent on it, so
  neither a model swap nor an ingest change can be mixed into one results file.

Execution is serial (backoff on transient errors lives in the cache wrapper);
typed OutputFailures are recorded in checkpoints, never raised (R27).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from triage.evals.cache import CachingClient, CallCache, call_cost, usage_delta
from triage.ingest.reconstruct import Thread
from triage.pipeline import result_dict, run_pipeline
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import build_user_text, degenerate_reason, render_thread
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
        f"{fragments.category_block()}\n\n"
        f"{fragments.queue_block()}\n\n"
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
            queue=fragments.DEGENERATE_QUEUE,
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


def cost_per_call(model: str) -> float | None:
    """Rough dollars per call, or None when the model has no price on file."""
    return call_cost(model, TOKENS_IN_PER_CALL, TOKENS_OUT_PER_CALL)


def format_preview(
    remaining: int,
    total: int,
    *,
    pipeline_model: str,
    baseline_model: str,
    profile: str,
    run_id: str = "",
    judge: Mapping[str, Any] | None = None,
) -> str:
    """The planned-call-count and rough-cost preview printed BEFORE executing (R32).

    ``judge`` is :func:`triage.evals.judge.judge_plan` when judging is enabled:
    judge calls cost several times a pipeline step, and the operator decides to
    spend on this number, so they are counted and priced here or not at all.
    """
    arms_cost = _arms_cost(
        remaining * PIPELINE_CALLS_PER_THREAD,
        pipeline_model,
        remaining * BASELINE_CALLS_PER_THREAD,
        baseline_model,
    )
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
    if judge is not None:
        lines.extend(_judge_preview_lines(remaining, judge, arms_cost))
    if run_id:
        lines.append(
            f"Run-id salt {run_id!r}: prior cache entries are bypassed for this run (KTD5)."
        )
    return "\n".join(lines)


def _arms_cost(
    pipeline_calls: int, pipeline_model: str, baseline_calls: int, baseline_model: str
) -> float | None:
    """Rough dollars for both arms, or None when either model has no price."""
    pipeline_cost = cost_per_call(pipeline_model)
    baseline_cost = cost_per_call(baseline_model)
    if pipeline_cost is None or baseline_cost is None:
        return None
    return pipeline_calls * pipeline_cost + baseline_calls * baseline_cost


def _judge_preview_lines(
    remaining: int, judge: Mapping[str, Any], arms_cost: float | None
) -> list[str]:
    """The judge's share of the preview, priced separately then totaled (R13)."""
    calls = remaining * judge["calls_per_thread"]
    cost_per_judge_call = judge["cost_per_call"]
    lines = [
        (
            f"Judge calls (R13): {remaining} threads x {judge['calls_per_thread']} "
            f"calls/thread (pipeline + baseline drafts) = {calls} calls on "
            f"{judge['model']}"
        )
    ]
    if cost_per_judge_call is None:
        lines.append(f"Rough judge cost: unknown (no pricing on file for {judge['model']!r})")
        return lines
    judge_cost = calls * cost_per_judge_call
    lines.append(
        f"Rough judge cost: {calls} calls x ~${cost_per_judge_call:.4f} "
        f"(~{judge['tokens_in']} in / ~{judge['tokens_out']} out tokens per call) "
        f"~= ${judge_cost:.2f}; cache hits cost $0"
    )
    if arms_cost is not None:
        lines.append(f"Rough total (both arms + judge): ~${arms_cost + judge_cost:.2f}")
    return lines


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


def thread_fingerprint(thread: Thread) -> str:
    """SHA-256 of the thread text actually fed to the model (R32).

    ``render_thread`` is the single rendering every surface uses, so this
    fingerprint moves whenever ingest changes what the model would see.
    """
    return sha256_text(render_thread(thread))


def threads_digest(fingerprints: Mapping[int, str]) -> str:
    """One digest over every thread's fingerprint, recorded in the results file."""
    return sha256_text(
        "\n".join(f"{tid}:{fingerprints[tid]}" for tid in sorted(fingerprints))
    )


def run_identity(
    *, pipeline_model: str, baseline_model: str, run_id: str = ""
) -> dict[str, Any]:
    """What a checkpoint must have been produced by to belong to this run (KTD5).

    Stamped into every checkpoint and re-checked on resume. A results file is
    only honest if all of its checkpoints came from the same models, prompts,
    and run-id salt — mixing them would stamp `complete: true` on outputs the
    recorded model ids do not describe.
    """
    return {
        "pipeline_model": pipeline_model,
        "baseline_model": baseline_model,
        "run_id": run_id,
        "prompt_hashes": prompt_hashes(),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    """One checkpoint's payload; an unreadable file is an EvalError, not a crash."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read checkpoint {path}: {exc}") from None


def identity_mismatch(expected: dict[str, Any], found: Any) -> str | None:
    """The first field where a checkpoint's identity differs, or None."""
    if not isinstance(found, dict):
        return "identity (checkpoint predates identity stamping)"
    for field, want in expected.items():
        if found.get(field) != want:
            return field
    return None


def _fingerprint_mismatch(entry: dict[str, Any], expected: str) -> str | None:
    """Why a checkpoint's thread text differs from the loaded thread's, or None."""
    found = entry.get("thread_fingerprint")
    if found is None:
        return "checkpoint predates thread fingerprinting"
    if found != expected:
        return "the rendered thread text has changed since it ran"
    return None


def _check_checkpoint_identity(
    out_dir: Path,
    thread_ids: Iterable[int],
    expected: dict[str, Any],
    fingerprints: Mapping[int, str] | None = None,
) -> None:
    """Raise EvalError if any existing checkpoint came from a different run.

    ``fingerprints`` additionally pins each checkpoint to the thread text now
    loaded: the run-level identity covers models and prompts, not what ingest
    fed the model (R32).
    """
    for thread_id in thread_ids:
        path = checkpoint_path(out_dir, thread_id)
        if not path.is_file():
            continue
        entry = load_checkpoint(path)
        field = identity_mismatch(expected, entry.get("identity"))
        if field is not None:
            raise EvalError(
                f"checkpoint {path} was produced by a different run ({field} differs); "
                "mixing runs would report results the recorded model ids do not "
                "describe. Use a fresh --out directory for this configuration, or "
                "delete that directory to re-run it from scratch."
            )
        if fingerprints is None or thread_id not in fingerprints:
            continue
        reason = _fingerprint_mismatch(entry, fingerprints[thread_id])
        if reason is not None:
            raise EvalError(
                f"checkpoint {path} was produced from different content for "
                f"thread {thread_id} ({reason}); mixing thread-text generations "
                "would report results the loaded eval threads do not describe. "
                "Use a fresh --out directory for this configuration, or delete "
                "that directory to re-run it from scratch."
            )


def resume_instruction(out_dir: Path | str) -> str:
    """How to resume a partial run (printed on every nonzero partial exit, R32)."""
    return (
        "re-run the same `triage eval` command to resume; completed threads are "
        f"skipped via their checkpoints in {checkpoint_dir(out_dir)}"
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
    re-run of an interrupted thread replays its completed calls for free. Each
    checkpoint carries its thread's text fingerprint and the tokens/cost/latency
    each arm spent on it (KTD6). Typed OutputFailures are recorded in
    checkpoints, never raised (R27); only transport-level exceptions propagate
    (the CLI turns them into a resume instruction).
    """
    out_dir = Path(out_dir)
    checkpoint_dir(out_dir).mkdir(parents=True, exist_ok=True)
    done = completed_thread_ids(out_dir)
    identity = run_identity(
        pipeline_model=pipeline_model, baseline_model=baseline_model, run_id=run_id
    )
    fingerprints = {tid: thread_fingerprint(thread) for tid, thread in threads.items()}
    # Refuse before spending anything if this out_dir holds another run's work.
    _check_checkpoint_identity(out_dir, sorted(done), identity, fingerprints)
    cache = CallCache(cache_path)
    try:
        caching_client = CachingClient(cache, inner=client, run_id=run_id)
        for thread_id in sorted(threads):
            if thread_id in done:
                continue
            thread = threads[thread_id]
            before = caching_client.usage_snapshot()
            pipeline_result = result_dict(
                run_pipeline(thread, model=pipeline_model, client=caching_client)
            )
            after_pipeline = caching_client.usage_snapshot()
            baseline = _baseline_payload(
                run_baseline(thread, model=baseline_model, client=caching_client)
            )
            entry = {
                "thread_id": thread_id,
                "run_id": run_id,
                "identity": identity,
                "thread_fingerprint": fingerprints[thread_id],
                "pipeline": pipeline_result,
                "baseline": baseline,
                "usage": {
                    "pipeline": usage_delta(before, after_pipeline),
                    "baseline": usage_delta(after_pipeline, caching_client.usage_snapshot()),
                },
                "ok": pipeline_result["ok"] and "failure" not in baseline,
            }
            write_json_atomic(checkpoint_path(out_dir, thread_id), entry)
            done.add(thread_id)
    finally:
        cache.close()
    return {"completed": len(done & set(threads)), "total": len(threads)}


# ---------------------------------------------------------------------------
# Results file — only ever written for 100%-complete runs (R32, KTD5).
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_hashes() -> dict[str, str]:
    """Frozen-prompt hashes recorded in every results file (KTD5).

    Sampling parameters are not accepted on the chosen models, so determinism
    control is pinned model ids + these hashes + measured variance.
    """
    return {
        "categorize": sha256_text(
            fragments.step_system(
                fragments.CATEGORIZE_INSTRUCTIONS, fragments.category_block()
            )
        ),
        "route": sha256_text(
            fragments.step_system(fragments.ROUTE_INSTRUCTIONS, fragments.queue_block())
        ),
        "escalate": sha256_text(fragments.step_system(fragments.ESCALATE_INSTRUCTIONS)),
        "draft": sha256_text(fragments.step_system(fragments.DRAFT_INSTRUCTIONS)),
        "baseline": sha256_text(baseline_system()),
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
    threads: Mapping[int, Thread] | None = None,
) -> Path:
    """Assemble the run's results file from checkpoints — 100%-complete only (R32).

    A partial run raises EvalError carrying the resume instruction; metrics and
    regression comparisons are computed only from files this function wrote.
    ``threads`` is the loaded eval set: when supplied, every checkpoint must
    also match its thread's current text fingerprint.
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
    fingerprints = (
        None
        if threads is None
        else {tid: thread_fingerprint(thread) for tid, thread in threads.items()}
    )
    # Defense in depth (R32): never stamp models or thread text onto outputs
    # they did not produce.
    _check_checkpoint_identity(
        out_dir,
        thread_ids,
        run_identity(
            pipeline_model=pipeline_model, baseline_model=baseline_model, run_id=run_id
        ),
        fingerprints,
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
        "thread_digest": threads_digest(
            {entry["thread_id"]: entry.get("thread_fingerprint", "") for entry in entries}
        ),
        "determinism": (
            "pinned model ids + frozen prompt hashes above + run-id salt for "
            "variance passes (KTD5); the chosen models accept no sampling parameters"
        ),
        "calls_per_thread": CALLS_PER_THREAD,
        "complete": True,
        "threads": entries,
    }
    path = out_dir / RESULTS_FILENAME
    write_json_atomic(path, results)
    return path
