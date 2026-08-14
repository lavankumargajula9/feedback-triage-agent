"""LLM judge for draft quality (R13, R14, R22, R27) — hardened by construction.

Draft quality is a DISTINCT score, reported alongside and never blended into
the classification metrics (R13). The judge:

- scores four dimensions 1-5 — correctness relative to the categorized issue,
  tone appropriateness, absence of hallucinated policy/fact claims, presence
  of an actionable next step — with the critique generated BEFORE each score
  (field order in the schema is generation order);
- also emits a binary "would a support lead send this as-is" verdict;
- uses anchored scale examples and length-neutral instructions; few-shot
  anchor examples are injected from a SEPARATE held-out anchor stock passed in
  by the caller (built in U6) — never carved from the eval set;
- runs on the judge model role, asserted distinct from the draft models at
  import time in :mod:`triage.config` (R22);
- scores both arms including escalated threads; a draft that is missing (its
  step failed) or that the judge cannot score after schema-enforced retries is
  scored at the floor and DISCLOSED as unscorable — never dropped from the
  denominator (R27, AE6);
- checkpoints every scored draft to disk and routes every call through the
  shared cache (R32, KTD5). These are the priciest calls in the system, so a
  crash keeps what was already paid for and a replay is free; each checkpoint
  is pinned to the judge model, judge prompt (anchors included) and draft text
  that produced it, and a mismatch is refused rather than mixed.

The judge-vs-human agreement protocol (R14) is frozen here, in code, before
any judge output is inspected — see :data:`AGREEMENT_PROTOCOL` and
:func:`compute_agreement`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from triage.evals.batch import (
    DEFAULT_POLL_SECONDS,
    MAX_POLL_SECONDS,
    BatchError,
    PendingBatchCall,
    batch_client_factory,
    drive_waves,
)
from triage.evals.cache import CachingClient, CallCache, call_cost
from triage.evals.runner import (
    EvalError,
    GoldLabel,
    identity_mismatch,
    load_checkpoint,
    sha256_text,
    write_json_atomic,
)
from triage.tools.llm import call_with_schema
from triage.tools.schemas import OutputFailure

JUDGE_STEP = "judge"

# The four R13 dimensions, in scoring order.
DIMENSIONS = ("correctness", "tone", "grounding", "actionability")
MIN_SCORE = 1
MAX_SCORE = 5
# R27/AE6: an unscorable draft scores the floor and stays in the denominator.
UNSCORABLE_SCORE = MIN_SCORE

# R14, frozen BEFORE any judge output is inspected: the agreement protocol.
AGREEMENT_PROTOCOL = (
    "Agreement protocol (R14, frozen in code before any judge output is inspected): "
    "the judge is validated against a 20-30 example human-labeled subsample (scores "
    "plus a one-line critique per score). The judge's few-shot anchor examples come "
    "from a separate held-out anchor stock — never carved from the eval set — and any "
    "item whose id appears in the anchor stock is EXCLUDED from agreement and never "
    "counted. Over the remaining judge-unseen items, per-dimension score pairs are "
    "pooled and reported as: exact agreement, percent-within-one-point, and "
    "quadratic-weighted kappa on the 1-5 scale, with judge-vs-human score "
    "distributions and the subsample size. Kappa convention for the degenerate "
    "all-same-score case (zero expected disagreement): 1.0 when observed disagreement "
    "is also zero, else 0.0."
)


class DimensionScore(BaseModel):
    """One dimension's judgment; the critique field precedes the score so the
    model writes its critique BEFORE committing to a number (R13)."""

    critique: str = Field(
        description="One or two sentences of critique for this dimension, written first."
    )
    score: int = Field(
        ge=MIN_SCORE,
        le=MAX_SCORE,
        description="The 1-5 score for this dimension, per the anchored scale.",
    )


class JudgeResult(BaseModel):
    """Four critique-then-score dimensions plus the binary send verdict (R13)."""

    correctness: DimensionScore = Field(
        description="Does the draft correctly address the categorized issue?"
    )
    tone: DimensionScore = Field(
        description="Is the tone appropriate for a public support reply?"
    )
    grounding: DimensionScore = Field(
        description="Is the draft free of hallucinated policy or fact claims?"
    )
    actionability: DimensionScore = Field(
        description="Does the draft give the customer a concrete, actionable next step?"
    )
    send_critique: str = Field(
        description="One or two sentences on overall send-worthiness, written first."
    )
    send_as_is: bool = Field(
        description="Would a support lead send this draft as-is, without edits?"
    )


@dataclass(frozen=True)
class AnchorExample:
    """One held-out anchor-stock example injected as a few-shot (R14).

    Anchor stock is assembled separately from the eval set (U6); items listed
    here are excluded from agreement by id.
    """

    item_id: str
    category: str
    draft: str
    scores: Mapping[str, int]
    critiques: Mapping[str, str]
    send_as_is: bool


ANCHOR_FIELDS = ("item_id", "category", "draft", "scores", "critiques", "send_as_is")


def load_anchor_stock(path: Path | str) -> list[AnchorExample]:
    """Load the held-out anchor stock the judge's few-shots come from (R14).

    Every defect is an EvalError: a malformed anchor file must never reach a
    paid judge call, and anchors carved from the eval set would invalidate the
    agreement protocol.
    """
    path = Path(path)
    if not path.is_file():
        raise EvalError(
            f"no anchor stock at {path}: the judge's few-shot anchors come from the "
            "held-out anchor stock produced by the U6 labeling pass — a JSON array of "
            f"objects with keys: {', '.join(ANCHOR_FIELDS)} (R14)"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"anchor stock {path} is not readable JSON: {exc}") from None
    if not isinstance(payload, list) or not payload:
        raise EvalError(f"anchor stock {path} must be a non-empty JSON array of anchors")
    return [_parse_anchor(item, path=path, index=index) for index, item in enumerate(payload)]


def _parse_anchor(item: Any, *, path: Path, index: int) -> AnchorExample:
    if not isinstance(item, dict):
        raise EvalError(f"anchor {index} in {path} must be a JSON object")
    missing = [field for field in ANCHOR_FIELDS if field not in item]
    if missing:
        raise EvalError(
            f"anchor {index} in {path} is missing key(s): {', '.join(missing)}; "
            f"expected: {', '.join(ANCHOR_FIELDS)}"
        )
    scores, critiques = item["scores"], item["critiques"]
    for dimension in DIMENSIONS:
        value = scores.get(dimension) if isinstance(scores, dict) else None
        if not isinstance(value, int) or not MIN_SCORE <= value <= MAX_SCORE:
            raise EvalError(
                f"anchor {index} in {path} needs an integer {dimension} score in "
                f"{MIN_SCORE}-{MAX_SCORE}; got {value!r}"
            )
        if not isinstance(critiques, dict) or not critiques.get(dimension):
            raise EvalError(f"anchor {index} in {path} is missing a {dimension} critique (R13)")
    return AnchorExample(
        item_id=str(item["item_id"]),
        category=str(item["category"]),
        draft=str(item["draft"]),
        scores={dimension: int(scores[dimension]) for dimension in DIMENSIONS},
        critiques={dimension: str(critiques[dimension]) for dimension in DIMENSIONS},
        send_as_is=bool(item["send_as_is"]),
    )


# ---------------------------------------------------------------------------
# Prompt assembly (R13): anchored scale, length-neutral, critique-then-score.
# ---------------------------------------------------------------------------

JUDGE_PREAMBLE = (
    "You are a strict support-quality judge. You grade ONE customer-support reply "
    "draft at a time against the issue it was written for. You never rewrite the "
    "draft; you only critique and score it."
)

LENGTH_NEUTRAL_INSTRUCTIONS = (
    "Judge quality, never length: a short draft that does the job outscores a long "
    "one that pads. Do not reward verbosity, hedging, or repetition, and do not "
    "penalize brevity."
)

CRITIQUE_FIRST_INSTRUCTIONS = (
    "For every dimension, write your critique FIRST, then commit to the score. "
    "Finally, give your overall send critique and the binary verdict: would a "
    "support lead send this draft as-is, without edits?"
)

SCALE_ANCHORS = (
    "Anchored 1-5 scale (the same anchors apply to every dimension):\n"
    "- 1: fails the dimension outright — e.g. addresses the wrong issue, hostile or "
    "dismissive tone, invents a policy/fact, or offers no next step at all.\n"
    "- 3: partially meets it — roughly on-topic but generic, tonally flat, makes an "
    "unverifiable minor claim, or gestures at a next step without making it concrete.\n"
    "- 5: fully meets it — squarely addresses the categorized issue, pitch-perfect "
    "public support voice, claims only what the thread supports, and names a "
    "concrete, actionable next step."
)

DIMENSION_RUBRIC = (
    "Score these four dimensions, in order:\n"
    "- correctness: is the draft a correct response to the categorized issue?\n"
    "- tone: is the tone appropriate for the brand's public support channel?\n"
    "- grounding: is the draft free of hallucinated policy or fact claims?\n"
    "- actionability: does it give the customer an actionable next step?"
)


def _render_anchor(anchor: AnchorExample) -> str:
    lines = [f"Example (category: {anchor.category}):", f"Draft: {anchor.draft}"]
    for dimension in DIMENSIONS:
        critique = anchor.critiques.get(dimension, "")
        score = anchor.scores.get(dimension)
        lines.append(f"- {dimension}: {critique} Score: {score}")
    lines.append(f"- send as-is: {'yes' if anchor.send_as_is else 'no'}")
    return "\n".join(lines)


def judge_system(anchors: Sequence[AnchorExample]) -> str:
    """The judge system prompt: rubric, anchored scale, length-neutral and
    critique-first instructions, plus the injected anchor-stock few-shots."""
    parts = [
        JUDGE_PREAMBLE,
        DIMENSION_RUBRIC,
        SCALE_ANCHORS,
        LENGTH_NEUTRAL_INSTRUCTIONS,
        CRITIQUE_FIRST_INSTRUCTIONS,
    ]
    if anchors:
        rendered = "\n\n".join(_render_anchor(anchor) for anchor in anchors)
        parts.append(f"Scored examples for calibration:\n\n{rendered}")
    return "\n\n".join(parts)


def judge_user_text(
    draft: str, *, category: str, escalated: bool, thread_text: str | None = None
) -> str:
    """What the judge grades: the draft, its categorized issue, and context."""
    lines = [
        f"Categorized issue: {category}",
        f"Thread was escalated for human attention: {'yes' if escalated else 'no'}",
    ]
    if thread_text:
        lines.append(f"Conversation thread:\n{thread_text}")
    lines.append(f"Reply draft to judge:\n{draft}")
    return "\n\n".join(lines)


def judge_draft(
    draft: str,
    *,
    category: str,
    escalated: bool,
    model: str,
    anchors: Sequence[AnchorExample],
    thread_text: str | None = None,
    client: Any = None,
) -> JudgeResult | OutputFailure:
    """Judge ONE draft through the schema-enforced wrapper (KTD4, R27)."""
    return call_with_schema(
        JUDGE_STEP,
        model=model,
        system=judge_system(anchors),
        user_text=judge_user_text(
            draft, category=category, escalated=escalated, thread_text=thread_text
        ),
        schema=JudgeResult,
        client=client,
    )


# ---------------------------------------------------------------------------
# Scoring a completed run: both arms, escalated threads included (R13).
# ---------------------------------------------------------------------------


def _pipeline_draft(entry: Mapping[str, Any]) -> str | None:
    payload = entry["pipeline"]["steps"].get("draft")
    if not payload or "failure" in payload:
        return None
    return payload["draft"]


def _baseline_draft(entry: Mapping[str, Any]) -> str | None:
    payload = entry["baseline"]
    if not payload or "failure" in payload:
        return None
    return payload["draft"]

_DRAFT_EXTRACTORS = {"pipeline": _pipeline_draft, "baseline": _baseline_draft}

# One judge call per (thread, arm) — judging is per draft, not per thread.
JUDGE_CALLS_PER_THREAD = len(_DRAFT_EXTRACTORS)

# Rough per-judge-call token budget from the plan; cost preview only.
JUDGE_TOKENS_IN_PER_CALL = 2500
JUDGE_TOKENS_OUT_PER_CALL = 500


def judge_cost_per_call(model: str) -> float | None:
    """Rough dollars per judge call, or None with no price on file."""
    return call_cost(model, JUDGE_TOKENS_IN_PER_CALL, JUDGE_TOKENS_OUT_PER_CALL)


def judge_plan(model: str) -> dict[str, Any]:
    """The judge's share of the pre-execution cost preview (R32)."""
    return {
        "model": model,
        "calls_per_thread": JUDGE_CALLS_PER_THREAD,
        "cost_per_call": judge_cost_per_call(model),
        "tokens_in": JUDGE_TOKENS_IN_PER_CALL,
        "tokens_out": JUDGE_TOKENS_OUT_PER_CALL,
    }


# ---------------------------------------------------------------------------
# Judge checkpoints (R32, KTD5) — the priciest calls in the system.
# ---------------------------------------------------------------------------

# A draft that never existed still gets a checkpoint, so a later run that DOES
# have one is a fingerprint mismatch rather than a silent reuse of the floor.
MISSING_DRAFT_FINGERPRINT = "missing-draft"


def judge_checkpoint_dir(out_dir: Path | str) -> Path:
    return Path(out_dir) / "judge"


def judge_checkpoint_path(out_dir: Path | str, system: str, thread_id: int) -> Path:
    return judge_checkpoint_dir(out_dir) / f"{system}-{thread_id}.json"


def draft_fingerprint(draft: str | None) -> str:
    """SHA-256 of the exact draft text the judge scored (R32)."""
    return MISSING_DRAFT_FINGERPRINT if draft is None else sha256_text(draft)


def judge_identity(
    *, model: str, anchors: Sequence[AnchorExample], run_id: str = ""
) -> dict[str, Any]:
    """What a judge checkpoint must have been produced by to belong here (KTD5).

    The anchors are part of the judge, so they enter the prompt hash: a
    different anchor stock is a different judge and its scores may not be
    mixed with these.
    """
    return {
        "judge_model": model,
        "judge_prompt_hash": sha256_text(judge_system(anchors)),
        "run_id": run_id,
    }


def _refuse_mismatched_checkpoints(
    out_dir: Path, items: Sequence[tuple[str, int, str | None]], identity: dict[str, Any]
) -> None:
    """Raise EvalError before spending anything if another judge ran here."""
    for system, thread_id, draft in items:
        path = judge_checkpoint_path(out_dir, system, thread_id)
        if not path.is_file():
            continue
        entry = load_checkpoint(path)
        field = identity_mismatch(identity, entry.get("identity"))
        if field is not None:
            raise EvalError(
                f"judge checkpoint {path} was produced by a different judge "
                f"({field} differs); mixing judge generations would report draft "
                "scores the recorded judge model and prompt do not describe. Use a "
                "fresh --out directory for this configuration, or delete "
                f"{judge_checkpoint_dir(out_dir)} to re-score from scratch."
            )
        found = entry.get("draft_fingerprint")
        expected = draft_fingerprint(draft)
        if found != expected:
            reason = (
                "the checkpoint predates draft fingerprinting"
                if found is None
                else "the draft text has changed since it was scored"
            )
            raise EvalError(
                f"judge checkpoint {path} scored different content for {system} "
                f"thread {thread_id} ({reason}); mixing draft generations would "
                "report scores the current run's drafts do not describe. Use a fresh "
                "--out directory, or delete that judge directory to re-score."
            )
        if "record" not in entry:
            raise EvalError(f"judge checkpoint {path} has no scored record; delete it to re-score")


def _unscorable_record(reason: str) -> dict[str, Any]:
    """The floor-scored, disclosed record for an unscorable draft (R27, AE6)."""
    return {
        "scores": {dimension: UNSCORABLE_SCORE for dimension in DIMENSIONS},
        "critiques": {dimension: f"unscorable: {reason}" for dimension in DIMENSIONS},
        "mean": float(UNSCORABLE_SCORE),
        "send_as_is": False,
        "unscorable": True,
    }


def _scored_record(result: JudgeResult) -> dict[str, Any]:
    scores = {dimension: getattr(result, dimension).score for dimension in DIMENSIONS}
    return {
        "scores": scores,
        "critiques": {dimension: getattr(result, dimension).critique for dimension in DIMENSIONS},
        "mean": mean(scores.values()),
        "send_as_is": result.send_as_is,
        "unscorable": False,
    }


def _summarize(per_thread: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    records = list(per_thread.values())
    total = len(records)
    unscorable = sum(1 for record in records if record["unscorable"])
    return {
        "per_thread": dict(per_thread),
        "mean_scores": {
            dimension: mean(record["scores"][dimension] for record in records)
            for dimension in DIMENSIONS
        },
        "mean_draft_score": mean(record["mean"] for record in records),
        "send_rate": sum(record["send_as_is"] for record in records) / total,
        # R27: unscorable drafts are counted and disclosed, never dropped.
        "unscorable": {
            "count": unscorable,
            "total": total,
            "rate": unscorable / total if total else 0.0,
        },
    }


def _judge_record(
    draft: str | None,
    gold_label: GoldLabel,
    *,
    model: str,
    anchors: Sequence[AnchorExample],
    thread_text: str | None,
    client: Any,
) -> dict[str, Any]:
    """One draft's scored (or disclosed-unscorable) record (R13, R27)."""
    if draft is None:
        return _unscorable_record("the draft step failed (R27)")
    result = judge_draft(
        draft,
        category=gold_label.category,
        escalated=gold_label.escalate,
        model=model,
        anchors=anchors,
        thread_text=thread_text,
        client=client,
    )
    if isinstance(result, OutputFailure):
        return _unscorable_record(f"judge output {result.kind} after {result.attempts} attempts")
    return _scored_record(result)


def score_run(
    results: Mapping[str, Any],
    gold: Mapping[int, GoldLabel],
    *,
    model: str,
    anchors: Sequence[AnchorExample],
    out_dir: Path | str,
    cache_path: Path | str,
    client: Any = None,
    threads: Mapping[int, str] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Judge every draft in a completed run — checkpointed and resumable
    (R13, R32, KTD5).

    Both arms are scored, escalated threads included; ``threads`` optionally
    maps thread ids to rendered thread text for extra judge context. Judging is
    per (thread, arm), so each scored draft is checkpointed under that key
    before the next call begins: an interruption keeps every judge call already
    paid for. Calls go through the shared SQLite cache, so a replay is free.
    Every checkpoint is pinned to the judge model, judge prompt (anchors
    included) and draft text that produced it, and a mismatch is refused rather
    than mixed. The result is a standalone report — draft quality is never
    blended into the classification metrics.
    """
    threads = threads or {}
    out_dir = Path(out_dir)
    judge_checkpoint_dir(out_dir).mkdir(parents=True, exist_ok=True)
    identity = judge_identity(model=model, anchors=anchors, run_id=run_id)
    items = [
        (system, entry["thread_id"], extract(entry))
        for entry in results["threads"]
        for system, extract in _DRAFT_EXTRACTORS.items()
    ]
    _refuse_mismatched_checkpoints(out_dir, items, identity)
    per_system: dict[str, dict[int, dict[str, Any]]] = {system: {} for system in _DRAFT_EXTRACTORS}
    cache = CallCache(cache_path)
    try:
        caching_client = CachingClient(cache, inner=client, run_id=run_id)
        for system, tid, draft in items:
            path = judge_checkpoint_path(out_dir, system, tid)
            if path.is_file():
                per_system[system][tid] = load_checkpoint(path)["record"]
                continue
            record = _judge_record(
                draft,
                gold[tid],
                model=model,
                anchors=anchors,
                thread_text=threads.get(tid),
                client=caching_client,
            )
            write_json_atomic(
                path,
                {
                    "thread_id": tid,
                    "system": system,
                    "identity": identity,
                    "draft_fingerprint": draft_fingerprint(draft),
                    "record": record,
                },
            )
            per_system[system][tid] = record
    finally:
        cache.close()
    return {
        "model": model,
        "n_threads": len(results["threads"]),
        "systems": {system: _summarize(records) for system, records in per_system.items()},
    }


def score_run_batch(
    results: Mapping[str, Any],
    gold: Mapping[int, GoldLabel],
    *,
    model: str,
    anchors: Sequence[AnchorExample],
    out_dir: Path | str,
    cache_path: Path | str,
    batch_client: Any = None,
    threads: Mapping[int, str] | None = None,
    run_id: str = "",
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_poll_seconds: float = MAX_POLL_SECONDS,
) -> dict[str, Any]:
    """score_run through the Messages Batch API at 50% of list price (R13, KTD5).

    Judge calls are independent per (thread, arm), so the whole set lands in
    one wave plus any R27 retry waves; checkpoints, identity guards, and the
    report shape are identical to :func:`score_run`.
    """
    threads = threads or {}
    out_dir = Path(out_dir)
    judge_checkpoint_dir(out_dir).mkdir(parents=True, exist_ok=True)
    identity = judge_identity(model=model, anchors=anchors, run_id=run_id)
    items = [
        (system, entry["thread_id"], extract(entry))
        for entry in results["threads"]
        for system, extract in _DRAFT_EXTRACTORS.items()
    ]
    _refuse_mismatched_checkpoints(out_dir, items, identity)
    per_system: dict[str, dict[int, dict[str, Any]]] = {system: {} for system in _DRAFT_EXTRACTORS}
    cache = CallCache(cache_path)
    client_factory = batch_client_factory(batch_client)

    def pass_fn(collector: Any) -> bool:
        all_done = True
        for system, tid, draft in items:
            if tid in per_system[system]:
                continue
            path = judge_checkpoint_path(out_dir, system, tid)
            if path.is_file():
                per_system[system][tid] = load_checkpoint(path)["record"]
                continue
            collector.owner = (tid, system)
            try:
                record = _judge_record(
                    draft,
                    gold[tid],
                    model=model,
                    anchors=anchors,
                    thread_text=threads.get(tid),
                    client=collector,
                )
            except PendingBatchCall:
                all_done = False
                continue
            write_json_atomic(
                path,
                {
                    "thread_id": tid,
                    "system": system,
                    "identity": identity,
                    "draft_fingerprint": draft_fingerprint(draft),
                    "record": record,
                },
            )
            per_system[system][tid] = record
        return all_done

    try:
        drive_waves(
            pass_fn,
            cache=cache,
            run_id=run_id,
            batch_client_factory=client_factory,
            poll_seconds=poll_seconds,
            max_poll_seconds=max_poll_seconds,
        )
    except BatchError as exc:
        raise EvalError(f"batch judge scoring failed: {exc}") from None
    finally:
        cache.close()
    return {
        "model": model,
        "n_threads": len(results["threads"]),
        "systems": {system: _summarize(records) for system, records in per_system.items()},
    }


# ---------------------------------------------------------------------------
# Judge-vs-human agreement (R14) — protocol frozen above.
# ---------------------------------------------------------------------------


def _as_dimensions(value: Mapping[str, int] | int) -> Mapping[str, int]:
    return value if isinstance(value, Mapping) else {"overall": value}


def quadratic_weighted_kappa(pairs: Iterable[tuple[int, int]]) -> float:
    """Quadratic-weighted kappa over 1-5 score pairs, pure stdlib.

    Weights are (i-j)^2 / (MAX-MIN)^2; expected disagreement comes from the
    two raters' marginals. Degenerate case (zero expected disagreement, i.e.
    both raters constant on the same score): 1.0 when observed disagreement is
    also zero, else 0.0 — per the frozen protocol.
    """
    pairs = list(pairs)
    n = len(pairs)
    scale = (MAX_SCORE - MIN_SCORE) ** 2
    observed = sum((a - b) ** 2 for a, b in pairs) / scale
    a_marginals = Counter(a for a, _ in pairs)
    b_marginals = Counter(b for _, b in pairs)
    expected = (
        sum(
            a_count * b_count * ((a - b) ** 2 / scale)
            for a, a_count in a_marginals.items()
            for b, b_count in b_marginals.items()
        )
        / n
    )
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1.0 - observed / expected


def compute_agreement(
    judge_scores: Mapping[Any, Mapping[str, int] | int],
    human_scores: Mapping[Any, Mapping[str, int] | int],
    *,
    anchor_ids: Iterable[Any] = (),
) -> dict[str, Any]:
    """Judge-vs-human agreement per the frozen protocol (R14).

    Items whose id appears in the anchor stock are excluded — agreement is
    computed only over items the judge never saw as examples. Scores may be
    per-dimension mappings (pooled) or single integers.
    """
    anchor_ids = set(anchor_ids)
    shared = set(judge_scores) & set(human_scores)
    items = sorted(shared - anchor_ids)
    pairs: list[tuple[int, int]] = []
    for item in items:
        judge_dims = _as_dimensions(judge_scores[item])
        human_dims = _as_dimensions(human_scores[item])
        for dimension in sorted(set(judge_dims) & set(human_dims)):
            pairs.append((judge_dims[dimension], human_dims[dimension]))
    if not pairs:
        raise ValueError(
            "no items to compute agreement over after excluding anchor-stock items (R14)"
        )
    n = len(pairs)
    return {
        "n_items": len(items),
        "n_scores": n,
        "excluded_anchor_items": len(shared & anchor_ids),
        "exact": sum(1 for a, b in pairs if a == b) / n,
        "within_one": sum(1 for a, b in pairs if abs(a - b) <= 1) / n,
        "weighted_kappa": quadratic_weighted_kappa(pairs),
        "judge_distribution": dict(sorted(Counter(a for a, _ in pairs).items())),
        "human_distribution": dict(sorted(Counter(b for _, b in pairs).items())),
    }
