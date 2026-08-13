"""Candidate-pool construction for the eval set (U6, KTD8; R11, R30).

Labeling cannot iterate the whole store, so the pool is built once, frozen, and
read back by ``triage label``. Stages: structural prefilter (SQL) -> seeded
uniform sample -> degeneracy screen -> rough classification on the dev profile
-> stratified selection toward per-class floors.

Two invariants the code cannot show:

- The SQL character floor may only ever OVER-admit relative to
  ``degenerate_reason``, which stays the single definition of degeneracy (R28).
  Cleaning only removes characters, which is what makes the floor safe to apply
  ahead of the real predicate.
- Pool membership and the rough labels are written to separate files, so the
  labeling path cannot put a model guess in front of the annotator.

Why a model rough-classifies rather than a keyword rule:
``docs/solutions/tooling-decisions/keyword-stratification-biases-eval-pool.md``.
"""

from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from triage import config
from triage.evals.cache import CachingClient, CallCache, call_cost
from triage.ingest.store import get_thread
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import MIN_DIAGNOSABLE_WORDS, clean_text, degenerate_reason
from triage.tools.schemas import CategoryLabel, OutputFailure, QueueLabel

POOL_FILE = "candidate_pool.csv"
ROUGH_FILE = "pool_rough.csv"
STATS_FILE = "pool_stats.json"

DEFAULT_SCAN_SIZE = 4000
DEFAULT_TARGET_N = 80
DEFAULT_CLASS_FLOOR = 12
SCAN_SEED = 90218837
SELECT_SEED = 5512099

# Every scanned thread costs one call per field, so the opening message is
# capped rather than sent whole; the rough pass only needs enough text to guess
# a stratum.
OPENING_CHAR_LIMIT = 600

# Pre-spend estimate inputs: a bare-label system prompt plus a capped opening
# message, against a one-label answer.
EST_TOKENS_IN = 240
EST_TOKENS_OUT = 10
EST_RETRY_FACTOR = 1.10

# A content word needs at least one character, and three of them need two
# separators, so raw customer text shorter than this cannot clear
# MIN_DIAGNOSABLE_WORDS however it is cleaned. Cleaning only ever removes
# characters, which is what makes the SQL floor safe to apply before the real
# predicate runs.
MIN_CUSTOMER_CHARS = 2 * MIN_DIAGNOSABLE_WORDS - 1

CRITERION_NO_CUSTOMER_TWEET = "no_customer_tweet"
CRITERION_TEXT_BELOW_FLOOR = "customer_text_below_floor"
CRITERION_NOT_SAMPLED = "not_sampled"
CRITERION_DEGENERATE = "degenerate"
CRITERION_ROUGH_FAILED = "rough_classify_failed"
CRITERION_NOT_SELECTED = "not_selected_by_stratification"
CRITERION_EXCLUDED = "excluded_by_caller"

BUCKET_CATEGORY = "category"
BUCKET_QUEUE = "queue"
BUCKET_ESCALATE = "escalate"


def bucket_key(kind: str, value: Any) -> str:
    """The one encoding of a quota-bucket name.

    Support keys and quota keys are looked up against each other, so a second
    encoding of this scheme would not raise — the quotas would simply never be
    met.
    """
    if isinstance(value, bool):
        value = str(value).lower()
    return f"{kind}:{value}"


ESCALATE_BUCKETS = (
    bucket_key(BUCKET_ESCALATE, True),
    bucket_key(BUCKET_ESCALATE, False),
)


class PoolError(Exception):
    """Raised when a pool cannot be built, read, or is missing entirely."""


# Rough-pass schemas: bare labels. The pipeline schemas' rationale field would
# multiply this pass's output tokens for a judgement nothing downstream reads.


class RoughCategory(BaseModel):
    """One category label, nothing else."""

    label: CategoryLabel = Field(description="Exactly one category label.")


class RoughQueue(BaseModel):
    """One queue label, nothing else."""

    queue: QueueLabel = Field(description="Exactly one support-queue label.")


class RoughEscalate(BaseModel):
    """The binary escalation guess, nothing else."""

    escalate: bool = Field(description="True when this thread needs human attention.")


@dataclass(frozen=True)
class RoughPass:
    """One rough-classification sweep over the scan pool.

    One field per sweep: a combined call would anchor queue on the category
    just chosen, under-representing category/queue divergence in the pool.
    """

    name: str
    schema: type[BaseModel]
    attribute: str
    labels: tuple[str, ...]
    instruction: str


ROUGH_PASSES: tuple[RoughPass, ...] = (
    RoughPass(
        name="category",
        schema=RoughCategory,
        attribute="label",
        labels=fragments.CATEGORY_LABELS,
        instruction="Reply with the one category label that best fits what this is about.",
    ),
    RoughPass(
        name="queue",
        schema=RoughQueue,
        attribute="queue",
        labels=fragments.QUEUE_LABELS,
        instruction="Reply with the one support-queue label for the team that should own this.",
    ),
    RoughPass(
        name="escalate",
        schema=RoughEscalate,
        attribute="escalate",
        labels=(),
        instruction="Reply true if this needs human attention now, false otherwise.",
    ),
)


@dataclass(frozen=True)
class RoughLabels:
    """One thread's rough stratum assignment."""

    category: str
    queue: str
    escalate: bool

    def buckets(self) -> tuple[str, str, str]:
        """The three quota buckets this thread contributes to."""
        return (
            bucket_key(BUCKET_CATEGORY, self.category),
            bucket_key(BUCKET_QUEUE, self.queue),
            bucket_key(BUCKET_ESCALATE, self.escalate),
        )


def manifest() -> dict[str, Any]:
    """Seeds and rough-pass shape, for the selection log's methodology entry."""
    return {
        "scan_seed": SCAN_SEED,
        "select_seed": SELECT_SEED,
        "min_customer_chars": MIN_CUSTOMER_CHARS,
        "opening_char_limit": OPENING_CHAR_LIMIT,
        "rough_passes": [p.name for p in ROUGH_PASSES],
        "profile": config.DEV,
    }


# ---------------------------------------------------------------------------
# Stage 1 — structural prefilter.
# ---------------------------------------------------------------------------

_ELIGIBLE_AGGREGATE = """
SELECT th.thread_id AS thread_id,
       SUM(CASE WHEN t.inbound = 1 THEN 1 ELSE 0 END) AS inbound_tweets,
       SUM(CASE WHEN t.inbound = 1 THEN LENGTH(t.text) ELSE 0 END) AS customer_chars
FROM threads th
JOIN thread_tweets tt ON tt.thread_id = th.thread_id
JOIN tweets t ON t.tweet_id = tt.tweet_id
GROUP BY th.thread_id
"""


def structural_prefilter(conn: sqlite3.Connection) -> tuple[list[int], Counter]:
    """Thread ids passing the class-neutral structural criteria, plus rejections.

    One grouped scan over the corpus, replacing a per-thread fetch per stored
    thread.
    """
    eligible: list[int] = []
    rejected: Counter = Counter()
    for row in conn.execute(_ELIGIBLE_AGGREGATE):
        thread_id, inbound_tweets, customer_chars = row[0], row[1] or 0, row[2] or 0
        if inbound_tweets == 0:
            rejected[CRITERION_NO_CUSTOMER_TWEET] += 1
        elif customer_chars < MIN_CUSTOMER_CHARS:
            rejected[CRITERION_TEXT_BELOW_FLOOR] += 1
        else:
            eligible.append(thread_id)
    return eligible, rejected


# ---------------------------------------------------------------------------
# Stage 2 and 3 — seeded sample, then the authoritative degeneracy screen.
# ---------------------------------------------------------------------------


def sample_scan_pool(
    eligible: list[int], scan_size: int, *, seed: int = SCAN_SEED
) -> list[int]:
    """A uniform seeded draw of ``scan_size`` ids, in ascending order."""
    if scan_size >= len(eligible):
        return sorted(eligible)
    return sorted(random.Random(seed).sample(sorted(eligible), scan_size))


def screen_degenerate(
    conn: sqlite3.Connection, thread_ids: list[int], *, exclude: set[int] | None = None
) -> tuple[list[tuple[int, str]], Counter]:
    """Drop degenerate and excluded threads, returning survivors and rejections.

    Survivors carry their opening customer message — the only text the rough
    pass sends. ``exclude`` holds threads spent elsewhere (the U5 pilot uses
    non-eval threads, per KTD8).
    """
    exclude = exclude or set()
    survivors: list[tuple[int, str]] = []
    rejected: Counter = Counter()
    for thread_id in thread_ids:
        if thread_id in exclude:
            rejected[CRITERION_EXCLUDED] += 1
            continue
        thread = get_thread(conn, thread_id)
        if degenerate_reason(thread) is not None:
            rejected[CRITERION_DEGENERATE] += 1
            continue
        survivors.append((thread_id, opening_message(thread)))
    return survivors, rejected


def opening_message(thread: Any) -> str:
    """The first customer message, cleaned and capped — the rough pass payload."""
    for tweet in thread.tweets:
        if tweet.inbound:
            text = clean_text(tweet.text)
            if text:
                return text[:OPENING_CHAR_LIMIT]
    return ""


# ---------------------------------------------------------------------------
# Stage 4a — rough classification on the dev profile (R30).
# ---------------------------------------------------------------------------


def _rough_system(pass_: RoughPass) -> str:
    """Bare label names, no definitions — the guess only picks a stratum."""
    parts = [
        "You sort customer-support threads into buckets. Answer with the label only.",
    ]
    if pass_.labels:
        parts.append("Labels (use one, exact wording): " + ", ".join(pass_.labels))
    parts.append(pass_.instruction)
    return "\n".join(parts)


def rough_classify(
    survivors: list[tuple[int, str]],
    *,
    model: str,
    client: Any,
    passes: tuple[RoughPass, ...] = ROUGH_PASSES,
) -> tuple[dict[int, RoughLabels], Counter]:
    """Rough per-field labels for each survivor, plus per-field failure counts.

    A thread whose any field fails after retries is dropped from the pool rather
    than guessed into a stratum (R27 keeps the failure typed, not raised).
    """
    collected: dict[int, dict[str, Any]] = {tid: {} for tid, _ in survivors}
    failed: set[int] = set()
    failures: Counter = Counter()
    for pass_ in passes:
        system = _rough_system(pass_)
        for thread_id, text in survivors:
            if thread_id in failed:
                continue
            result = call_with_schema(
                f"rough_{pass_.name}",
                model=model,
                system=system,
                user_text=text,
                schema=pass_.schema,
                client=client,
                max_tokens=64,
            )
            if isinstance(result, OutputFailure):
                failures[f"{CRITERION_ROUGH_FAILED}:{pass_.name}"] += 1
                failed.add(thread_id)
                continue
            collected[thread_id][pass_.name] = getattr(result, pass_.attribute)

    rough: dict[int, RoughLabels] = {}
    for thread_id, fields in collected.items():
        if thread_id in failed or len(fields) != len(passes):
            continue
        rough[thread_id] = RoughLabels(
            category=fields["category"],
            queue=fields["queue"],
            escalate=bool(fields["escalate"]),
        )
    return rough, failures


def estimate_scan_cost(
    survivor_count: int,
    *,
    model: str,
    tokens_in: int = EST_TOKENS_IN,
    tokens_out: int = EST_TOKENS_OUT,
    passes: int = len(ROUGH_PASSES),
    retry_factor: float = EST_RETRY_FACTOR,
) -> dict[str, Any]:
    """Itemized cost of the rough pass, for the pre-spend gate (R32)."""
    calls = survivor_count * passes
    total_in = calls * tokens_in
    total_out = calls * tokens_out
    base = call_cost(model, total_in, total_out)
    return {
        "model": model,
        "threads": survivor_count,
        "passes": passes,
        "calls": calls,
        "tokens_in_per_call": tokens_in,
        "tokens_out_per_call": tokens_out,
        "tokens_in_total": total_in,
        "tokens_out_total": total_out,
        "cost_before_retries": base,
        "retry_factor": retry_factor,
        "cost": None if base is None else base * retry_factor,
    }


# ---------------------------------------------------------------------------
# Stage 4b — stratified selection.
# ---------------------------------------------------------------------------


def quota_buckets(floor: int) -> dict[str, int]:
    """Every bucket that carries a minimum-support floor (R11)."""
    buckets = {
        bucket_key(BUCKET_CATEGORY, label): floor
        for label in fragments.CATEGORY_LABELS
    }
    buckets.update(
        {bucket_key(BUCKET_QUEUE, label): floor for label in fragments.QUEUE_LABELS}
    )
    buckets.update(dict.fromkeys(ESCALATE_BUCKETS, floor))
    return buckets


def stratified_select(
    rough: dict[int, RoughLabels],
    *,
    target_n: int = DEFAULT_TARGET_N,
    floor: int = DEFAULT_CLASS_FLOOR,
    seed: int = SELECT_SEED,
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    """Select toward per-class floors, then top up to ``target_n``.

    Returns the selected ids, achieved support per bucket, and any bucket left
    short of its floor. Quotas are the hard constraint: a selection that
    satisfies them is never truncated back to ``target_n``, because truncating
    would break the support the floors exist to guarantee.
    """
    quotas = quota_buckets(floor)
    members: dict[str, list[int]] = {bucket: [] for bucket in quotas}
    for thread_id, labels in sorted(rough.items()):
        for bucket in labels.buckets():
            members[bucket].append(thread_id)

    rng = random.Random(seed)
    for candidates in members.values():
        rng.shuffle(candidates)

    selected: list[int] = []
    chosen: set[int] = set()
    support: Counter = Counter()
    exhausted: set[str] = set()

    def take(thread_id: int) -> None:
        selected.append(thread_id)
        chosen.add(thread_id)
        for bucket in rough[thread_id].buckets():
            support[bucket] += 1

    ordering = list(quotas)
    while True:
        deficits = [
            (quotas[b] - support[b], b)
            for b in ordering
            if b not in exhausted and support[b] < quotas[b]
        ]
        if not deficits:
            break
        # max() returns the first maximal element, so iterating a canonical
        # bucket order makes ties resolve identically on every run.
        _, bucket = max(deficits, key=lambda item: item[0])
        available = [tid for tid in members[bucket] if tid not in chosen]
        if not available:
            exhausted.add(bucket)
            continue
        take(available[0])

    if len(selected) < target_n:
        remaining = [tid for tid in sorted(rough) if tid not in chosen]
        rng.shuffle(remaining)
        for thread_id in remaining[: target_n - len(selected)]:
            take(thread_id)

    shortfalls = {b: support[b] for b in quotas if support[b] < quotas[b]}
    return sorted(selected), dict(support), shortfalls


# ---------------------------------------------------------------------------
# Freeze and read.
# ---------------------------------------------------------------------------


def write_pool(
    out_dir: Path | str,
    selected: list[int],
    rough: dict[int, RoughLabels],
    stats: dict[str, Any],
) -> None:
    """Freeze the pool: membership, rough labels, and stats in separate files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / POOL_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("thread_id",))
        writer.writerows((thread_id,) for thread_id in selected)

    with open(out_dir / ROUGH_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("thread_id", "category", "queue", "escalate"))
        for thread_id in selected:
            labels = rough[thread_id]
            writer.writerow(
                (thread_id, labels.category, labels.queue, str(labels.escalate).lower())
            )

    (out_dir / STATS_FILE).write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_pool(out_dir: Path | str) -> list[int]:
    """The frozen pool's thread ids. Raises PoolError when none is frozen."""
    path = Path(out_dir) / POOL_FILE
    if not path.is_file():
        raise PoolError(
            f"no frozen candidate pool at {path}; build one first with "
            "`triage pool` (U6/KTD8)"
        )
    with open(path, encoding="utf-8", newline="") as handle:
        ids = [int(row["thread_id"]) for row in csv.DictReader(handle)]
    if not ids:
        raise PoolError(f"frozen candidate pool at {path} is empty")
    return ids


def read_stats(out_dir: Path | str) -> dict[str, Any]:
    """The frozen pool's stats record."""
    path = Path(out_dir) / STATS_FILE
    if not path.is_file():
        raise PoolError(f"no pool stats at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanPrep:
    """Everything decided before the first LLM call.

    Split out so the pre-spend gate can price the scan from the real survivor
    count rather than an estimate of it.
    """

    corpus_threads: int
    eligible: list[int]
    scanned: list[int]
    survivors: list[tuple[int, str]]
    rejections: Counter


def prepare_scan(
    conn: sqlite3.Connection,
    *,
    scan_size: int = DEFAULT_SCAN_SIZE,
    exclude: set[int] | None = None,
) -> ScanPrep:
    """Run every stage up to (not including) rough classification."""
    eligible, rejected = structural_prefilter(conn)
    scanned = sample_scan_pool(eligible, scan_size)
    rejected[CRITERION_NOT_SAMPLED] = len(eligible) - len(scanned)

    survivors, screen_rejects = screen_degenerate(conn, scanned, exclude=exclude)
    rejected.update(screen_rejects)

    return ScanPrep(
        corpus_threads=conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0],
        eligible=eligible,
        scanned=scanned,
        survivors=survivors,
        rejections=rejected,
    )


def build_pool(
    conn: sqlite3.Connection,
    *,
    out_dir: Path | str,
    scan_size: int = DEFAULT_SCAN_SIZE,
    target_n: int = DEFAULT_TARGET_N,
    floor: int = DEFAULT_CLASS_FLOOR,
    cache_path: Path | str | None = None,
    client: Any = None,
    exclude: set[int] | None = None,
    model: str | None = None,
    prep: ScanPrep | None = None,
) -> dict[str, Any]:
    """Build and freeze the candidate pool; returns the stats record."""
    model = model or config.get_model(config.ROLE_PIPELINE, config.DEV)

    if prep is None:
        prep = prepare_scan(conn, scan_size=scan_size, exclude=exclude)
    eligible, scanned, survivors = prep.eligible, prep.scanned, prep.survivors
    rejected = Counter(prep.rejections)

    if client is None:
        cache = CallCache(cache_path or Path(out_dir) / "pool_cache.db")
        client = CachingClient(cache, run_id="pool-scan")

    rough, failures = rough_classify(survivors, model=model, client=client)
    rejected.update(failures)

    selected, support, shortfalls = stratified_select(
        rough, target_n=target_n, floor=floor
    )
    rejected[CRITERION_NOT_SELECTED] = len(rough) - len(selected)

    stats = {
        "corpus_threads": prep.corpus_threads,
        "structurally_eligible": len(eligible),
        "scanned": len(scanned),
        "rough_classified": len(rough),
        "selected": len(selected),
        "class_floor": floor,
        "target_n": target_n,
        "rejections": dict(sorted(rejected.items())),
        "support": dict(sorted(support.items())),
        "shortfalls": dict(sorted(shortfalls.items())),
        "model": model,
        "manifest": manifest(),
    }
    if hasattr(client, "usage_snapshot"):
        stats["usage"] = client.usage_snapshot()

    write_pool(out_dir, selected, rough, stats)
    return stats
