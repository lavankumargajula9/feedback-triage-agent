"""Headless CLI (R7): ``triage ingest``, ``run``, ``pool``, ``label``, ``eval``.

``triage run`` processes one thread (or a batch) end-to-end through the
LangGraph pipeline and emits the structured triage result as JSON, with
distinct exit codes: 0 success, 1 pipeline failure, 2 input error. Batch runs
continue past per-thread failures and exit nonzero if any failed (R7).

``triage eval`` runs both eval arms (4-step pipeline and single-prompt
baseline, R8) over the gold-labeled set: cache-first (KTD5), checkpointed and
resumable (R32); it prints its planned call count and rough cost BEFORE
executing, and ``--dry-run`` executes nothing. ``--judge`` additionally scores
both arms' drafts (R13) and puts mean draft score into the regression gate —
opt-in, because judge calls are the priciest in the system, and counted and
priced in the same preview. Recording a reference requires that judge score
and the variance-pass spreads the R15 tolerances are derived from, so the
gate can never look armed while silently doing nothing.

Threads come from the SQLite store by tweet id (any tweet id in a thread
resolves to that thread, R31) or from an ``--input`` JSON file built into a
Thread in memory. The pipeline client is injectable via :func:`main` so tests
never touch the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from triage.evals import pool as pool_mod
from triage.ingest.download import DATASET_HANDLE, DEFAULT_CSV_PATH

# The bundled sample fixture lives in the repo, resolved relative to this file
# (editable install), with a cwd-relative fallback.
_SAMPLE_CSV_RELATIVE = Path("tests") / "fixtures" / "sample_tweets.csv"
DEFAULT_DB_PATH = Path("data") / "triage.db"
DEFAULT_SAMPLE_DB_PATH = Path("data") / "sample.db"
DEFAULT_EVAL_OUT_DIR = Path("results") / "eval"
DEFAULT_LABEL_DIR = Path("data") / "eval"
DEFAULT_CACHE_PATH = Path("data") / "llm_cache.db"

# Exit codes (R7).
EXIT_OK = 0
EXIT_PIPELINE_FAILURE = 1
EXIT_INPUT_ERROR = 2

_INPUT_SCHEMA_HINT = (
    '{"messages": [{"author": ..., "inbound": true|false, "text": ..., "created_at": ...}]}'
)

# R2: authenticity + redistribution-license verdict is a manual human step.
R2_REMINDER = (
    "Reminder (R2): the dataset authenticity and redistribution-license verdict is a\n"
    "MANUAL human step. Record the verdict in the README data section and set the\n"
    "store's eval-set distribution mode ('text' or 'id'); it stays 'pending' until then."
)


class InputError(Exception):
    """A problem with the operator's input (exit code 2, R7)."""


def _sample_csv_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / _SAMPLE_CSV_RELATIVE
    return candidate if candidate.is_file() else _SAMPLE_CSV_RELATIVE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage",
        description="Customer-feedback triage system.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser(
        "ingest",
        help="Reconstruct conversation threads from the dataset CSV into the SQLite store.",
        description=(
            "Reconstruct conversation threads (R31) from the dataset CSV into the "
            "single-file SQLite store (KTD9)."
        ),
    )
    ingest.add_argument(
        "--sample",
        action="store_true",
        help="ingest the small bundled fixture CSV instead of the Kaggle dataset (no network)",
    )
    ingest.add_argument(
        "--download",
        action="store_true",
        help=f"first fetch the pinned Kaggle dataset {DATASET_HANDLE!r} via kagglehub",
    )
    ingest.add_argument("--csv", type=Path, default=None, help="path to the dataset CSV")
    ingest.add_argument("--db", type=Path, default=None, help="path to the SQLite store")

    run = subparsers.add_parser(
        "run",
        help="Run a thread (or batch) through the triage pipeline; JSON result to stdout.",
        description=(
            "Run one thread (or a batch) end-to-end through the four-step pipeline (R3) "
            "and emit the structured triage result as JSON (R7). Exit codes: 0 success, "
            "1 pipeline failure, 2 input error. Batch runs continue past per-thread "
            "failures and exit nonzero if any failed."
        ),
    )
    run.add_argument(
        "tweet_id",
        nargs="?",
        type=int,
        default=None,
        help="a tweet id; any tweet id in a thread resolves to that thread (R31)",
    )
    run.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"JSON file describing a thread in memory: {_INPUT_SCHEMA_HINT}",
    )
    run.add_argument(
        "--batch",
        type=Path,
        default=None,
        help=(
            "file of tweet ids (one per line) or a JSON array of tweet ids and/or "
            "inline message objects; continues past per-thread failures"
        ),
    )
    run.add_argument(
        "--profile",
        choices=["dev", "measurement"],
        default="dev",
        help="model profile (R30); default dev",
    )
    run.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"path to the SQLite store (default: {DEFAULT_DB_PATH})",
    )

    pool = subparsers.add_parser(
        "pool",
        help="Build and freeze the stratified candidate pool the eval set is labeled from.",
        description=(
            "Candidate-pool construction (U6, KTD8; R11, R30). Narrows the corpus with a "
            "class-neutral structural filter, takes a seeded uniform sample, screens "
            "degenerate threads, rough-classifies the survivors on the DEV profile, and "
            "selects toward per-class floors. Prints the funnel and an itemized cost "
            "before spending; --dry-run stops there. Rough labels are a stratification "
            "aid only and are never shown to the annotator."
        ),
    )
    pool.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"path to the SQLite store (default: {DEFAULT_DB_PATH})",
    )
    pool.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"directory the frozen pool is written to (default: {DEFAULT_LABEL_DIR})",
    )
    pool.add_argument(
        "--scan-size",
        type=int,
        default=pool_mod.DEFAULT_SCAN_SIZE,
        help=f"threads to rough-classify (default: {pool_mod.DEFAULT_SCAN_SIZE})",
    )
    pool.add_argument(
        "--target-n",
        type=int,
        default=pool_mod.DEFAULT_TARGET_N,
        help=f"minimum pool size after quotas (default: {pool_mod.DEFAULT_TARGET_N})",
    )
    pool.add_argument(
        "--floor",
        type=int,
        default=pool_mod.DEFAULT_CLASS_FLOOR,
        help=f"minimum support per class (default: {pool_mod.DEFAULT_CLASS_FLOOR})",
    )
    pool.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="LLM cache for the rough pass (default: <out>/pool_cache.db)",
    )
    pool.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="CSV of thread_ids to keep out of the pool (e.g. the U5 pilot threads, KTD8)",
    )
    pool.add_argument(
        "--force",
        action="store_true",
        help="replace an existing pool even when labeling has started against it",
    )
    pool.add_argument(
        "--dry-run",
        action="store_true",
        help="print the funnel and cost estimate, then stop without any model call",
    )

    label = subparsers.add_parser(
        "label",
        help="Hand-label the eval set one field at a time (U6), or merge the passes.",
        description=(
            "Three-pass labeling (R11, R24). Each pass collects exactly one field over "
            "the whole set in its own seeded order, writing its own file; passes cannot "
            "read each other, so a later field is never anchored on an earlier answer. "
            "Run all three, then --merge to produce the gold-labels CSV."
        ),
    )
    label.add_argument(
        "--pass",
        dest="pass_name",
        choices=("category", "queue", "escalate"),
        default=None,
        help="which field this pass collects",
    )
    label.add_argument(
        "--merge",
        action="store_true",
        help="combine the three completed passes into the gold-labels CSV",
    )
    label.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"path to the SQLite store (default: {DEFAULT_DB_PATH})",
    )
    label.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"directory holding the per-pass files (default: {DEFAULT_LABEL_DIR})",
    )
    label.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="destination gold-labels CSV for --merge",
    )
    label.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many threads in this sitting (default: the whole pool)",
    )

    evaluate = subparsers.add_parser(
        "eval",
        help="Run both eval arms over the gold-labeled set; cache-first, resumable.",
        description=(
            "Cache-first (KTD5), checkpointed, resumable (R32) execution of both eval "
            "arms — the 4-step pipeline and the single-prompt baseline (R8) — over the "
            "gold-labeled eval set. Prints the planned call count and rough cost BEFORE "
            "executing. Exit codes: 0 complete, 1 partial/interrupted (with a resume "
            "instruction), 2 input error."
        ),
    )
    evaluate.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="gold-labels CSV (columns: thread_id,category,queue,escalate; from U6)",
    )
    evaluate.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"path to the SQLite store (default: {DEFAULT_DB_PATH})",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "checkpoint/results directory "
            f"(default: {DEFAULT_EVAL_OUT_DIR}, or {DEFAULT_EVAL_OUT_DIR}-<run-id>)"
        ),
    )
    evaluate.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"SQLite LLM call cache shared across runs (KTD5; default: {DEFAULT_CACHE_PATH})",
    )
    evaluate.add_argument(
        "--profile",
        choices=["dev", "measurement"],
        default="dev",
        help="model profile (R30); default dev — measurement only for final recorded runs",
    )
    evaluate.add_argument(
        "--run-id",
        default="",
        help="salt for variance passes: bypasses prior cache entries (KTD5)",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned call count and rough cost, then execute nothing",
    )
    evaluate.add_argument(
        "--judge",
        action="store_true",
        help=(
            "score both arms' drafts with the judge model (R13) and gate mean draft "
            "score; OPT-IN because judge calls are the priciest in the system"
        ),
    )
    evaluate.add_argument(
        "--anchors",
        type=Path,
        default=None,
        help=(
            "held-out anchor-stock JSON supplying the judge's few-shot examples "
            "(R14; required with --judge)"
        ),
    )
    evaluate.add_argument(
        "--spreads",
        type=Path,
        default=None,
        help=(
            "JSON map of gated metric -> observed run-to-run spread from the variance "
            "passes (R21); the R15 tolerances are derived from it"
        ),
    )
    evaluate.add_argument(
        "--variance-results",
        type=Path,
        action="append",
        default=None,
        metavar="RESULTS",
        help=(
            "results.json from a prior variance pass (repeat 2+ times, 3 per R21) to "
            "derive the spreads instead of supplying them with --spreads"
        ),
    )
    evaluate.add_argument(
        "--regression",
        type=Path,
        default=None,
        metavar="REFERENCE",
        help=(
            "after the run completes, check the gated metrics against this recorded "
            "reference artifact (R15); fails naming the metric, hard-warns on "
            "environment drift"
        ),
    )
    evaluate.add_argument(
        "--record-reference",
        type=Path,
        default=None,
        metavar="REFERENCE",
        help=(
            "after the run completes, record the reference artifact here — the ONLY "
            "way a reference is ever recorded (KTD10)"
        ),
    )
    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    from triage.ingest import IngestError, store
    from triage.ingest.download import download_dataset

    try:
        if args.sample:
            csv_path = args.csv or _sample_csv_path()
            db_path = args.db or DEFAULT_SAMPLE_DB_PATH
        else:
            csv_path = args.csv or DEFAULT_CSV_PATH
            db_path = args.db or DEFAULT_DB_PATH
            if args.download:
                csv_path = download_dataset()
        stats = store.ingest_csv(csv_path, db_path)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Ingested {stats['tweets']} tweets into {stats['threads']} threads -> {db_path} "
        f"(truncated: {stats['truncated']}, cycle_flagged: {stats['cycle_flagged']})"
    )
    print()
    print(R2_REMINDER)
    return 0


def _open_existing_store(db_path: Path):
    """Open the store for reading; a missing file is an input error (R7)."""
    from triage.ingest.store import open_store

    if not db_path.is_file():
        raise InputError(
            f"no store at {db_path}: run `triage ingest` (or `triage ingest --sample`) "
            "first, or pass --db"
        )
    return open_store(db_path)


def _resolve_thread(conn, tweet_id: int):
    """Resolve a tweet id to its stored thread (R31).

    When a tweet (e.g. a shared brand reply) belongs to more than one stored
    thread, the thread whose customer authored the tweet wins; otherwise the
    lowest thread id, deterministically.
    """
    from triage.ingest.store import get_thread, thread_ids_for_tweet

    thread_ids = thread_ids_for_tweet(conn, tweet_id)
    if not thread_ids:
        raise InputError(f"tweet id {tweet_id} is not in the store")
    if len(thread_ids) > 1:
        row = conn.execute(
            "SELECT author_id, inbound FROM tweets WHERE tweet_id = ?", (tweet_id,)
        ).fetchone()
        if row is not None and row["inbound"]:
            for thread_id in thread_ids:
                thread = get_thread(conn, thread_id)
                if thread.customer_author_id == row["author_id"]:
                    return thread
    return get_thread(conn, thread_ids[0])


def _thread_from_messages(payload: Any):
    """Build an in-memory Thread from the --input schema; bad shape -> InputError."""
    from triage.ingest.reconstruct import Thread, Tweet

    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise InputError(f"input JSON must look like {_INPUT_SCHEMA_HINT}")
    messages = payload["messages"]
    if not messages:
        raise InputError("input JSON has an empty 'messages' list; nothing to triage")
    tweets = []
    customer: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InputError(f"message {index} must be a JSON object")
        try:
            author = str(message["author"])
            inbound = message["inbound"]
            text = str(message["text"])
            created_at = str(message["created_at"])
        except KeyError as exc:
            raise InputError(
                f"message {index} is missing key {exc.args[0]!r} "
                "(need author, inbound, text, created_at)"
            ) from None
        if not isinstance(inbound, bool):
            raise InputError(f"message {index}: 'inbound' must be true or false")
        tweets.append(
            Tweet(
                tweet_id=index + 1,
                author_id=author,
                inbound=inbound,
                created_at=created_at,
                text=text,
                in_response_to_tweet_id=None if index == 0 else index,
            )
        )
        if customer is None and inbound:
            customer = author
    return Thread(
        root_tweet_id=1,
        customer_author_id=customer,
        tweets=tuple(tweets),
        truncated=False,
        cycle_flagged=False,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from None


def _load_json_file(path: Path) -> Any:
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from None


def _load_batch_items(path: Path) -> list[Any]:
    """Batch file: newline-separated tweet ids, or a JSON array of ids/objects."""
    text = _read_text(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        items: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(int(line))
            except ValueError:
                raise InputError(f"batch line is neither a tweet id nor JSON: {line!r}") from None
        if not items:
            raise InputError(f"batch file {path} is empty")
        return items
    if isinstance(payload, int):
        # A one-line batch file holding a single tweet id parses as bare JSON
        # before the newline path above ever runs.
        return [payload]
    if not isinstance(payload, list) or not payload:
        raise InputError(
            "batch JSON must be a non-empty array of tweet ids and/or message objects"
        )
    return payload


def _run_one(thread, *, model: str, client: Any) -> dict[str, Any]:
    from triage.pipeline import result_dict, run_pipeline

    return result_dict(run_pipeline(thread, model=model, client=client))


def _run_batch(args: argparse.Namespace, *, model: str, client: Any) -> int:
    """Batch mode (R7): continue past per-thread failures; nonzero if any failed."""
    items = _load_batch_items(args.batch)
    conn = None
    if any(isinstance(item, int) for item in items):
        conn = _open_existing_store(args.db or DEFAULT_DB_PATH)
    entries: list[dict[str, Any]] = []
    any_failed = False
    try:
        for index, item in enumerate(items):
            entry: dict[str, Any] = {
                "input": item if isinstance(item, int) else f"inline[{index}]"
            }
            try:
                if isinstance(item, int):
                    thread = _resolve_thread(conn, item)
                else:
                    thread = _thread_from_messages(item)
            except InputError as exc:
                entry.update(ok=False, error=str(exc))
                any_failed = True
                entries.append(entry)
                continue
            result = _run_one(thread, model=model, client=client)
            entry.update(ok=result["ok"], result=result)
            any_failed = any_failed or not result["ok"]
            entries.append(entry)
    finally:
        if conn is not None:
            conn.close()
    print(json.dumps(entries, indent=2))
    return EXIT_PIPELINE_FAILURE if any_failed else EXIT_OK


def _cmd_run(args: argparse.Namespace, client: Any = None) -> int:
    from triage.config import ROLE_PIPELINE, get_model

    model = get_model(ROLE_PIPELINE, args.profile)
    sources = sum(x is not None for x in (args.tweet_id, args.input, args.batch))
    if sources != 1:
        print(
            "error: provide exactly one of <tweet_id>, --input, or --batch",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR
    try:
        if args.batch is not None:
            return _run_batch(args, model=model, client=client)
        if args.input is not None:
            thread = _thread_from_messages(_load_json_file(args.input))
        else:
            conn = _open_existing_store(args.db or DEFAULT_DB_PATH)
            try:
                thread = _resolve_thread(conn, args.tweet_id)
            finally:
                conn.close()
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    result = _run_one(thread, model=model, client=client)
    print(json.dumps(result, indent=2))
    return EXIT_OK if result["ok"] else EXIT_PIPELINE_FAILURE


def _load_eval_threads(args: argparse.Namespace, thread_ids) -> dict[int, Any]:
    """Resolve every gold-labeled thread id against the store (input errors, R7)."""
    from triage.ingest import IngestError
    from triage.ingest.store import add_eval_item, get_thread

    conn = _open_existing_store(args.db or DEFAULT_DB_PATH)
    try:
        threads = {}
        for thread_id in thread_ids:
            try:
                threads[thread_id] = get_thread(conn, thread_id)
            except IngestError:
                raise InputError(
                    f"gold-labeled thread {thread_id} is not in the store "
                    f"({args.db or DEFAULT_DB_PATH}); labels and store must describe "
                    "the same eval set"
                ) from None
            # Arms the store's gold-label guard (R11): labels live in the CSV, but
            # ingest can only refuse to rewrite a labeled thread it knows about.
            add_eval_item(conn, thread_id, "gold")
        conn.commit()
        return threads
    finally:
        conn.close()


def _judge_options(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the judge/gate flags BEFORE anything is spent (R13, R15, KTD10).

    Every way the draft gate could end up decorative — no judge, no anchor
    stock, no measured spreads, a reference that gates a metric this run will
    not produce — is refused here, where refusing is free.
    """
    from triage.config import ROLE_JUDGE, get_model
    from triage.evals.judge import judge_plan, load_anchor_stock
    from triage.evals.regression import DRAFT_SCORE_METRIC, RegressionError, load_reference

    reference = None
    if args.regression is not None:
        try:
            reference = load_reference(args.regression)
        except RegressionError as exc:
            raise InputError(str(exc)) from None
        if DRAFT_SCORE_METRIC in reference["metrics"] and not args.judge:
            raise InputError(
                f"the reference gates {DRAFT_SCORE_METRIC!r} but this run would produce "
                "no draft score: re-run with --judge --anchors <anchor stock>, or the "
                "draft-quality gate silently does not exist (R15)"
            )
    if args.record_reference is not None and not args.judge:
        raise InputError(
            f"--record-reference needs a judge-scored {DRAFT_SCORE_METRIC!r}: re-run with "
            "--judge --anchors <anchor stock> (R13); a reference recorded without it "
            "would pass the draft gate vacuously"
        )
    variance = args.variance_results or []
    if args.record_reference is not None and args.spreads is None and not variance:
        raise InputError(
            "--record-reference needs the variance-pass spreads (R21): pass "
            "--spreads <map.json>, or --variance-results <results.json> at least twice; "
            "without them every tolerance collapses to the R15 floor"
        )
    if args.spreads is not None and variance:
        raise InputError("pass either --spreads or --variance-results, not both")
    if len(variance) == 1:
        raise InputError(
            "--variance-results needs at least two passes to measure a spread "
            "(R21 runs three); pass it again, or supply --spreads directly"
        )
    if args.judge and args.anchors is None:
        raise InputError(
            "--judge needs --anchors <anchor stock JSON>: the judge's few-shot anchors "
            "come from the separate held-out anchor stock, never from the eval set (R14)"
        )
    judge_model = get_model(ROLE_JUDGE, args.profile) if args.judge else None
    return {
        "reference": reference,
        "anchors": load_anchor_stock(args.anchors) if args.judge else [],
        "judge_model": judge_model,
        "plan": judge_plan(judge_model) if args.judge else None,
        "spreads": _load_spreads(args.spreads) if args.spreads is not None else None,
    }


def _load_spreads(path: Path) -> dict[str, float]:
    """The observed run-to-run spread per gated metric (R15, R21)."""
    payload = _load_json_file(path)
    if not isinstance(payload, dict) or not payload:
        raise InputError(
            f"{path} must be a non-empty JSON object mapping gated metric names to "
            "their observed run-to-run spread (R21)"
        )
    try:
        return {str(name): float(value) for name, value in payload.items()}
    except (TypeError, ValueError):
        raise InputError(f"{path} has a non-numeric spread value") from None


def _score_drafts(
    results: dict[str, Any],
    labels,
    *,
    args: argparse.Namespace,
    options: dict[str, Any],
    out_dir: Path,
    client: Any,
    threads: dict[int, str],
) -> dict[str, Any]:
    """Judge both arms' drafts for one complete run (R13, R32, KTD5)."""
    from triage.evals.judge import score_run

    return score_run(
        results,
        labels,
        model=options["judge_model"],
        anchors=options["anchors"],
        out_dir=out_dir,
        cache_path=args.cache,
        client=client,
        threads=threads,
        run_id=results.get("run_id", ""),
    )


def _format_judge_summary(report: dict[str, Any]) -> str:
    """Draft quality reported alongside, never blended into, the metrics (R13)."""
    systems = report["systems"]
    scores = ", ".join(
        f"{name}={summary['mean_draft_score']:.4f}" for name, summary in systems.items()
    )
    disclosure = ", ".join(
        f"{name} {summary['unscorable']['count']}/{summary['unscorable']['total']} "
        f"({summary['unscorable']['rate']:.1%})"
        for name, summary in systems.items()
    )
    return (
        f"Judge (R13, model={report['model']}): mean draft score {scores}\n"
        f"Unscorable drafts, counted not dropped (R27): {disclosure}"
    )


def _resolve_spreads(
    args: argparse.Namespace,
    labels,
    results: dict[str, Any],
    *,
    options: dict[str, Any],
    client: Any,
    threads: dict[int, str],
) -> dict[str, float]:
    """The R15 tolerances' input: supplied spreads, or spreads measured across
    variance-pass results files (R21)."""
    from triage.evals.metrics import compute_report
    from triage.evals.regression import environment_of, gated_metrics

    if options["spreads"] is not None:
        return options["spreads"]
    values: dict[str, list[float]] = {}
    for path in args.variance_results:
        pass_results = _load_json_file(path)
        try:
            same_environment = environment_of(pass_results) == environment_of(results)
        except (KeyError, TypeError):
            raise InputError(f"{path} is not a complete eval results file") from None
        if not same_environment:
            raise InputError(
                f"variance pass {path} was produced in a different environment (models, "
                "prompt hashes or eval set differ), so its spread does not describe this "
                "run (R21, KTD10)"
            )
        mean_draft_score = None
        if args.judge:
            report = _score_drafts(
                pass_results, labels, args=args, options=options,
                out_dir=Path(path).parent, client=client, threads=threads,
            )
            mean_draft_score = report["systems"]["pipeline"]["mean_draft_score"]
        pass_metrics = gated_metrics(
            compute_report(pass_results, labels), mean_draft_score=mean_draft_score
        )
        for name, value in pass_metrics.items():
            values.setdefault(name, []).append(value)
    return {name: max(observed) - min(observed) for name, observed in values.items()}


def _cmd_eval(args: argparse.Namespace, client: Any = None) -> int:
    """Cache-first, checkpointed, resumable eval over both arms (R8, R32, KTD5)."""
    from triage.config import ROLE_BASELINE, ROLE_PIPELINE, get_model
    from triage.evals.runner import (
        EvalError,
        completed_thread_ids,
        format_preview,
        load_gold_labels,
        resume_instruction,
        run_eval,
        write_results,
    )

    if args.labels is None:
        print(
            "error: --labels is required: point it at the gold-labels CSV "
            "(columns: thread_id,category,queue,escalate; produced by the U6 "
            "labeling pass)",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR
    pipeline_model = get_model(ROLE_PIPELINE, args.profile)
    baseline_model = get_model(ROLE_BASELINE, args.profile)
    out_dir = args.out
    if out_dir is None:
        suffix = f"-{args.run_id}" if args.run_id else ""
        out_dir = DEFAULT_EVAL_OUT_DIR.with_name(DEFAULT_EVAL_OUT_DIR.name + suffix)
    try:
        options = _judge_options(args)
        labels = load_gold_labels(args.labels)
        threads = _load_eval_threads(args, list(labels))
    except (EvalError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # R32: the planned call count and rough cost print BEFORE any execution.
    done = completed_thread_ids(out_dir) & set(threads)
    print(
        format_preview(
            len(threads) - len(done),
            len(threads),
            pipeline_model=pipeline_model,
            baseline_model=baseline_model,
            profile=args.profile,
            run_id=args.run_id,
            judge=options["plan"],
        )
    )
    if args.dry_run:
        print("Dry run: nothing executed.")
        return EXIT_OK

    try:
        run_eval(
            threads,
            out_dir=out_dir,
            cache_path=args.cache,
            pipeline_model=pipeline_model,
            baseline_model=baseline_model,
            run_id=args.run_id,
            client=client,
        )
    # Deliberately broad, and KeyboardInterrupt is not an Exception: ANY mid-run
    # stop must yield the resume instruction rather than a bare stack trace (R32).
    except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001
        done_now = len(completed_thread_ids(out_dir) & set(threads))
        print(f"error: eval run interrupted: {exc or type(exc).__name__}", file=sys.stderr)
        print(
            f"{done_now}/{len(threads)} threads checkpointed — {resume_instruction(out_dir)}",
            file=sys.stderr,
        )
        return EXIT_PIPELINE_FAILURE
    try:
        results_path = write_results(
            out_dir,
            list(threads),
            labels_path=args.labels,
            profile=args.profile,
            pipeline_model=pipeline_model,
            baseline_model=baseline_model,
            run_id=args.run_id,
            threads=threads,
        )
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_PIPELINE_FAILURE
    print(f"Results: {results_path}")
    if not (args.judge or args.record_reference is not None or args.regression is not None):
        return EXIT_OK
    return _eval_judge_and_gate(
        args, options, results_path, labels, threads, out_dir=out_dir, client=client
    )


def _eval_judge_and_gate(
    args: argparse.Namespace,
    options: dict[str, Any],
    results_path: Path,
    labels,
    threads,
    *,
    out_dir: Path,
    client: Any,
) -> int:
    """Judge the drafts, then record and/or check the reference (R13, R15, KTD10)."""
    from triage.evals import regression
    from triage.evals.judge import judge_checkpoint_dir
    from triage.evals.metrics import compute_report
    from triage.evals.runner import EvalError
    from triage.tools.retrieval import render_thread

    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rendered = {tid: render_thread(thread) for tid, thread in threads.items()}
    mean_draft_score = None
    if args.judge:
        try:
            report = _score_drafts(
                results, labels, args=args, options=options, out_dir=out_dir,
                client=client, threads=rendered,
            )
        except EvalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_PIPELINE_FAILURE
        # Deliberately broad, and KeyboardInterrupt is not an Exception: judge
        # calls are the most expensive in the system, so any mid-scoring stop
        # must yield the resume instruction (R32).
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001
            print(
                f"error: judge scoring interrupted: {exc or type(exc).__name__}",
                file=sys.stderr,
            )
            print(
                "re-run the same `triage eval` command to resume; already-scored "
                f"drafts are skipped via their checkpoints in {judge_checkpoint_dir(out_dir)}",
                file=sys.stderr,
            )
            return EXIT_PIPELINE_FAILURE
        mean_draft_score = report["systems"]["pipeline"]["mean_draft_score"]
        print(_format_judge_summary(report))
    gated = regression.gated_metrics(
        compute_report(results, labels), mean_draft_score=mean_draft_score
    )
    if args.record_reference is not None:
        try:
            spreads = _resolve_spreads(
                args, labels, results, options=options, client=client, threads=rendered
            )
            # KTD10: this explicit flag is the ONLY path that records a reference.
            reference_path = regression.record_reference(
                results, gated, spreads=spreads, path=args.record_reference
            )
        except (InputError, EvalError, regression.RegressionError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        print(f"Reference recorded: {reference_path}")
    reference = options["reference"]
    if reference is None:
        return EXIT_OK
    check = regression.check_regression(
        gated, reference, current_env=regression.environment_of(results)
    )
    # Hard environment-drift warnings, distinct from metric failures (R15).
    for warning in check["drift"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    for failure in check["failures"]:
        print(f"error: {failure['message']}", file=sys.stderr)
    if not check["ok"]:
        return EXIT_PIPELINE_FAILURE
    print(
        f"Regression check passed: {len(check['checked'])} gated metrics within "
        f"tolerance of {args.regression}"
    )
    return EXIT_OK


def _read_thread_ids(path: Path) -> set[int]:
    """Thread ids from a one-column CSV, with or without a ``thread_id`` header."""
    import csv

    if not path.is_file():
        raise InputError(f"no such file: {path}")
    ids: set[int] = set()
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from None
    for row in rows:
        if not row or not row[0].strip():
            continue
        field = row[0].strip()
        if field == "thread_id":
            continue
        try:
            ids.add(int(field))
        except ValueError:
            raise InputError(f"{path}: {field!r} is not a thread id") from None
    if not ids:
        raise InputError(f"{path} contains no thread ids")
    return ids


def _cmd_pool(args: argparse.Namespace, *, client: Any = None) -> int:
    """Build and freeze the stratified candidate pool (U6, KTD8; R11, R30)."""
    from triage.config import DEV, ROLE_PIPELINE, get_model

    out_dir = args.out or DEFAULT_LABEL_DIR
    try:
        conn = _open_existing_store(args.db or DEFAULT_DB_PATH)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # R30 pins the stratifier to the dev profile; there is deliberately no
    # --profile flag, so a measurement model cannot select the eval pool.
    model = get_model(ROLE_PIPELINE, DEV)

    try:
        pool_mod.assert_freezable(out_dir, force=args.force)
        exclude = _read_thread_ids(args.exclude) if args.exclude else None
    except (pool_mod.PoolError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    prep = pool_mod.prepare_scan(conn, scan_size=args.scan_size, exclude=exclude)
    if not prep.survivors:
        print("error: no labelable threads survived the scan.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print(f"corpus threads:          {prep.corpus_threads}")
    print(f"structurally eligible:   {len(prep.eligible)}")
    print(f"sampled for scan:        {len(prep.scanned)}")
    print(f"survived degeneracy:     {len(prep.survivors)}")
    for criterion, count in sorted(prep.rejections.items()):
        print(f"  rejected [{criterion}]: {count}")

    estimate = pool_mod.estimate_scan_cost(len(prep.survivors), model=model)
    print(f"\nrough pass on {model} (dev profile, R30):")
    print(
        f"  {estimate['threads']} threads x {estimate['passes']} passes "
        f"= {estimate['calls']} calls"
    )
    print(
        f"  in:  {estimate['calls']} x {estimate['tokens_in_per_call']} tok "
        f"= {estimate['tokens_in_total']} tok"
    )
    print(
        f"  out: {estimate['calls']} x {estimate['tokens_out_per_call']} tok "
        f"= {estimate['tokens_out_total']} tok"
    )
    if estimate["cost"] is None:
        print(f"  cost: UNPRICED — no price on file for {model}")
    else:
        print(
            f"  cost: ${estimate['cost_before_retries']:.2f} x "
            f"{estimate['retry_factor']} retry = ${estimate['cost']:.2f}"
        )

    if args.dry_run:
        print("\ndry run: no model calls made.")
        return EXIT_OK

    stats = pool_mod.build_pool(
        conn,
        out_dir=out_dir,
        target_n=args.target_n,
        floor=args.floor,
        cache_path=args.cache,
        client=client,
        model=model,
        prep=prep,
        force=args.force,
    )

    if not stats["selected"]:
        print("error: nothing was selected; no pool frozen.", file=sys.stderr)
        return EXIT_PIPELINE_FAILURE

    print(f"\nrough-classified: {stats['rough_classified']}")
    if stats["rough_unclassified"]:
        print(
            f"unclassified:     {stats['rough_unclassified']} "
            f"(rough pass failed; still eligible, no stratum)"
        )
    print(f"selected:         {stats['selected']} (target {stats['target_n']})")
    print(f"frozen to {Path(out_dir) / pool_mod.POOL_FILE}")
    if stats["shortfalls"]:
        # A class that cannot reach its floor is a data-sparsity result, not a
        # defect — but it must never exit clean (CONCEPTS.md, per-class floor).
        print("\nclass floors NOT met — genuine data sparsity, not backfilled:")
        for bucket, achieved in sorted(stats["shortfalls"].items()):
            print(f"  {bucket}: {achieved}/{stats['class_floor']}")
        return EXIT_PIPELINE_FAILURE
    return EXIT_OK


def _cmd_label(args: argparse.Namespace) -> int:
    """Run one labeling pass, or merge the three completed passes (U6)."""
    from triage.evals import label_helper as lh

    if not args.merge and args.pass_name is None:
        print("error: pass --pass {category,queue,escalate} or --merge.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    out_dir = args.out or DEFAULT_LABEL_DIR

    if args.merge:
        dest = args.labels or (Path(out_dir) / "gold_labels.csv")
        try:
            written = lh.merge_passes(out_dir, dest)
        except lh.LabelHelperError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        print(f"merged {written} threads into {dest}")
        return EXIT_OK

    pass_ = {p.name: p for p in lh.PASSES}[args.pass_name]
    try:
        conn = _open_existing_store(args.db or DEFAULT_DB_PATH)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        items, excluded = _label_candidates(conn, out_dir)
    except pool_mod.PoolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not items:
        print("error: no labelable threads in the frozen pool.", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # Limit after the pass shuffle, never before: slicing the id-ordered pool
    # would make a part-finished pass a biased subset of it.
    pending = lh.pending_items(pass_, out_dir, items)
    if args.limit is not None:
        pending = pending[: args.limit]

    if excluded:
        print(f"excluded {excluded} thread(s) with no diagnosable content (R11)")
    print(f"pass '{pass_.name}': {len(pending)} of {len(items)} threads remaining")
    print(f"answers append to {lh.pass_path(pass_, out_dir)}")
    print("enter the number of your choice, or 'q' to stop and resume later.\n")

    options = list(pass_.options)
    for position, item in enumerate(pending, start=1):
        print("=" * 70)
        print(f"[{position}/{len(pending)}] thread {item.thread_id}\n")
        print(item.text)
        print(f"\n{pass_.prompt}")
        for index, option in enumerate(options, start=1):
            print(f"  {index}. {option}")
        # Re-prompt rather than exit: this loop runs a few hundred times per
        # pass, and int(choice)-1 would index backwards from the end on "0",
        # recording a wrong gold label without complaining.
        while True:
            choice = input("> ").strip()
            if choice.lower() in {"q", "quit"}:
                print("stopped; rerun the same command to resume.")
                return EXIT_OK
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                value = options[int(choice) - 1]
                break
            print(f"enter 1-{len(options)}, or 'q' to stop.", file=sys.stderr)
        lh.record_answer(pass_, out_dir, item.thread_id, value)

    print(f"\npass '{pass_.name}' complete.")
    return EXIT_OK


def _label_candidates(conn, out_dir: Path | str):
    """Labelable items read from the frozen candidate pool (U6, KTD8).

    Membership only — never the rough labels. Degeneracy is re-checked because
    a re-ingest between freeze and labeling can change a thread's text.
    """
    from triage.evals.label_helper import LabelItem
    from triage.ingest.store import IngestError
    from triage.tools.retrieval import degenerate_reason, get_thread_by_id, render_thread

    items, excluded = [], 0
    for thread_id in pool_mod.read_pool(out_dir):
        try:
            thread = get_thread_by_id(conn, thread_id)
        except IngestError as exc:
            raise pool_mod.PoolError(
                f"frozen pool references thread {thread_id}, absent from this store "
                f"({exc}); rebuild the pool or point --db at the store it was built from"
            ) from None
        if degenerate_reason(thread) is not None:
            excluded += 1
            continue
        items.append(LabelItem(thread_id=thread_id, text=render_thread(thread)))
    return tuple(items), excluded


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    """CLI entrypoint; ``client`` is injectable so tests never touch the network."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "run":
        return _cmd_run(args, client=client)
    if args.command == "eval":
        return _cmd_eval(args, client=client)
    if args.command == "pool":
        return _cmd_pool(args, client=client)
    if args.command == "label":
        return _cmd_label(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
