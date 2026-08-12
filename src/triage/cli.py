"""Headless CLI (R7): ``triage ingest`` plus the ``triage run`` surface.

``triage run`` processes one thread (or a batch) end-to-end through the
LangGraph pipeline and emits the structured triage result as JSON, with
distinct exit codes: 0 success, 1 pipeline failure, 2 input error. Batch runs
continue past per-thread failures and exit nonzero if any failed (R7).

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

from triage.ingest.download import DATASET_HANDLE, DEFAULT_CSV_PATH

# The bundled sample fixture lives in the repo, resolved relative to this file
# (editable install), with a cwd-relative fallback.
_SAMPLE_CSV_RELATIVE = Path("tests") / "fixtures" / "sample_tweets.csv"
DEFAULT_DB_PATH = Path("data") / "triage.db"
DEFAULT_SAMPLE_DB_PATH = Path("data") / "sample.db"

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


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    """CLI entrypoint; ``client`` is injectable so tests never touch the network."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "run":
        return _cmd_run(args, client=client)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
