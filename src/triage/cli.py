"""Minimal CLI: only the ``ingest`` subcommand for now.

The full command surface (``triage run`` etc.) arrives in a later unit (U4);
this stays a thin argparse shell until then.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triage.ingest.download import DATASET_HANDLE, DEFAULT_CSV_PATH

# The bundled sample fixture lives in the repo, resolved relative to this file
# (editable install), with a cwd-relative fallback.
_SAMPLE_CSV_RELATIVE = Path("tests") / "fixtures" / "sample_tweets.csv"
DEFAULT_DB_PATH = Path("data") / "triage.db"
DEFAULT_SAMPLE_DB_PATH = Path("data") / "sample.db"

# R2: authenticity + redistribution-license verdict is a manual human step.
R2_REMINDER = (
    "Reminder (R2): the dataset authenticity and redistribution-license verdict is a\n"
    "MANUAL human step. Record the verdict in the README data section and set the\n"
    "store's eval-set distribution mode ('text' or 'id'); it stays 'pending' until then."
)


def _sample_csv_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / _SAMPLE_CSV_RELATIVE
    return candidate if candidate.is_file() else _SAMPLE_CSV_RELATIVE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage",
        description="Customer-feedback triage system. More commands arrive in a later unit.",
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
