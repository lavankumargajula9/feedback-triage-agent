"""SQLite thread store — single file, stdlib sqlite3 (KTD9), dual-mode export (R29).

Schema:

- ``tweets``: raw rows keyed by tweet id.
- ``threads``: one row per reconstructed thread with its R31 flags
  (``truncated``, ``cycle_flagged``) and its ``membership`` fingerprint;
  identified by (root tweet, customer).
- ``thread_tweets``: ordered thread membership (a tweet may belong to more than
  one thread when branches share a prefix).
- ``eval_items``: labeled threads for the eval set.
- ``meta``: key/value store metadata, including ``distribution_mode``.

Dual-mode eval-set distribution (R29): ``export_eval_set`` produces either the
full labeled thread text ("text" mode) or an ID manifest plus the local rebuild
command ("id" mode). Which mode ships is a MANUAL human license verdict (R2);
``distribution_mode`` therefore defaults to "pending" and is never chosen here.

Gold-label safety (R11): the hand-labeled eval set keys on ``thread_id``, so
``ingest_tweets`` refuses to rewrite or supersede a thread that carries a label
unless the operator accepts the change explicitly. An unlabeled store — the
common case before labeling starts — ingests freely and silently.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from triage.ingest import IngestError
from triage.ingest.download import DATASET_HANDLE, DOWNLOAD_COMMAND, missing_csv_error
from triage.ingest.reconstruct import Thread, Tweet, reconstruct_all

# Distribution modes (R29). "pending" = the R2 license verdict is not recorded yet.
DIST_MODE_PENDING = "pending"
DIST_MODE_TEXT = "text"
DIST_MODE_ID = "id"
DISTRIBUTION_MODES = (DIST_MODE_PENDING, DIST_MODE_TEXT, DIST_MODE_ID)
EXPORT_MODES = (DIST_MODE_TEXT, DIST_MODE_ID)

# Named in ID-mode exports: how a recipient rebuilds thread text locally.
REBUILD_COMMAND = f"{DOWNLOAD_COMMAND}  (then: triage ingest)"

# Ways a re-ingest can invalidate a frozen gold label (R11).
HAZARD_MEMBERSHIP_CHANGED = "membership_changed"
HAZARD_SUPERSEDED = "superseded"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id INTEGER PRIMARY KEY,
    author_id TEXT NOT NULL,
    inbound INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    in_response_to_tweet_id INTEGER
);
CREATE TABLE IF NOT EXISTS threads (
    thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_tweet_id INTEGER NOT NULL,
    customer_author_id TEXT,
    truncated INTEGER NOT NULL,
    cycle_flagged INTEGER NOT NULL,
    membership TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_threads_identity
    ON threads (root_tweet_id, IFNULL(customer_author_id, ''));
CREATE TABLE IF NOT EXISTS thread_tweets (
    thread_id INTEGER NOT NULL REFERENCES threads (thread_id),
    position INTEGER NOT NULL,
    tweet_id INTEGER NOT NULL REFERENCES tweets (tweet_id),
    PRIMARY KEY (thread_id, position)
);
CREATE INDEX IF NOT EXISTS idx_thread_tweets_tweet ON thread_tweets (tweet_id);
CREATE TABLE IF NOT EXISTS eval_items (
    thread_id INTEGER PRIMARY KEY REFERENCES threads (thread_id),
    label TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def load_tweets_csv(csv_path: Path | str) -> dict[int, Tweet]:
    """Load the dataset CSV into Tweet records keyed by tweet id.

    A missing file raises the instructive R25 error naming the download command.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise missing_csv_error(csv_path)
    tweets: dict[int, Tweet] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tweet = _tweet_from_row(row, csv_path)
            tweets[tweet.tweet_id] = tweet
    return tweets


def _tweet_from_row(row: dict[str, str], csv_path: Path) -> Tweet:
    try:
        return Tweet(
            tweet_id=_to_int_id(row["tweet_id"]),
            author_id=row["author_id"],
            inbound=row["inbound"].strip().lower() in ("true", "1"),
            created_at=row["created_at"],
            text=row["text"],
            in_response_to_tweet_id=_parse_optional_id(row["in_response_to_tweet_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestError(f"Malformed row in {csv_path}: {row!r} ({exc})") from exc


def _to_int_id(value: str) -> int:
    return int(float(value))  # tolerate "12.0"-style values


def _parse_optional_id(value: str | None) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    return _to_int_id(value)


def _membership(tweet_ids: tuple[int, ...] | list[int]) -> str:
    """A thread's membership fingerprint: its ordered tweet ids."""
    return ",".join(str(tweet_id) for tweet_id in tweet_ids)


def _ordered_tweet_ids(conn: sqlite3.Connection, thread_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT tweet_id FROM thread_tweets WHERE thread_id = ? ORDER BY position",
        (thread_id,),
    ).fetchall()
    return [row["tweet_id"] for row in rows]


def _add_membership_column(conn: sqlite3.Connection) -> None:
    """Add ``threads.membership`` to a store created before it existed.

    ``CREATE TABLE IF NOT EXISTS`` cannot add columns, and the gold-label guard
    reads this one, so it is backfilled from ``thread_tweets`` here rather than
    left NULL (a NULL fingerprint would silently disarm the guard).
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(threads)")}
    if "membership" in columns:
        return
    conn.execute("ALTER TABLE threads ADD COLUMN membership TEXT")
    thread_ids = [row["thread_id"] for row in conn.execute("SELECT thread_id FROM threads")]
    backfill = [(_membership(_ordered_tweet_ids(conn, tid)), tid) for tid in thread_ids]
    conn.executemany("UPDATE threads SET membership = ? WHERE thread_id = ?", backfill)


def open_store(db_path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the single-file SQLite store (KTD9)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _add_membership_column(conn)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('distribution_mode', ?)",
        (DIST_MODE_PENDING,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('dataset_handle', ?)",
        (DATASET_HANDLE,),
    )
    conn.commit()
    return conn


def _upsert_thread(conn: sqlite3.Connection, thread: Thread) -> int:
    """Return this thread's stable id, inserting it only if it is new.

    A thread's identity is (root_tweet_id, customer_author_id). Re-ingesting
    keeps the existing ``thread_id`` for an identity already in the store, so
    frozen gold labels — which key on ``thread_id`` — keep pointing at the same
    conversation across re-ingests.
    """
    row = conn.execute(
        "SELECT thread_id FROM threads "
        "WHERE root_tweet_id = ? AND IFNULL(customer_author_id, '') = IFNULL(?, '')",
        (thread.root_tweet_id, thread.customer_author_id),
    ).fetchone()
    if row is None:
        cursor = conn.execute(
            "INSERT INTO threads (root_tweet_id, customer_author_id, truncated, cycle_flagged, "
            "membership) VALUES (?, ?, ?, ?, ?)",
            (
                thread.root_tweet_id,
                thread.customer_author_id,
                int(thread.truncated),
                int(thread.cycle_flagged),
                _membership(thread.tweet_ids),
            ),
        )
        return int(cursor.lastrowid)
    thread_id = int(row["thread_id"])
    conn.execute(
        "UPDATE threads SET truncated = ?, cycle_flagged = ?, membership = ? WHERE thread_id = ?",
        (
            int(thread.truncated),
            int(thread.cycle_flagged),
            _membership(thread.tweet_ids),
            thread_id,
        ),
    )
    return thread_id


def _labeled_thread_hazards(
    conn: sqlite3.Connection, threads: list[Thread]
) -> list[dict[str, Any]]:
    """Ways this ingest would invalidate a frozen gold label (R11).

    Gold labels key on ``thread_id``. Two hazards, both invisible once written:
    the labeled thread's membership changes under a stable id, or its identity
    (root tweet, customer) no longer reconstructs — a previously-missing parent
    re-roots the conversation, which would insert it as a NEW ``thread_id`` and
    leave the labeled row describing a superseded fragment.

    A labeled thread that this ingest does not touch at all is not a hazard:
    partial CSVs are ingested cumulatively, so absence means "not in this batch".
    """
    labeled = conn.execute(
        "SELECT e.thread_id, t.root_tweet_id, t.customer_author_id, t.membership "
        "FROM eval_items e JOIN threads t ON t.thread_id = e.thread_id "
        "ORDER BY e.thread_id"
    ).fetchall()
    if not labeled:
        return []
    incoming = {thread.signature: thread for thread in threads}
    containing: dict[int, list[Thread]] = {}
    for thread in threads:
        for tweet_id in thread.tweet_ids:
            containing.setdefault(tweet_id, []).append(thread)

    hazards: list[dict[str, Any]] = []
    for row in labeled:
        stored = row["membership"] or ""
        match = incoming.get((row["root_tweet_id"], row["customer_author_id"]))
        if match is not None:
            now = _membership(match.tweet_ids)
            if stored != now:
                hazards.append(
                    {
                        "hazard": HAZARD_MEMBERSHIP_CHANGED,
                        "thread_id": row["thread_id"],
                        "was": stored,
                        "now": now,
                    }
                )
            continue
        successors = {
            thread.signature
            for tweet_id in _ordered_tweet_ids(conn, row["thread_id"])
            for thread in containing.get(tweet_id, ())
        }
        if successors:
            hazards.append(
                {
                    "hazard": HAZARD_SUPERSEDED,
                    "thread_id": row["thread_id"],
                    "was": stored,
                    "now_in": [
                        {"root_tweet_id": root, "customer_author_id": customer}
                        for root, customer in sorted(successors, key=lambda s: (s[0], s[1] or ""))
                    ],
                }
            )
    return hazards


def _describe_hazard(hazard: dict[str, Any]) -> str:
    if hazard["hazard"] == HAZARD_MEMBERSHIP_CHANGED:
        return (
            f"  thread {hazard['thread_id']}: membership changed from "
            f"[{hazard['was']}] to [{hazard['now']}]"
        )
    targets = "; ".join(
        f"root {target['root_tweet_id']} / customer {target['customer_author_id']}"
        for target in hazard["now_in"]
    )
    # Plain ASCII: operators read this on Windows consoles that mangle em dashes.
    return (
        f"  thread {hazard['thread_id']}: superseded, its tweets [{hazard['was']}] now "
        f"reconstruct as {targets}, which would be stored as a different thread_id"
    )


def _hazard_error(hazards: list[dict[str, Any]]) -> IngestError:
    detail = "\n".join(_describe_hazard(hazard) for hazard in hazards)
    return IngestError(
        f"Refusing to ingest: it would change {len(hazards)} gold-labeled thread(s) (R11). "
        "Gold labels key on thread_id, so this would silently repoint frozen labels at "
        f"content nobody labeled:\n{detail}\n"
        "Either re-ingest into a fresh store and re-map the gold labels to the new thread "
        "ids, or (if the change is intended) re-run the ingest with "
        "accept_labeled_thread_changes=True, which completes it and lists the affected "
        "thread ids under 'labeled_thread_hazards' in the ingest summary."
    )


def ingest_tweets(
    conn: sqlite3.Connection,
    tweets: dict[int, Tweet],
    *,
    accept_labeled_thread_changes: bool = False,
) -> dict[str, Any]:
    """Write tweets and their reconstructed threads; return summary counts.

    Refuses the write when it would change a gold-labeled thread (R11) unless
    ``accept_labeled_thread_changes`` is set, in which case the changes are
    reported in the summary under ``labeled_thread_hazards``.
    """
    threads = reconstruct_all(tweets)
    # The guard runs before any write so a refusal leaves the store untouched.
    hazards = _labeled_thread_hazards(conn, threads)
    if hazards and not accept_labeled_thread_changes:
        raise _hazard_error(hazards)
    conn.executemany(
        "INSERT OR REPLACE INTO tweets VALUES (?, ?, ?, ?, ?, ?)",
        (
            (
                t.tweet_id,
                t.author_id,
                int(t.inbound),
                t.created_at,
                t.text,
                t.in_response_to_tweet_id,
            )
            for t in tweets.values()
        ),
    )
    try:
        for thread in threads:
            thread_id = _upsert_thread(conn, thread)
            # Membership is rewritten wholesale so a re-ingest cannot accumulate
            # duplicate or stale positions for a thread that changed shape.
            conn.execute("DELETE FROM thread_tweets WHERE thread_id = ?", (thread_id,))
            conn.executemany(
                "INSERT INTO thread_tweets (thread_id, position, tweet_id) VALUES (?, ?, ?)",
                ((thread_id, pos, tid) for pos, tid in enumerate(thread.tweet_ids)),
            )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise IngestError(f"Could not write reconstructed threads: {exc}") from exc
    return {
        "tweets": len(tweets),
        "threads": len(threads),
        "truncated": sum(1 for t in threads if t.truncated),
        "cycle_flagged": sum(1 for t in threads if t.cycle_flagged),
        "labeled_thread_hazards": hazards,
    }


def ingest_csv(
    csv_path: Path | str,
    db_path: Path | str,
    *,
    accept_labeled_thread_changes: bool = False,
) -> dict[str, Any]:
    """End-to-end ingest: CSV -> reconstructed threads -> SQLite store."""
    tweets = load_tweets_csv(csv_path)
    conn = open_store(db_path)
    try:
        return ingest_tweets(
            conn, tweets, accept_labeled_thread_changes=accept_labeled_thread_changes
        )
    finally:
        conn.close()


def get_thread(conn: sqlite3.Connection, thread_id: int) -> Thread:
    """Rebuild a Thread (tweets in stored chronological order) from the store."""
    row = conn.execute(
        "SELECT root_tweet_id, customer_author_id, truncated, cycle_flagged "
        "FROM threads WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        raise IngestError(f"No thread with id {thread_id} in the store.")
    tweet_rows = conn.execute(
        "SELECT t.* FROM thread_tweets tt JOIN tweets t ON t.tweet_id = tt.tweet_id "
        "WHERE tt.thread_id = ? ORDER BY tt.position",
        (thread_id,),
    ).fetchall()
    return Thread(
        root_tweet_id=row["root_tweet_id"],
        customer_author_id=row["customer_author_id"],
        tweets=tuple(
            Tweet(
                tweet_id=r["tweet_id"],
                author_id=r["author_id"],
                inbound=bool(r["inbound"]),
                created_at=r["created_at"],
                text=r["text"],
                in_response_to_tweet_id=r["in_response_to_tweet_id"],
            )
            for r in tweet_rows
        ),
        truncated=bool(row["truncated"]),
        cycle_flagged=bool(row["cycle_flagged"]),
    )


def thread_ids_for_tweet(conn: sqlite3.Connection, tweet_id: int) -> list[int]:
    """Ids of every thread containing the tweet (branches may share a prefix)."""
    rows = conn.execute(
        "SELECT DISTINCT thread_id FROM thread_tweets WHERE tweet_id = ? ORDER BY thread_id",
        (tweet_id,),
    ).fetchall()
    return [r["thread_id"] for r in rows]


def get_distribution_mode(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = 'distribution_mode'").fetchone()
    return row["value"] if row is not None else DIST_MODE_PENDING


def set_distribution_mode(conn: sqlite3.Connection, mode: str) -> None:
    """Record the human license verdict (R2/R29). Never called automatically."""
    if mode not in DISTRIBUTION_MODES:
        raise IngestError(
            f"Unknown distribution mode {mode!r}; expected one of: {DISTRIBUTION_MODES}"
        )
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('distribution_mode', ?)", (mode,)
    )
    conn.commit()


def add_eval_item(conn: sqlite3.Connection, thread_id: int, label: str) -> None:
    """Attach an eval-set label to a stored thread."""
    exists = conn.execute(
        "SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    if exists is None:
        raise IngestError(f"Cannot label thread {thread_id}: no such thread in the store.")
    conn.execute(
        "INSERT OR REPLACE INTO eval_items (thread_id, label) VALUES (?, ?)",
        (thread_id, label),
    )
    conn.commit()


def export_eval_set(conn: sqlite3.Connection, mode: str) -> dict[str, Any]:
    """Export the labeled eval set in one of the two distribution modes (R29).

    - ``text``: labeled threads with their full tweet text (repo-shippable if the
      license verdict allows redistribution).
    - ``id``: tweet-id manifest plus labels only; recipients rebuild the text
      locally from the Kaggle download (``rebuild_command``).

    ``mode`` must be chosen explicitly; "pending" is not exportable because the
    mode decision is the manual R2 license verdict, not a default.
    """
    if mode not in EXPORT_MODES:
        raise IngestError(
            f"Cannot export eval set in mode {mode!r}; expected one of {EXPORT_MODES}. "
            "The mode is selected by the manual dataset-license verdict (R2) — record "
            "it with set_distribution_mode() first."
        )
    items: list[dict[str, Any]] = []
    rows = conn.execute("SELECT thread_id, label FROM eval_items ORDER BY thread_id").fetchall()
    for row in rows:
        thread = get_thread(conn, row["thread_id"])
        item: dict[str, Any] = {
            "thread_id": row["thread_id"],
            "label": row["label"],
            "root_tweet_id": thread.root_tweet_id,
        }
        if mode == DIST_MODE_TEXT:
            item["tweets"] = [
                {
                    "tweet_id": t.tweet_id,
                    "author_id": t.author_id,
                    "inbound": t.inbound,
                    "created_at": t.created_at,
                    "text": t.text,
                }
                for t in thread.tweets
            ]
        else:
            item["tweet_ids"] = list(thread.tweet_ids)
        items.append(item)
    export: dict[str, Any] = {
        "mode": mode,
        "dataset": DATASET_HANDLE,
        "items": items,
    }
    if mode == DIST_MODE_ID:
        export["rebuild_command"] = REBUILD_COMMAND
    return export
