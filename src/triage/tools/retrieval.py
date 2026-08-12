"""Read-only thread retrieval, rendering, and degeneracy detection (R10, R28).

Retrieval helpers (list/search/get) sit over the SQLite store for the MCP
surface; they never write. ``render_thread`` is the one thread-to-prompt-text
rendering every surface uses, so the pipeline and the baseline see identical
thread text (R8).

``degenerate_reason`` is the pure predicate behind R28: threads with no
diagnosable content (single-tweet stubs, empty-after-cleaning, deflect-to-DM)
are detected BEFORE any LLM call, so the steps can short-circuit to the
General Inquiry + escalated result without burning retries.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from triage.ingest.reconstruct import Thread
from triage.ingest.store import get_thread

_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+|\bpic\.twitter\.com/\S+")
_WORD_RE = re.compile(r"[a-z0-9']+")

# A thread whose cleaned customer text has fewer content-bearing words than
# this is undiagnosable (R28) — e.g. a bare "@brand" or "help pls".
MIN_DIAGNOSABLE_WORDS = 3

# Words that signal (or pad out) a move-to-DM message rather than describing
# a problem. If nothing else remains, the public thread is undiagnosable.
_DM_WORDS = frozenset(
    {"dm", "dms", "dmed", "dm'd", "pm", "pms", "direct", "message", "messaged", "messages",
     "inbox"}
)
_FILLER_WORDS = frozenset(
    {"i", "i've", "we", "we've", "you", "your", "the", "a", "an", "my", "our", "to", "us",
     "have", "has", "had", "just", "sent", "send", "sending", "will", "ok", "okay", "sure",
     "done", "thanks", "thank", "hi", "hello", "hey", "please", "check", "checking", "now",
     "it", "them", "there", "here"}
)


def clean_text(text: str) -> str:
    """Strip @mentions and links, collapse whitespace."""
    text = _MENTION_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return " ".join(text.split())


def degenerate_reason(thread: Thread) -> str | None:
    """Why this thread is undiagnosable (R28), or None if it is triageable.

    Pure function of the thread; performs no I/O and no LLM call.
    """
    parts = [clean_text(t.text) for t in thread.tweets if t.inbound]
    combined = " ".join(p for p in parts if p).strip()
    if not combined:
        return "no customer text remains after cleaning (mentions and links stripped)"
    words = [w for w in _WORD_RE.findall(combined.lower()) if any(c.isalpha() for c in w)]
    if len(words) < MIN_DIAGNOSABLE_WORDS:
        return (
            f"customer text is too short to diagnose "
            f"({len(words)} content word(s) after cleaning)"
        )
    meaningful = [w for w in words if w not in _DM_WORDS and w not in _FILLER_WORDS]
    if not meaningful and any(w in _DM_WORDS for w in words):
        return "conversation deflected to DM; no diagnosable content in the public thread"
    return None


def render_thread(thread: Thread) -> str:
    """Render a thread as the prompt text shown to the model (all surfaces)."""
    lines = []
    if thread.truncated:
        lines.append("(note: earlier tweets in this thread are missing from the dataset)")
    for tweet in thread.tweets:
        role = "Customer" if tweet.inbound else "Support"
        lines.append(f"[{tweet.created_at}] {role} ({tweet.author_id}): {tweet.text}")
    return "\n".join(lines)


def get_thread_by_id(conn: sqlite3.Connection, thread_id: int) -> Thread:
    """Fetch one stored thread (delegates to the store; read-only)."""
    return get_thread(conn, thread_id)


def list_threads(
    conn: sqlite3.Connection, limit: int = 20, offset: int = 0
) -> list[dict[str, Any]]:
    """List stored threads (id, root, customer, size) in stable id order."""
    rows = conn.execute(
        "SELECT th.thread_id, th.root_tweet_id, th.customer_author_id, "
        "       COUNT(tt.tweet_id) AS tweet_count "
        "FROM threads th JOIN thread_tweets tt ON tt.thread_id = th.thread_id "
        "GROUP BY th.thread_id ORDER BY th.thread_id LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [dict(row) for row in rows]


def search_threads(
    conn: sqlite3.Connection, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    """Find threads whose tweet text contains ``query`` (case-insensitive)."""
    rows = conn.execute(
        "SELECT DISTINCT tt.thread_id, t.tweet_id, t.text "
        "FROM tweets t JOIN thread_tweets tt ON tt.tweet_id = t.tweet_id "
        "WHERE t.text LIKE ? ORDER BY tt.thread_id LIMIT ?",
        (f"%{query}%", limit),
    ).fetchall()
    return [dict(row) for row in rows]
