"""SQLite LLM call cache — cache-first eval execution (KTD5, R30, R32).

A single-file disk cache keyed by SHA-256 of (model, params, full prompt
[system + user], schema name); a run-id salt participates in the key so
variance passes bypass prior entries (KTD5). It integrates at the LLM-wrapper
seam: :class:`CachingClient` is anthropic-client-shaped, so
:func:`triage.tools.llm.call_with_schema` — and therefore BOTH eval arms, the
pipeline steps and the single-prompt baseline — goes through the same cache
unchanged.

Only successfully parsed outputs are stored (as their JSON payload); raised
errors, malformed outputs, and refusals are never cached, so a later retry can
still succeed. Transient API errors (429/5xx) get a short serial backoff —
deliberately simple, no concurrency machinery (per the plan).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Transient API statuses worth a serial retry: rate limit and server errors.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 529})
# One sleep per retry; len(BACKOFF_SECONDS) == number of transport retries.
BACKOFF_SECONDS = (1.0, 2.0)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    schema TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def cache_key(
    *,
    model: str,
    system: str,
    user_text: str,
    schema_name: str,
    params: dict[str, Any] | None = None,
    run_id: str = "",
) -> str:
    """SHA-256 over every determinism-relevant component of one call (KTD5)."""
    material = json.dumps(
        {
            "model": model,
            "params": params or {},
            "system": system,
            "user_text": user_text,
            "schema": schema_name,
            "run_id": run_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class CallCache:
    """Single-file SQLite cache of parsed LLM outputs (KTD5, stdlib sqlite3)."""

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> str | None:
        """The cached JSON payload for ``key``, or None on a miss."""
        row = self._conn.execute("SELECT payload FROM calls WHERE key = ?", (key,)).fetchone()
        return row[0] if row is not None else None

    def put(self, key: str, *, model: str, schema_name: str, run_id: str, payload: str) -> None:
        """Store one successfully parsed output; committed immediately (R32)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO calls (key, model, schema, run_id, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, model, schema_name, run_id, payload),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


@dataclass
class _CachedMessage:
    """The message shape call_with_schema reads, replayed from the cache."""

    parsed_output: Any
    stop_reason: str = "cached"


class _CachingMessages:
    """The ``client.messages`` facade: cache lookup, then the real client."""

    def __init__(self, cache: CallCache, inner_factory: Any, run_id: str) -> None:
        self._cache = cache
        self._inner_factory = inner_factory
        self._run_id = run_id
        self.hits = 0
        self.misses = 0

    def parse(self, **kwargs: Any) -> Any:
        schema = kwargs["output_format"]
        schema_name = f"{schema.__module__}.{schema.__qualname__}"
        model = kwargs["model"]
        params = {
            k: v
            for k, v in kwargs.items()
            if k not in ("model", "system", "messages", "output_format")
        }
        key = cache_key(
            model=model,
            system=kwargs.get("system", ""),
            user_text=json.dumps(kwargs["messages"], sort_keys=True, ensure_ascii=False),
            schema_name=schema_name,
            params=params,
            run_id=self._run_id,
        )
        payload = self._cache.get(key)
        if payload is not None:
            self.hits += 1
            return _CachedMessage(parsed_output=schema.model_validate_json(payload))
        self.misses += 1
        message = self._parse_with_backoff(kwargs)
        parsed = getattr(message, "parsed_output", None)
        if parsed is not None:  # never cache refusals/empty responses
            self._cache.put(
                key,
                model=model,
                schema_name=schema_name,
                run_id=self._run_id,
                payload=parsed.model_dump_json(),
            )
        return message

    def _parse_with_backoff(self, kwargs: dict[str, Any]) -> Any:
        """Serial retry on transient (429/5xx) errors; everything else raises."""
        for attempt in range(len(BACKOFF_SECONDS) + 1):
            try:
                return self._inner_factory().messages.parse(**kwargs)
            except Exception as exc:
                retryable = getattr(exc, "status_code", None) in RETRYABLE_STATUS
                if not retryable or attempt >= len(BACKOFF_SECONDS):
                    raise
                time.sleep(BACKOFF_SECONDS[attempt])
        raise AssertionError("unreachable")  # pragma: no cover


class CachingClient:
    """Anthropic-client-shaped cache wrapper both eval arms call through (KTD5).

    ``inner`` is the real (or fake, in tests) client, constructed lazily from
    the environment key only on the first cache miss — a fully cached run
    never needs credentials (R30).
    """

    def __init__(self, cache: CallCache, inner: Any = None, run_id: str = "") -> None:
        self._inner = inner
        self.messages = _CachingMessages(cache, self._inner_client, run_id)

    def _inner_client(self) -> Any:
        if self._inner is None:
            from triage.tools.llm import make_client

            self._inner = make_client()
        return self._inner

    @property
    def hits(self) -> int:
        return self.messages.hits

    @property
    def misses(self) -> int:
        return self.messages.misses
