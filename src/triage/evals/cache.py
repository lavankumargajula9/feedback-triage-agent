"""SQLite LLM call cache — cache-first eval execution (KTD5, R30, R32).

A single-file disk cache keyed by SHA-256 of (model, params, full prompt
[system + user], schema name and schema shape); a run-id salt participates in
the key so variance passes bypass prior entries (KTD5). It integrates at the
LLM-wrapper seam: :class:`CachingClient` is anthropic-client-shaped, so
:func:`triage.tools.llm.call_with_schema` — and therefore BOTH eval arms, the
pipeline steps and the single-prompt baseline — goes through the same cache
unchanged.

The output schema's JSON shape is part of the key, and a stored payload that no
longer decodes is treated as a MISS: a schema edit therefore costs one honest
call instead of replaying a stale payload that the wrapper would retry three
times and score as a model failure (R27).

Only successfully parsed outputs are stored (as their JSON payload); raised
errors, malformed outputs, and refusals are never cached, so a later retry can
still succeed. Transient API errors (429/5xx) get a short serial backoff —
deliberately simple, no concurrency machinery (per the plan).

Every call — live or replayed — advances a usage tally (calls, tokens, dollars,
wall-clock seconds) that the runner snapshots per arm (KTD6). Replayed calls
add zeros to the money fields, so reported totals describe money actually
spent.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# Transient API statuses worth a serial retry: rate limit and server errors.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 529})
# One sleep per retry; len(BACKOFF_SECONDS) == number of transport retries.
BACKOFF_SECONDS = (1.0, 2.0)

# (input $/MTok, output $/MTok) per model id used by any role (R30, R22).
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Per-call tally fields; ``unpriced_calls`` keeps a model with no price on file
# from silently reporting as free.
USAGE_FIELDS = (
    "calls",
    "cached_calls",
    "unpriced_calls",
    "tokens_in",
    "tokens_out",
    "cost",
    "latency_s",
)


def call_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """Dollars for one call's measured tokens, or None with no price on file."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (tokens_in * in_price + tokens_out * out_price) / 1_000_000


def empty_usage() -> dict[str, float]:
    """A zeroed usage tally."""
    return dict.fromkeys(USAGE_FIELDS, 0)


def usage_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """The tally advanced between two snapshots — one arm's share of a run."""
    return {field: after[field] - before[field] for field in USAGE_FIELDS}

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
    schema_digest: str = "",
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
            "schema_digest": schema_digest,
            "run_id": run_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def schema_digest(schema: Any) -> str:
    """SHA-256 of the output schema's JSON shape.

    Part of the cache key: a schema-only edit under an unchanged class name must
    not replay payloads the new schema rejects.
    """
    return hashlib.sha256(
        json.dumps(schema.model_json_schema(), sort_keys=True).encode("utf-8")
    ).hexdigest()


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

    def delete(self, key: str) -> None:
        """Drop one entry; committed immediately."""
        self._conn.execute("DELETE FROM calls WHERE key = ?", (key,))
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
        self.usage = empty_usage()

    def usage_snapshot(self) -> dict[str, float]:
        return dict(self.usage)

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
            schema_digest=schema_digest(schema),
            params=params,
            run_id=self._run_id,
        )
        payload = self._cache.get(key)
        if payload is not None:
            replayed = self._replay(key, payload, schema)
            if replayed is not None:
                self.hits += 1
                self._record_usage(model=model, cached=True)
                return replayed
        self.misses += 1
        started = time.perf_counter()
        message = self._parse_with_backoff(kwargs)
        latency_s = time.perf_counter() - started
        usage = getattr(message, "usage", None)
        self._record_usage(
            model=model,
            tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
            latency_s=latency_s,
        )
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

    def _replay(self, key: str, payload: str, schema: Any) -> Any | None:
        """The stored payload as a message, or None when it no longer decodes.

        A payload the schema now rejects is dropped and answered live: left in
        place it would surface as a ValidationError inside the wrapper and be
        scored as a model failure (R27) on every retry.
        """
        try:
            return _CachedMessage(parsed_output=schema.model_validate_json(payload))
        except ValidationError:
            self._cache.delete(key)
            return None

    def _record_usage(
        self,
        *,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_s: float = 0.0,
        cached: bool = False,
    ) -> None:
        """Tally one call (KTD6); a replayed call adds zeros to the money fields."""
        self.usage["calls"] += 1
        if cached:
            self.usage["cached_calls"] += 1
            return
        cost = call_cost(model, tokens_in, tokens_out)
        if cost is None:
            self.usage["unpriced_calls"] += 1
        self.usage["tokens_in"] += tokens_in
        self.usage["tokens_out"] += tokens_out
        self.usage["cost"] += cost or 0.0
        self.usage["latency_s"] += latency_s

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

    def usage_snapshot(self) -> dict[str, float]:
        """The run's cumulative usage tally so far (KTD6)."""
        return self.messages.usage_snapshot()
