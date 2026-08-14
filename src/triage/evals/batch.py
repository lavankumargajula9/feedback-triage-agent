"""Wave-based execution of measured runs via the Messages Batch API (KTD5).

The Batch API bills at 50% of list price. Nothing about the per-thread
execution code changes: threads run against :class:`BatchCollectingClient`,
which replays the shared call cache exactly like the sync caching client but
records every miss instead of calling the API, aborting that caller with
:class:`PendingBatchCall`. :func:`drive_waves` then submits all recorded
requests as one batch, validates each result against its schema, writes
successes into the same cache, and re-executes — completed calls replay for
free, so each dependent pipeline step becomes the next wave.

Failure semantics mirror the sync path (R27): a result the schema rejects —
or a refusal — is resubmitted up to the same three total attempts, after which
the collector replays the recorded failure (the last bad payload's true
ValidationError, or a no-output message for refusals) so ``call_with_schema``
returns the same typed OutputFailure kinds. Transport failures
(errored/canceled/expired batch entries) raise, as they do in sync mode,
leaving the cache resumable.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from triage.evals.cache import (
    BACKOFF_SECONDS,
    RETRYABLE_STATUS,
    CallCache,
    _CachedMessage,
    call_cost,
    empty_usage,
    replay_message,
    request_identity,
)
from triage.tools.llm import MAX_RETRIES_PER_CALL

# Batch API bills at half of list price.
BATCH_DISCOUNT = 0.5
# Same total-attempt budget as one sync call_with_schema invocation (R27).
MAX_ATTEMPTS = 1 + MAX_RETRIES_PER_CALL
# Transport retries per entry before the run raises, matching the sync
# backoff's bounded retries.
MAX_TRANSPORT_FAILURES = 3
# Upper bound on collect/submit rounds: four dependent pipeline steps plus the
# independent calls, each with up to MAX_ATTEMPTS submissions, fits well under
# this; exceeding it means the run is not converging.
MAX_WAVES = 20
DEFAULT_POLL_SECONDS = 10.0
# The Batch API completes within 24h; past this the batch is presumed stuck.
MAX_POLL_SECONDS = 26 * 3600


class BatchError(Exception):
    """A batch run that cannot proceed (transport failures, no convergence)."""


class PendingBatchCall(Exception):
    """Raised by the collector on a cache miss: the call joins the next wave."""

    def __init__(self, key: str) -> None:
        super().__init__(f"call {key} pending batch submission")
        self.key = key


@dataclass
class WaveLedger:
    """Cross-wave state: what is pending, what failed, and what was spent."""

    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    owners: dict[str, Any] = field(default_factory=dict)
    attempts: Counter = field(default_factory=Counter)
    # key -> ("payload", rejected_text) | ("refusal", stop_reason)
    bad_results: dict[str, tuple[str, str]] = field(default_factory=dict)
    transport_failures: Counter = field(default_factory=Counter)
    spent: dict[Any, dict[str, float]] = field(default_factory=dict)

    def spent_for(self, owner: Any) -> dict[str, float]:
        return dict(self.spent.get(owner, empty_usage()))


class _CollectingMessages:
    """The ``client.messages`` facade: cache replay, else collect-and-raise."""

    def __init__(self, cache: CallCache, ledger: WaveLedger, run_id: str, client: Any) -> None:
        self._cache = cache
        self._ledger = ledger
        self._run_id = run_id
        self._client = client

    def parse(self, **kwargs: Any) -> Any:
        key, _, schema, _ = request_identity(kwargs, self._run_id)
        payload = self._cache.get(key)
        if payload is not None:
            try:
                return replay_message(payload, schema)
            except ValidationError:
                self._cache.delete(key)
        ledger = self._ledger
        if ledger.attempts[key] >= MAX_ATTEMPTS and key in ledger.bad_results:
            kind, value = ledger.bad_results[key]
            if kind == "refusal":
                return _CachedMessage(parsed_output=None, stop_reason=value)
            # Replay the last rejected text so call_with_schema raises the
            # true ValidationError and reports the true failure kind (R27).
            return replay_message(value, schema)
        ledger.pending[key] = dict(kwargs)
        ledger.owners.setdefault(key, self._client.owner)
        raise PendingBatchCall(key)


class BatchCollectingClient:
    """Anthropic-client-shaped collector both eval arms can run against.

    ``owner`` is set by the wave driver before each unit of work so batch
    spend can be attributed back to it (e.g. ``(thread_id, "pipeline")``).
    """

    def __init__(self, cache: CallCache, ledger: WaveLedger, run_id: str = "") -> None:
        self.ledger = ledger
        self.owner: Any = None
        self.messages = _CollectingMessages(cache, ledger, run_id, self)


# Constraint keywords structured outputs rejects; ranges stay enforced
# client-side because absorption validates with the pydantic schema.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    }
)
# Mappings whose KEYS are names, not schema keywords; only their values are
# schema nodes.
_NAMED_CHILD_KEYS = frozenset({"properties", "$defs", "definitions"})


def _strict_schema(schema: Any) -> dict[str, Any]:
    """The schema's JSON shape with every object closed and unsupported
    constraint keywords removed, as structured outputs require; the SDK's
    parse() does this transform for sync calls."""
    node = schema.model_json_schema()
    _strictify(node)
    return node


def _strictify(node: Any) -> None:
    if isinstance(node, dict):
        for keyword in _UNSUPPORTED_KEYWORDS & node.keys():
            del node[keyword]
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for key, value in node.items():
            if key in _NAMED_CHILD_KEYS and isinstance(value, dict):
                for child in value.values():
                    _strictify(child)
            else:
                _strictify(value)
    elif isinstance(node, list):
        for value in node:
            _strictify(value)


def _batch_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """messages.parse kwargs as raw batch params: the pydantic ``output_format``
    becomes an ``output_config`` json_schema block."""
    params = {k: v for k, v in kwargs.items() if k != "output_format"}
    params["output_config"] = {
        "format": {"type": "json_schema", "schema": _strict_schema(kwargs["output_format"])}
    }
    return params


def _first_text(message: Any) -> str | None:
    """The first text block's text, or None when the message has none."""
    for block in getattr(message, "content", None) or ():
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return None


def _tally(ledger: WaveLedger, key: str, model: str, message: Any) -> None:
    usage = getattr(message, "usage", None)
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    spent = ledger.spent.setdefault(ledger.owners.get(key), empty_usage())
    spent["calls"] += 1
    cost = call_cost(model, tokens_in, tokens_out)
    if cost is None:
        spent["unpriced_calls"] += 1
    else:
        spent["cost"] += cost * BATCH_DISCOUNT
    spent["tokens_in"] += tokens_in
    spent["tokens_out"] += tokens_out


def _absorb_success(
    ledger: WaveLedger, cache: CallCache, run_id: str, key: str, message: Any
) -> None:
    kwargs = ledger.pending.get(key)
    if kwargs is None:
        return
    _, model, schema, schema_name = request_identity(kwargs, run_id)
    _tally(ledger, key, model, message)
    text = _first_text(message)
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "refusal" or text is None:
        ledger.attempts[key] += 1
        ledger.bad_results[key] = ("refusal", str(stop_reason))
        return
    try:
        parsed = schema.model_validate_json(text)
    except ValidationError:
        ledger.attempts[key] += 1
        ledger.bad_results[key] = ("payload", text)
        return
    cache.put(
        key, model=model, schema_name=schema_name, run_id=run_id, payload=parsed.model_dump_json()
    )
    ledger.bad_results.pop(key, None)


def _register_transport_failure(ledger: WaveLedger, key: str, detail: str) -> None:
    ledger.transport_failures[key] += 1
    if ledger.transport_failures[key] >= MAX_TRANSPORT_FAILURES:
        raise BatchError(
            f"batch entry {key} failed {ledger.transport_failures[key]} times "
            f"(last: {detail}); completed calls are cached — re-run to resume"
        )


def batch_client_factory(batch_client: Any) -> Callable[[], Any]:
    """A factory for the injected client, else the lazily built real one (R30)."""

    def factory() -> Any:
        if batch_client is not None:
            return batch_client
        from triage.tools.llm import make_client

        return make_client()

    return factory


def _retrieve_with_backoff(api: Any, batch_id: str) -> Any:
    """Serial retry on transient (429/5xx) errors; everything else raises."""
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            return api.messages.batches.retrieve(batch_id)
        except Exception as exc:
            retryable = getattr(exc, "status_code", None) in RETRYABLE_STATUS
            if not retryable or attempt >= len(BACKOFF_SECONDS):
                raise
            time.sleep(BACKOFF_SECONDS[attempt])
    raise AssertionError("unreachable")  # pragma: no cover


def _open_batch(ledger: WaveLedger, *, cache: CallCache, run_id: str, api: Any) -> tuple[str, Any]:
    """(batch_id, initial status): re-attach to the recorded in-flight batch —
    already paid for — when its keys are all still pending, else create."""
    inflight = cache.get_inflight(run_id)
    if inflight is not None:
        batch_id, keys = inflight
        if set(keys) <= set(ledger.pending):
            return batch_id, _retrieve_with_backoff(api, batch_id)
        cache.clear_inflight(run_id)
    requests = [
        {"custom_id": key, "params": _batch_params(kwargs)}
        for key, kwargs in ledger.pending.items()
    ]
    batch = api.messages.batches.create(requests=requests)
    cache.record_inflight(run_id, batch.id, list(ledger.pending))
    return batch.id, batch


def submit_wave(
    ledger: WaveLedger,
    *,
    cache: CallCache,
    run_id: str,
    client_factory: Callable[[], Any],
    poll_seconds: float,
    max_poll_seconds: float = MAX_POLL_SECONDS,
) -> None:
    """Submit every pending request as ONE batch and absorb its results."""
    api = client_factory()
    batch_id, status = _open_batch(ledger, cache=cache, run_id=run_id, api=api)
    started = time.monotonic()
    while getattr(status, "processing_status", "ended") != "ended":
        if time.monotonic() - started >= max_poll_seconds:
            raise BatchError(
                f"batch {batch_id} still processing after {max_poll_seconds:.0f}s; "
                "it stays recorded — re-run to re-attach without paying again"
            )
        time.sleep(poll_seconds)
        status = _retrieve_with_backoff(api, batch_id)
    seen: set[str] = set()
    failures: list[tuple[str, str]] = []
    for result in api.messages.batches.results(batch_id):
        seen.add(result.custom_id)
        outcome = result.result
        if getattr(outcome, "type", None) == "succeeded":
            _absorb_success(ledger, cache, run_id, result.custom_id, outcome.message)
        else:
            failures.append((result.custom_id, str(getattr(outcome, "type", "unknown"))))
    for key in set(ledger.pending) - seen:
        failures.append((key, "missing from batch results"))
    ledger.pending.clear()
    cache.clear_inflight(run_id)
    # Raised only after every success in the wave is cached, so a transport
    # failure never discards paid-for answers.
    for key, detail in failures:
        _register_transport_failure(ledger, key, detail)


def drive_waves(
    pass_fn: Callable[[BatchCollectingClient], bool],
    *,
    cache: CallCache,
    run_id: str,
    batch_client_factory: Callable[[], Any],
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_poll_seconds: float = MAX_POLL_SECONDS,
) -> WaveLedger:
    """Run ``pass_fn`` against fresh collectors until it reports completion.

    ``pass_fn`` executes the whole workload against the given collector,
    checkpointing every item that completes, and returns True when nothing is
    left. The batch client is built lazily: a fully cached run never
    constructs it (R30).
    """
    ledger = WaveLedger()
    for _ in range(MAX_WAVES):
        collector = BatchCollectingClient(cache, ledger, run_id=run_id)
        if pass_fn(collector):
            return ledger
        if not ledger.pending:
            raise BatchError(
                "work remains but no batchable calls were collected; "
                "the run cannot make progress"
            )
        submit_wave(
            ledger,
            cache=cache,
            run_id=run_id,
            client_factory=batch_client_factory,
            poll_seconds=poll_seconds,
            max_poll_seconds=max_poll_seconds,
        )
    raise BatchError(f"batch execution did not converge within {MAX_WAVES} waves")
