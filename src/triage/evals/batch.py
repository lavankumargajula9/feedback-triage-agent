"""Wave-based execution of measured runs via the Messages Batch API (KTD5).

The Batch API bills at 50% of list price. Nothing about the per-thread
execution code changes: threads run against :class:`BatchCollectingClient`,
which replays the shared call cache exactly like the sync caching client but
records every miss instead of calling the API, aborting that caller with
:class:`PendingBatchCall`. :func:`drive_waves` then submits all recorded
requests as one batch, validates each result against its schema, writes
successes into the same cache, and re-executes — completed calls replay for
free, so each dependent pipeline step becomes the next wave.

Failure semantics mirror the sync path (R27): a result the schema rejects is
resubmitted up to the same three total attempts, after which the collector
replays the last bad payload so ``call_with_schema`` raises the same
ValidationError kinds and returns the same typed OutputFailure. Transport
failures (errored/canceled/expired batch entries) raise, as they do in sync
mode, leaving the cache resumable.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from triage.evals.cache import (
    CallCache,
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
    bad_payloads: dict[str, str] = field(default_factory=dict)
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
        key, model, schema, schema_name = request_identity(kwargs, self._run_id)
        payload = self._cache.get(key)
        if payload is not None:
            try:
                return replay_message(payload, schema)
            except ValidationError:
                self._cache.delete(key)
        ledger = self._ledger
        if ledger.attempts[key] >= MAX_ATTEMPTS and key in ledger.bad_payloads:
            # Replay the last rejected text so call_with_schema raises the
            # true ValidationError and reports the true failure kind (R27).
            message = replay_message(ledger.bad_payloads[key], schema)
            # The schema accepts it now (schema changed since the batch ran):
            # promote it to a normal cached success.
            self._cache.put(
                key,
                model=model,
                schema_name=schema_name,
                run_id=self._run_id,
                payload=message.parsed_output.model_dump_json(),
            )
            del ledger.bad_payloads[key]
            return message
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


def _strict_schema(schema: Any) -> dict[str, Any]:
    """The schema's JSON shape with every object closed, as structured
    outputs require; the SDK's parse() does this transform for sync calls."""
    node = schema.model_json_schema()
    _close_objects(node)
    return node


def _close_objects(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for value in node.values():
            _close_objects(value)
    elif isinstance(node, list):
        for value in node:
            _close_objects(value)


def _batch_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """messages.parse kwargs as raw batch params: the pydantic ``output_format``
    becomes an ``output_config`` json_schema block."""
    params = {k: v for k, v in kwargs.items() if k != "output_format"}
    params["output_config"] = {
        "format": {"type": "json_schema", "schema": _strict_schema(kwargs["output_format"])}
    }
    return params


def _first_text(message: Any) -> str:
    for block in getattr(message, "content", None) or ():
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return ""


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
    try:
        parsed = schema.model_validate_json(text)
    except ValidationError:
        ledger.attempts[key] += 1
        ledger.bad_payloads[key] = text
        return
    cache.put(
        key, model=model, schema_name=schema_name, run_id=run_id, payload=parsed.model_dump_json()
    )
    ledger.bad_payloads.pop(key, None)


def _register_transport_failure(ledger: WaveLedger, key: str, detail: str) -> None:
    ledger.transport_failures[key] += 1
    if ledger.transport_failures[key] >= MAX_TRANSPORT_FAILURES:
        raise BatchError(
            f"batch entry {key} failed {ledger.transport_failures[key]} times "
            f"(last: {detail}); completed calls are cached — re-run to resume"
        )


def submit_wave(
    ledger: WaveLedger,
    *,
    cache: CallCache,
    run_id: str,
    client_factory: Callable[[], Any],
    poll_seconds: float,
) -> None:
    """Submit every pending request as ONE batch and absorb its results."""
    requests = [
        {"custom_id": key, "params": _batch_params(kwargs)}
        for key, kwargs in ledger.pending.items()
    ]
    api = client_factory()
    batch = api.messages.batches.create(requests=requests)
    status = batch
    while getattr(status, "processing_status", "ended") != "ended":
        time.sleep(poll_seconds)
        status = api.messages.batches.retrieve(batch.id)
    seen: set[str] = set()
    failures: list[tuple[str, str]] = []
    for result in api.messages.batches.results(batch.id):
        seen.add(result.custom_id)
        outcome = result.result
        if getattr(outcome, "type", None) == "succeeded":
            _absorb_success(ledger, cache, run_id, result.custom_id, outcome.message)
        else:
            failures.append((result.custom_id, str(getattr(outcome, "type", "unknown"))))
    for key in set(ledger.pending) - seen:
        failures.append((key, "missing from batch results"))
    ledger.pending.clear()
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
        )
    raise BatchError(f"batch execution did not converge within {MAX_WAVES} waves")
