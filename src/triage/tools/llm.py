"""Schema-enforced Anthropic call wrapper — the one LLM entry point (KTD4, R27).

Every tool step calls the model through :func:`call_with_schema`:

- raw ``anthropic`` SDK, ``client.messages.parse`` with a Pydantic
  ``output_format`` (no langchain);
- two retries per call on parse/validation failure or refusal;
- after that, a typed :class:`OutputFailure` result — never an exception for
  bad model output (R27).

The client is injectable so tests (and any surface that wants to manage its
own client) never touch the network here.
"""

from __future__ import annotations

import functools
from typing import Any

import anthropic
from pydantic import ValidationError

from triage.config import get_api_key
from triage.tools.schemas import (
    FAILURE_MALFORMED,
    FAILURE_OFF_TAXONOMY,
    FAILURE_REFUSAL,
    OutputFailure,
)

# R27: two schema-enforced retries per call, after the initial attempt.
MAX_RETRIES_PER_CALL = 2
DEFAULT_MAX_TOKENS = 1024
_DETAIL_LIMIT = 500

# Pydantic error types indicating the text was not valid JSON at all.
_MALFORMED_ERROR_TYPES = frozenset({"json_invalid", "json_type"})
# Pydantic error types indicating valid JSON carrying a label outside the taxonomy.
_OFF_TAXONOMY_ERROR_TYPES = frozenset({"literal_error", "enum"})


@functools.cache
def make_client() -> anthropic.Anthropic:
    """The default client from the environment key (never in tests).

    Cached so one process reuses a single client (and its connection pool)
    across steps and batch entries, while staying lazy: a run that never
    reaches an LLM call (e.g. all-degenerate threads) never needs the key.
    """
    return anthropic.Anthropic(api_key=get_api_key())


def _failure_kind(exc: ValidationError) -> str:
    error_types = {err.get("type") for err in exc.errors()}
    if error_types & _MALFORMED_ERROR_TYPES:
        return FAILURE_MALFORMED
    if error_types & _OFF_TAXONOMY_ERROR_TYPES:
        return FAILURE_OFF_TAXONOMY
    # Valid JSON with missing/mistyped fields is still a malformed output.
    return FAILURE_MALFORMED


def call_with_schema(
    step: str,
    *,
    model: str,
    system: str,
    user_text: str,
    schema: type,
    client: Any = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Any:
    """Call the model once (plus up to two retries), enforcing ``schema``.

    Returns a validated ``schema`` instance, or :class:`OutputFailure` when
    every attempt produced malformed, off-taxonomy, or refused output (R27).
    """
    if client is None:
        client = make_client()
    attempts = 0
    kind = FAILURE_MALFORMED
    detail = "no attempt made"
    while attempts < 1 + MAX_RETRIES_PER_CALL:
        attempts += 1
        try:
            message = client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_text}],
                output_format=schema,
            )
        except ValidationError as exc:
            kind = _failure_kind(exc)
            detail = f"attempt {attempts}: {exc}"[:_DETAIL_LIMIT]
            continue
        parsed = message.parsed_output
        if parsed is None:
            # No parseable text content — e.g. stop_reason 'refusal' or an
            # empty response.
            kind = FAILURE_REFUSAL
            stop_reason = getattr(message, "stop_reason", None)
            detail = f"attempt {attempts}: no parsed output (stop_reason={stop_reason!r})"
            continue
        return parsed
    return OutputFailure(step=step, kind=kind, attempts=attempts, detail=detail)
