"""Tests for the SQLite LLM call cache (U7) — KTD5, R30, R32.

Covers:
- Cache-first behavior (KTD5): two identical requests reach the real client
  once; the second is served from disk.
- Run-id salting: a variance pass bypasses prior entries.
- Persistence: entries survive closing and reopening the cache file.
- Never-cache-failures: raised errors and refusals are not stored, so a later
  retry can succeed.
- Simple serial backoff on transient (429/5xx) API errors — no concurrency
  machinery.
- Key construction: every component (model, params, system, user text, schema
  name, schema shape, run id) participates in the key.
- Schema identity: a schema-only edit under the same class name is a MISS, and
  a stored payload that no longer validates is a miss too — never a scored
  model failure.

Every test is fully mocked: zero network, no anthropic client constructed.
"""

from datetime import date

import pytest
from pydantic import ValidationError, create_model
from test_tools import (
    CATEGORIZED,
    DIAGNOSABLE,
    ExplodingClient,
    FakeClient,
    ok,
    refusal,
    validation_error_for,
)

from triage.evals.cache import CachingClient, CallCache, cache_key, prices_per_mtok
from triage.tools import CategorizeResult, categorize

MODEL = "claude-haiku-4-5"

# One fixed, fully-specified request, shaped like call_with_schema's calls.
REQUEST = {
    "model": MODEL,
    "max_tokens": 1024,
    "system": "system text",
    "messages": [{"role": "user", "content": "user text"}],
    "output_format": CategorizeResult,
}


@pytest.fixture
def cache(tmp_path):
    call_cache = CallCache(tmp_path / "cache.db")
    yield call_cache
    call_cache.close()


def schema_variant(**fields):
    """Two output schemas with the SAME module and qualname but different fields
    — exactly what a schema-only edit (new field, tightened Literal) produces."""
    return create_model("CategorizeResult", __module__="triage.tools.schemas", **fields)


def poison_stored_payloads(cache, payload):
    """Leave every stored payload no longer decodable by its schema."""
    cache._conn.execute("UPDATE calls SET payload = ?", (payload,))
    cache._conn.commit()


class TestCacheHit:
    def test_identical_requests_hit_cache_once(self, cache):
        # KTD5 happy path: the second identical request never reaches the client.
        inner = FakeClient([ok(CATEGORIZED)])
        client = CachingClient(cache, inner=inner)
        first = client.messages.parse(**REQUEST)
        second = client.messages.parse(**REQUEST)
        assert len(inner.messages.calls) == 1
        assert first.parsed_output == CATEGORIZED
        assert second.parsed_output == CATEGORIZED
        assert client.misses == 1
        assert client.hits == 1

    def test_salted_run_id_bypasses_cache(self, cache):
        # KTD5: variance passes salt the key with a run id, bypassing entries.
        inner = FakeClient([ok(CATEGORIZED), ok(CATEGORIZED)])
        CachingClient(cache, inner=inner).messages.parse(**REQUEST)
        salted = CachingClient(cache, inner=inner, run_id="variance-1")
        salted.messages.parse(**REQUEST)
        assert len(inner.messages.calls) == 2
        assert salted.hits == 0

    def test_entries_survive_reopen(self, tmp_path):
        path = tmp_path / "cache.db"
        first = CallCache(path)
        CachingClient(first, inner=FakeClient([ok(CATEGORIZED)])).messages.parse(**REQUEST)
        first.close()
        reopened = CallCache(path)
        try:
            # ExplodingClient: any real call would fail the test.
            client = CachingClient(reopened, inner=ExplodingClient())
            message = client.messages.parse(**REQUEST)
            assert message.parsed_output == CATEGORIZED
            assert client.hits == 1
        finally:
            reopened.close()

    def test_cache_works_through_call_with_schema(self, cache):
        # The wrapper is client-shaped, so tool steps use it unchanged.
        inner = FakeClient([ok(CATEGORIZED)])
        client = CachingClient(cache, inner=inner)
        first = categorize(DIAGNOSABLE, model=MODEL, client=client)
        second = categorize(DIAGNOSABLE, model=MODEL, client=client)
        assert isinstance(first, CategorizeResult)
        assert first == second
        assert len(inner.messages.calls) == 1


class TestNeverCacheFailures:
    def test_validation_error_is_not_cached(self, cache):
        # A malformed output propagates (call_with_schema handles retries) and
        # leaves no entry, so the retry reaches the client.
        inner = FakeClient(
            [validation_error_for(CategorizeResult, "{not json"), ok(CATEGORIZED)]
        )
        client = CachingClient(cache, inner=inner)
        with pytest.raises(ValidationError):
            client.messages.parse(**REQUEST)
        message = client.messages.parse(**REQUEST)
        assert message.parsed_output == CATEGORIZED
        assert len(inner.messages.calls) == 2

    def test_refusal_is_not_cached(self, cache):
        inner = FakeClient([refusal(), ok(CATEGORIZED)])
        client = CachingClient(cache, inner=inner)
        assert client.messages.parse(**REQUEST).parsed_output is None
        assert client.messages.parse(**REQUEST).parsed_output == CATEGORIZED
        assert len(inner.messages.calls) == 2


class TestBackoff:
    def test_transient_status_errors_are_retried_with_backoff(self, cache, monkeypatch):
        sleeps = []
        monkeypatch.setattr("triage.evals.cache.time.sleep", sleeps.append)

        class Overloaded(Exception):
            status_code = 529

        inner = FakeClient([Overloaded(), Overloaded(), ok(CATEGORIZED)])
        message = CachingClient(cache, inner=inner).messages.parse(**REQUEST)
        assert message.parsed_output == CATEGORIZED
        assert len(inner.messages.calls) == 3
        assert len(sleeps) == 2

    def test_non_transient_errors_propagate_immediately(self, cache, monkeypatch):
        monkeypatch.setattr(
            "triage.evals.cache.time.sleep",
            lambda _: pytest.fail("must not back off on non-transient errors"),
        )
        inner = FakeClient([RuntimeError("boom")])
        with pytest.raises(RuntimeError):
            CachingClient(cache, inner=inner).messages.parse(**REQUEST)
        assert len(inner.messages.calls) == 1


class TestSchemaIdentity:
    def test_schema_edit_under_the_same_name_is_a_miss(self, cache):
        # A schema-only edit must not replay a payload the new schema rejects:
        # that payload would fail validation inside parse() and be scored as a
        # model failure instead of costing one honest call.
        old = schema_variant(label=(str, ...))
        new = schema_variant(label=(str, ...), severity=(str, ...))
        inner = FakeClient(
            [
                ok(old(label="Technical/Product")),
                ok(new(label="Technical/Product", severity="high")),
            ]
        )
        client = CachingClient(cache, inner=inner)
        client.messages.parse(**{**REQUEST, "output_format": old})
        message = client.messages.parse(**{**REQUEST, "output_format": new})
        assert len(inner.messages.calls) == 2
        assert message.parsed_output.severity == "high"
        assert client.hits == 0

    def test_stale_payload_is_a_miss_not_an_output_failure(self, cache):
        # An entry already poisoned before the key change: decoding it raises a
        # ValidationError, which the LLM wrapper would otherwise count as bad
        # model output and retry three times against the same stale entry.
        inner = FakeClient([ok(CATEGORIZED), ok(CATEGORIZED)])
        client = CachingClient(cache, inner=inner)
        categorize(DIAGNOSABLE, model=MODEL, client=client)
        poison_stored_payloads(cache, "{}")
        result = categorize(DIAGNOSABLE, model=MODEL, client=client)
        assert isinstance(result, CategorizeResult)
        assert len(inner.messages.calls) == 2
        # The stale entry is replaced, so the next call replays for free again.
        assert categorize(DIAGNOSABLE, model=MODEL, client=client) == result
        assert len(inner.messages.calls) == 2


class TestModelPricing:
    def test_sonnet_5_intro_price_through_cutover(self):
        assert prices_per_mtok("claude-sonnet-5", today=date(2026, 8, 31)) == (2.0, 10.0)

    def test_sonnet_5_list_price_after_cutover(self):
        assert prices_per_mtok("claude-sonnet-5", today=date(2026, 9, 1)) == (3.0, 15.0)

    def test_unknown_model_has_no_price(self):
        assert prices_per_mtok("some-unknown-model") is None


class TestInflightRecords:
    def test_inflight_round_trip_survives_reopen(self, tmp_path):
        path = tmp_path / "cache.db"
        recording = CallCache(path)
        recording.record_inflight("run-1", "batch_abc", ["k1", "k2"])
        recording.close()
        reopened = CallCache(path)
        try:
            assert reopened.get_inflight("run-1") == ("batch_abc", ["k1", "k2"])
            assert reopened.get_inflight("other-run") is None
            reopened.clear_inflight("run-1")
            assert reopened.get_inflight("run-1") is None
        finally:
            reopened.close()


class TestCacheKey:
    def test_every_component_participates_in_the_key(self):
        base = {
            "model": MODEL,
            "system": "s",
            "user_text": "u",
            "schema_name": "CategorizeResult",
            "schema_digest": "d1",
            "params": {"max_tokens": 1024},
            "run_id": "",
        }
        variants = [
            {**base, "model": "claude-opus-5"},
            {**base, "system": "s2"},
            {**base, "user_text": "u2"},
            {**base, "schema_name": "RouteResult"},
            {**base, "schema_digest": "d2"},
            {**base, "params": {"max_tokens": 2048}},
            {**base, "run_id": "variance-1"},
        ]
        keys = {cache_key(**kwargs) for kwargs in [base, *variants]}
        assert len(keys) == len(variants) + 1  # all distinct
        assert cache_key(**base) == cache_key(**base)  # and stable
