"""Tests for the shared tool layer (U3) — R4, R5, R6, R9, R10, R27, R28, KTD2, KTD4, KTD7.

Covers:
- Happy path per step: valid mock responses parse into result schemas with
  labels restricted to the single-sourced taxonomy (R4, R9).
- Retry semantics (R27): persistent malformed / off-taxonomy / refused outputs
  exhaust two retries and return a typed OutputFailure — never an exception.
- Retry-then-success on a later attempt.
- Degenerate threads (R28/AE7): short-circuit to General Inquiry + escalated
  with an insufficient-content reason, with zero LLM calls made.
- Single-sourcing (KTD7): schema label types and prompt fragments agree.
- Integration: categorize -> route -> draft -> escalate chained by explicit
  parameter passing (KTD2) produces a complete result dict.
- Retrieval helpers over a store fixture (read-only, for the MCP surface).

Every test is fully mocked: zero network, no anthropic client constructed.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from triage.ingest.reconstruct import Thread, Tweet
from triage.ingest.store import ingest_csv, open_store
from triage.prompts import fragments
from triage.tools import (
    CategorizeResult,
    DraftResult,
    EscalateResult,
    OutputFailure,
    RouteResult,
    categorize,
    draft,
    escalate,
    route,
)
from triage.tools.retrieval import (
    degenerate_reason,
    get_thread_by_id,
    list_threads,
    render_thread,
    search_threads,
)
from triage.tools.schemas import (
    FAILURE_MALFORMED,
    FAILURE_OFF_TAXONOMY,
    FAILURE_REFUSAL,
    CategoryLabel,
    QueueLabel,
)

MODEL = "claude-haiku-4-5"
FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_tweets.csv"


def make_thread(texts, inbound_flags=None, author="100001"):
    """Build a minimal Thread from customer/support tweet texts."""
    inbound_flags = inbound_flags or [True] * len(texts)
    tweets = tuple(
        Tweet(
            tweet_id=i + 1,
            author_id=author if inbound else "brand",
            inbound=inbound,
            created_at=f"Tue Oct 31 22:0{i}:00 +0000 2017",
            text=text,
            in_response_to_tweet_id=None if i == 0 else i,
        )
        for i, (text, inbound) in enumerate(zip(texts, inbound_flags))
    )
    return Thread(
        root_tweet_id=1,
        customer_author_id=author,
        tweets=tweets,
        truncated=False,
        cycle_flagged=False,
    )


def validation_error_for(schema, text):
    """Produce the same pydantic ValidationError messages.parse would raise."""
    try:
        TypeAdapter(schema).validate_json(text)
    except ValidationError as exc:
        return exc
    raise AssertionError(f"expected {text!r} to fail validation against {schema}")


class FakeMessages:
    """Stands in for anthropic client.messages: scripted parse outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.messages = FakeMessages(outcomes)


class ExplodingClient:
    """A client whose use is an immediate test failure (asserts zero LLM calls)."""

    @property
    def messages(self):
        raise AssertionError("LLM client must not be touched for this thread")


def ok(parsed):
    return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn")


def refusal():
    return SimpleNamespace(parsed_output=None, stop_reason="refusal")


DIAGNOSABLE = make_thread(
    [
        "@sprintcare my data plan stopped working today and I keep getting error 105",
        "@100001 sorry to hear that! Which device are you on?",
        "@sprintcare iPhone 12, started this morning",
    ],
    inbound_flags=[True, False, True],
)

CATEGORIZED = CategorizeResult(label="Technical/Product", rationale="Service outage reported.")
ROUTED = RouteResult(queue="Technical/Product", rationale="Needs the technical team.")
DRAFTED = DraftResult(draft="So sorry about that — can you share your account email?")
ESCALATED = EscalateResult(escalate=False, reason="Routine technical issue, macro-resolvable.")

# One full pipeline's worth of successful step outcomes, in node order.
HAPPY_OUTCOMES = [ok(CATEGORIZED), ok(ROUTED), ok(ESCALATED), ok(DRAFTED)]


class TestHappyPath:
    def test_categorize_parses_and_stays_on_taxonomy(self):
        client = FakeClient([ok(CATEGORIZED)])
        result = categorize(DIAGNOSABLE, model=MODEL, client=client)
        assert isinstance(result, CategorizeResult)
        assert result.label in fragments.CATEGORY_LABELS
        assert result.rationale
        assert len(client.messages.calls) == 1
        call = client.messages.calls[0]
        assert call["model"] == MODEL
        # The thread text and the single-sourced taxonomy reach the model.
        assert "data plan stopped working" in call["messages"][0]["content"]
        assert fragments.taxonomy_block() in call["system"]
        assert call["output_format"] is CategorizeResult

    def test_route_passes_prior_category_explicitly(self):
        client = FakeClient([ok(ROUTED)])
        result = route(DIAGNOSABLE, CATEGORIZED, model=MODEL, client=client)
        assert isinstance(result, RouteResult)
        assert result.queue in fragments.QUEUE_LABELS
        assert "Technical/Product" in client.messages.calls[0]["messages"][0]["content"]

    def test_draft_result_is_never_sent(self):
        client = FakeClient([ok(DRAFTED)])
        result = draft(DIAGNOSABLE, CATEGORIZED, ROUTED, model=MODEL, client=client)
        assert isinstance(result, DraftResult)
        assert result.status == "never_sent"  # R6/R9: explicit never-sent status
        assert result.draft

    def test_escalate_returns_binary_decision_with_reason(self):
        client = FakeClient([ok(ESCALATED)])
        result = escalate(DIAGNOSABLE, CATEGORIZED, ROUTED, model=MODEL, client=client)
        assert isinstance(result, EscalateResult)
        assert result.escalate is False
        assert result.reason


class TestRetrySemantics:
    def test_persistent_malformed_output_becomes_typed_failure(self):
        # R27: 1 initial attempt + 2 schema-enforced retries, then OutputFailure.
        bad = [validation_error_for(CategorizeResult, "{not json") for _ in range(3)]
        client = FakeClient(bad)
        result = categorize(DIAGNOSABLE, model=MODEL, client=client)
        assert isinstance(result, OutputFailure)
        assert result.step == "categorize"
        assert result.kind == FAILURE_MALFORMED
        assert result.attempts == 3
        assert len(client.messages.calls) == 3

    def test_off_taxonomy_label_becomes_typed_failure(self):
        off = '{"label": "Weather", "rationale": "sunny"}'
        bad = [validation_error_for(CategorizeResult, off) for _ in range(3)]
        result = categorize(DIAGNOSABLE, model=MODEL, client=FakeClient(bad))
        assert isinstance(result, OutputFailure)
        assert result.kind == FAILURE_OFF_TAXONOMY

    def test_refusal_becomes_typed_failure(self):
        result = escalate(
            DIAGNOSABLE, CATEGORIZED, ROUTED, model=MODEL, client=FakeClient([refusal()] * 3)
        )
        assert isinstance(result, OutputFailure)
        assert result.step == "escalate"
        assert result.kind == FAILURE_REFUSAL
        assert result.attempts == 3

    def test_retry_then_success_on_second_attempt(self):
        outcomes = [validation_error_for(RouteResult, "{oops"), ok(ROUTED)]
        client = FakeClient(outcomes)
        result = route(DIAGNOSABLE, CATEGORIZED, model=MODEL, client=client)
        assert isinstance(result, RouteResult)
        assert len(client.messages.calls) == 2

    def test_downstream_step_tolerates_upstream_failure(self):
        # KTD2: a typed failure passed as a prior output degrades gracefully.
        failure = OutputFailure(step="categorize", kind="malformed", attempts=3, detail="x")
        client = FakeClient([ok(ROUTED)])
        result = route(DIAGNOSABLE, failure, model=MODEL, client=client)
        assert isinstance(result, RouteResult)


class TestDegenerateThreads:
    def test_single_tweet_no_content_short_circuits(self):
        # R28/AE7: General Inquiry + escalate=True + insufficient-content
        # reason, decided before any LLM call.
        thread = make_thread(["@sprintcare"])
        client = ExplodingClient()
        cat = categorize(thread, model=MODEL, client=client)
        esc = escalate(thread, cat, None, model=MODEL, client=client)
        assert isinstance(cat, CategorizeResult)
        assert cat.label == "General Inquiry"
        assert esc.escalate is True
        assert "insufficient content" in esc.reason.lower()

    def test_degenerate_thread_still_gets_route_and_never_sent_draft(self):
        # R28: every step still produces a valid, in-taxonomy result; R6: a
        # draft exists even here and is never sent.
        thread = make_thread(["@brand help"])
        client = ExplodingClient()
        routed = route(thread, None, model=MODEL, client=client)
        drafted = draft(thread, None, None, model=MODEL, client=client)
        assert routed.queue == "General Inquiry"
        assert "insufficient content" in routed.rationale.lower()
        assert isinstance(drafted, DraftResult)
        assert drafted.status == "never_sent"
        assert drafted.draft

    def test_dm_deflection_is_degenerate(self):
        thread = make_thread(["@brand just sent the DM"])
        assert degenerate_reason(thread) is not None

    def test_diagnosable_thread_is_not_degenerate(self):
        assert degenerate_reason(DIAGNOSABLE) is None

    def test_thread_with_only_links_is_degenerate(self):
        thread = make_thread(["@brand https://t.co/abc123"])
        reason = degenerate_reason(thread)
        assert reason is not None
        assert "cleaning" in reason


class TestSingleSourcedTaxonomy:
    def test_schema_labels_match_fragments(self):
        # KTD7: the Literal types are built from the fragments tuples.
        assert get_args(CategoryLabel) == fragments.CATEGORY_LABELS
        assert get_args(QueueLabel) == fragments.QUEUE_LABELS

    def test_category_and_queue_share_one_working_set(self):
        # R4: the indicative six-label set serves both, defined in one place.
        assert fragments.QUEUE_LABELS is fragments.CATEGORY_LABELS
        assert len(fragments.CATEGORY_LABELS) == 6
        assert "General Inquiry" in fragments.CATEGORY_LABELS

    def test_draft_status_cannot_be_anything_but_never_sent(self):
        with pytest.raises(ValidationError):
            DraftResult(draft="hi", status="sent")


class TestIntegrationChain:
    def test_full_chain_by_explicit_parameter_passing(self):
        # KTD2/R10: the four steps chain through explicit parameters only.
        client = FakeClient([ok(CATEGORIZED), ok(ROUTED), ok(DRAFTED), ok(ESCALATED)])
        cat = categorize(DIAGNOSABLE, model=MODEL, client=client)
        routed = route(DIAGNOSABLE, cat, model=MODEL, client=client)
        drafted = draft(DIAGNOSABLE, cat, routed, model=MODEL, client=client)
        esc = escalate(DIAGNOSABLE, cat, routed, model=MODEL, client=client)
        result = {
            "category": cat,
            "queue": routed,
            "draft": drafted,
            "escalation": esc,
        }
        assert isinstance(result["category"], CategorizeResult)
        assert isinstance(result["queue"], RouteResult)
        assert isinstance(result["draft"], DraftResult)
        assert isinstance(result["escalation"], EscalateResult)
        assert result["draft"].status == "never_sent"
        assert len(client.messages.calls) == 4


class TestRetrieval:
    @pytest.fixture
    def store_conn(self, tmp_path):
        db_path = tmp_path / "triage.db"
        ingest_csv(FIXTURE_CSV, db_path)
        conn = open_store(db_path)
        yield conn
        conn.close()

    def test_list_threads(self, store_conn):
        rows = list_threads(store_conn, limit=3)
        assert len(rows) == 3
        assert {"thread_id", "root_tweet_id", "customer_author_id", "tweet_count"} <= set(
            rows[0]
        )

    def test_search_threads_finds_by_text(self, store_conn):
        rows = search_threads(store_conn, "package never arrived")
        assert rows
        thread = get_thread_by_id(store_conn, rows[0]["thread_id"])
        assert any("package never arrived" in t.text for t in thread.tweets)

    def test_render_thread_marks_roles_and_text(self, store_conn):
        rows = search_threads(store_conn, "data plan")
        rendered = render_thread(get_thread_by_id(store_conn, rows[0]["thread_id"]))
        assert "Customer (100001):" in rendered
        assert "Support (sprintcare):" in rendered
        assert "my data plan stopped working today" in rendered

    def test_render_thread_notes_truncation(self, store_conn):
        rows = search_threads(store_conn, "still broken after our earlier thread")
        thread = get_thread_by_id(store_conn, rows[0]["thread_id"])
        assert thread.truncated
        assert "missing from the dataset" in render_thread(thread)
