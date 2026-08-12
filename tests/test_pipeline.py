"""Tests for the LangGraph pipeline (U4) — R3, R6, R27, R28, KTD4, AE1, AE2.

Covers:
- Graph wiring: four distinct nodes, one per step (R3), on a LangGraph 1.x
  StateGraph (KTD4).
- AE1 happy path: one invoke yields category, queue, escalation + reason, and
  draft — each attributable to its own step in the result dict.
- AE2: an escalated thread still carries a never-sent draft (R6) — the
  conditional edge after the escalate assessment never suppresses drafting.
- Failure propagation: a node-level OutputFailure (R27) is carried in state,
  downstream nodes still run, and the result names the failing step.
- Degenerate threads (R28): the full pipeline short-circuits to the
  General Inquiry + escalated result with zero LLM calls.

Every test is fully mocked: zero network, no anthropic client constructed.
"""

from test_tools import (
    CATEGORIZED,
    DIAGNOSABLE,
    DRAFTED,
    ESCALATED,
    HAPPY_OUTCOMES,
    MODEL,
    ROUTED,
    ExplodingClient,
    FakeClient,
    make_thread,
    ok,
    validation_error_for,
)

from triage.pipeline import build_graph, result_dict, run_pipeline
from triage.tools import (
    CategorizeResult,
    DraftResult,
    EscalateResult,
    OutputFailure,
    RouteResult,
)

# AE2 needs an affirmative escalation (test_tools' ESCALATED is escalate=False).
ESCALATED_UP = EscalateResult(escalate=True, reason="Customer is furious and threatening churn.")


class TestGraphWiring:
    def test_four_distinct_nodes(self):
        # R3: categorize, route, draft, escalate are separate nodes, not one prompt.
        graph = build_graph()
        assert {"categorize", "route", "escalate", "draft"} <= set(graph.nodes)

    def test_nodes_run_in_step_order_with_own_schemas(self):
        # Each node makes its own schema-enforced call: distinct inputs/outputs (R3).
        client = FakeClient(list(HAPPY_OUTCOMES))
        run_pipeline(DIAGNOSABLE, model=MODEL, client=client)
        formats = [call["output_format"] for call in client.messages.calls]
        assert formats == [CategorizeResult, RouteResult, EscalateResult, DraftResult]


class TestHappyPath:
    def test_full_pipeline_yields_all_four_step_outputs(self):
        # AE1: category, queue, escalation decision + reason, and draft — each
        # attributable to its own pipeline step.
        client = FakeClient(list(HAPPY_OUTCOMES))
        state = run_pipeline(DIAGNOSABLE, model=MODEL, client=client)
        assert isinstance(state["category"], CategorizeResult)
        assert isinstance(state["queue"], RouteResult)
        assert isinstance(state["escalation"], EscalateResult)
        assert isinstance(state["draft"], DraftResult)
        assert len(client.messages.calls) == 4

        result = result_dict(state)
        steps = result["steps"]
        assert steps["categorize"]["label"] == "Technical/Product"
        assert steps["categorize"]["rationale"]
        assert steps["route"]["queue"] == "Technical/Product"
        assert steps["route"]["rationale"]
        assert steps["escalate"]["escalate"] is False
        assert steps["escalate"]["reason"]
        assert steps["draft"]["status"] == "never_sent"
        assert steps["draft"]["draft"]
        assert result["ok"] is True
        assert result["failed_steps"] == []

    def test_result_carries_thread_identity(self):
        client = FakeClient(list(HAPPY_OUTCOMES))
        state = run_pipeline(DIAGNOSABLE, model=MODEL, client=client)
        result = result_dict(state)
        assert result["thread"]["root_tweet_id"] == DIAGNOSABLE.root_tweet_id
        assert tuple(result["thread"]["tweet_ids"]) == DIAGNOSABLE.tweet_ids


class TestEscalationNeverSuppressesDraft:
    def test_escalated_thread_still_carries_never_sent_draft(self):
        # AE2/R6: escalation is a branch point in the graph, but both branches
        # still produce the draft, marked for human review.
        client = FakeClient([ok(CATEGORIZED), ok(ROUTED), ok(ESCALATED_UP), ok(DRAFTED)])
        state = run_pipeline(DIAGNOSABLE, model=MODEL, client=client)
        assert state["escalation"].escalate is True
        assert isinstance(state["draft"], DraftResult)
        assert state["draft"].status == "never_sent"
        result = result_dict(state)
        assert result["steps"]["escalate"]["escalate"] is True
        assert result["steps"]["draft"]["status"] == "never_sent"


class TestFailurePropagation:
    def test_node_failure_is_carried_and_downstream_still_runs(self):
        # R27: a persistent categorize failure becomes a typed OutputFailure in
        # state; route/escalate/draft still run (tolerating the failed prior).
        bad = [validation_error_for(CategorizeResult, "{not json") for _ in range(3)]
        client = FakeClient(bad + [ok(ROUTED), ok(ESCALATED), ok(DRAFTED)])
        state = run_pipeline(DIAGNOSABLE, model=MODEL, client=client)
        assert isinstance(state["category"], OutputFailure)
        assert state["failed_steps"] == ["categorize"]
        assert isinstance(state["queue"], RouteResult)
        assert isinstance(state["escalation"], EscalateResult)
        assert isinstance(state["draft"], DraftResult)

        result = result_dict(state)
        assert result["ok"] is False
        assert result["failed_steps"] == ["categorize"]
        assert result["steps"]["categorize"]["failure"]["step"] == "categorize"
        assert result["steps"]["categorize"]["failure"]["kind"] == "malformed"


class TestDegenerateThreads:
    def test_degenerate_thread_short_circuits_with_zero_llm_calls(self):
        # R28: the full pipeline over an undiagnosable thread yields the valid
        # General Inquiry + escalated result without touching the client.
        thread = make_thread(["@brand help"])
        state = run_pipeline(thread, model=MODEL, client=ExplodingClient())
        assert state["category"].label == "General Inquiry"
        assert state["queue"].queue == "General Inquiry"
        assert state["escalation"].escalate is True
        assert "insufficient content" in state["escalation"].reason.lower()
        assert state["draft"].status == "never_sent"
        result = result_dict(state)
        assert result["ok"] is True
        assert result["failed_steps"] == []
