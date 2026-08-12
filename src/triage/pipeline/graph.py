"""The four-step LangGraph pipeline: categorize -> route -> escalate -> draft (R3, KTD4).

A LangGraph 1.x ``StateGraph`` over :class:`PipelineState`; each node is a thin
adapter over its tool-layer step function (KTD2), passing prior outputs
explicitly and returning a partial state update. No checkpointer (batch
pipeline), no ``langgraph.prebuilt``.

The conditional edge after the escalate assessment makes the escalation
decision a visible branch point in the graph — but both branches lead to the
draft node, because escalation never suppresses the draft (R6, AE2): every
thread gets a never-sent draft for human review.

A node whose tool returns a typed :class:`OutputFailure` (R27) carries it into
state and appends its step name to ``failed_steps``; downstream nodes still
run (the tool layer tolerates failed priors), so the result surfaces which
step failed without losing the rest.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from triage.ingest.reconstruct import Thread
from triage.pipeline.state import PipelineState
from triage.tools import EscalateResult, OutputFailure, categorize, draft, escalate, route


def _step_update(key: str, result: Any) -> dict[str, Any]:
    """Partial state update for one node; records typed failures (R27)."""
    update: dict[str, Any] = {key: result}
    if isinstance(result, OutputFailure):
        update["failed_steps"] = [result.step]
    return update


def _categorize_node(state: PipelineState) -> dict[str, Any]:
    """Categorize step (R4): thread -> category label + rationale."""
    result = categorize(state["thread"], model=state["model"], client=state.get("client"))
    return _step_update("category", result)


def _route_node(state: PipelineState) -> dict[str, Any]:
    """Route step (R4): thread + category -> queue label + rationale."""
    result = route(
        state["thread"],
        state.get("category"),
        model=state["model"],
        client=state.get("client"),
    )
    return _step_update("queue", result)


def _escalate_node(state: PipelineState) -> dict[str, Any]:
    """Escalate step (R5): thread + category + queue -> decision + reason."""
    result = escalate(
        state["thread"],
        state.get("category"),
        state.get("queue"),
        model=state["model"],
        client=state.get("client"),
    )
    return _step_update("escalation", result)


def _draft_node(state: PipelineState) -> dict[str, Any]:
    """Draft step (R6): thread + category + queue -> never-sent reply draft."""
    result = draft(
        state["thread"],
        state.get("category"),
        state.get("queue"),
        model=state["model"],
        client=state.get("client"),
    )
    return _step_update("draft", result)


def _after_escalate(state: PipelineState) -> str:
    """Branch on the escalation assessment — both branches draft (R6, AE2)."""
    escalation = state.get("escalation")
    if isinstance(escalation, EscalateResult) and escalation.escalate:
        return "escalated"
    return "routine"


def build_graph():
    """Compile the four-node StateGraph (R3, KTD4)."""
    builder = StateGraph(PipelineState)
    builder.add_node("categorize", _categorize_node)
    builder.add_node("route", _route_node)
    builder.add_node("escalate", _escalate_node)
    builder.add_node("draft", _draft_node)
    builder.add_edge(START, "categorize")
    builder.add_edge("categorize", "route")
    builder.add_edge("route", "escalate")
    # Escalation is a decision point, never a draft suppressor (R6, AE2).
    builder.add_conditional_edges(
        "escalate", _after_escalate, {"escalated": "draft", "routine": "draft"}
    )
    builder.add_edge("draft", END)
    return builder.compile()


def run_pipeline(thread: Thread, *, model: str, client: Any = None) -> PipelineState:
    """Run one thread through the pipeline; returns the final state.

    ``client`` is injectable for tests (KTD2); ``None`` lets the tool layer
    construct the real client from the environment.
    """
    state: PipelineState = {"thread": thread, "model": model, "failed_steps": []}
    if client is not None:
        state["client"] = client
    return build_graph().invoke(state)


def _step_payload(result: Any) -> dict[str, Any] | None:
    """JSON-ready payload for one step: the result, or its typed failure."""
    if result is None:
        return None
    if isinstance(result, OutputFailure):
        return {"failure": result.model_dump()}
    return result.model_dump()


def result_dict(state: PipelineState) -> dict[str, Any]:
    """Structured triage result: four step outputs, each attributable (R7, AE1)."""
    thread = state["thread"]
    failed_steps = list(state.get("failed_steps") or [])
    return {
        "thread": {
            "root_tweet_id": thread.root_tweet_id,
            "customer_author_id": thread.customer_author_id,
            "tweet_ids": list(thread.tweet_ids),
            "truncated": thread.truncated,
            "cycle_flagged": thread.cycle_flagged,
        },
        "steps": {
            "categorize": _step_payload(state.get("category")),
            "route": _step_payload(state.get("queue")),
            "escalate": _step_payload(state.get("escalation")),
            "draft": _step_payload(state.get("draft")),
        },
        "failed_steps": failed_steps,
        "ok": not failed_steps,
    }
