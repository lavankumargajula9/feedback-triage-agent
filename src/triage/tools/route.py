"""Route step: one support-queue label per thread, with rationale (R4, R9).

Pure, explicitly-parameterized step function (KTD2): prior-step outputs are
passed as parameters, never read from hidden state.
"""

from __future__ import annotations

from typing import Any

from triage.ingest.reconstruct import Thread
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import build_user_text, degenerate_reason
from triage.tools.schemas import CategorizeResult, OutputFailure, RouteResult

STEP_NAME = "route"


def route(
    thread: Thread,
    category: CategorizeResult | OutputFailure | None = None,
    *,
    model: str,
    client: Any = None,
) -> RouteResult | OutputFailure:
    """Assign one queue label from the single-sourced taxonomy (R4).

    ``category`` is the categorize step's output; a typed failure or None
    simply omits the prior-outputs block. Degenerate threads short-circuit to
    the General Inquiry queue before any LLM call (R28).
    """
    reason = degenerate_reason(thread)
    if reason is not None:
        return RouteResult(
            queue=fragments.GENERAL_INQUIRY,
            rationale=fragments.insufficient_content_reason(reason),
        )
    return call_with_schema(
        STEP_NAME,
        model=model,
        system=fragments.step_system(fragments.ROUTE_INSTRUCTIONS),
        user_text=build_user_text(thread, category),
        schema=RouteResult,
        client=client,
    )
