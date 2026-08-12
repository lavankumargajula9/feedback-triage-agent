"""Route step: one support-queue label per thread, with rationale (R4, R9).

Pure, explicitly-parameterized step function (KTD2): prior-step outputs are
passed as parameters, never read from hidden state.
"""

from __future__ import annotations

from typing import Any

from triage.ingest.reconstruct import Thread
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import degenerate_reason, render_thread
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
    category_label = category.label if isinstance(category, CategorizeResult) else None
    user_text = fragments.thread_block(render_thread(thread))
    prior = fragments.prior_outputs_block(category=category_label)
    if prior:
        user_text = f"{user_text}\n\n{prior}"
    return call_with_schema(
        STEP_NAME,
        model=model,
        system=fragments.step_system(fragments.ROUTE_INSTRUCTIONS),
        user_text=user_text,
        schema=RouteResult,
        client=client,
    )
