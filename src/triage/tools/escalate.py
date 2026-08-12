"""Escalate step: binary needs-human-attention decision with a reason (R5).

Pure, explicitly-parameterized step function (KTD2). Escalation never blocks
drafting — the draft step runs for escalated threads too (R6).
"""

from __future__ import annotations

from typing import Any

from triage.ingest.reconstruct import Thread
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import degenerate_reason, render_thread
from triage.tools.schemas import CategorizeResult, EscalateResult, OutputFailure, RouteResult

STEP_NAME = "escalate"


def escalate(
    thread: Thread,
    category: CategorizeResult | OutputFailure | None = None,
    queue: RouteResult | OutputFailure | None = None,
    *,
    model: str,
    client: Any = None,
) -> EscalateResult | OutputFailure:
    """Decide whether the thread needs human attention, with a stated reason (R5).

    Degenerate threads short-circuit to escalate=True with the stated
    insufficient-content reason, before any LLM call (R28).
    """
    reason = degenerate_reason(thread)
    if reason is not None:
        return EscalateResult(
            escalate=True, reason=fragments.insufficient_content_reason(reason)
        )
    category_label = category.label if isinstance(category, CategorizeResult) else None
    queue_label = queue.queue if isinstance(queue, RouteResult) else None
    user_text = fragments.thread_block(render_thread(thread))
    prior = fragments.prior_outputs_block(category=category_label, queue=queue_label)
    if prior:
        user_text = f"{user_text}\n\n{prior}"
    return call_with_schema(
        STEP_NAME,
        model=model,
        system=fragments.step_system(fragments.ESCALATE_INSTRUCTIONS),
        user_text=user_text,
        schema=EscalateResult,
        client=client,
    )
