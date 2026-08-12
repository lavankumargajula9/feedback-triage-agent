"""Escalate step: binary needs-human-attention decision with a reason (R5).

Pure, explicitly-parameterized step function (KTD2). Escalation never blocks
drafting — the draft step runs for escalated threads too (R6).
"""

from __future__ import annotations

from typing import Any

from triage.ingest.reconstruct import Thread
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import build_user_text, degenerate_reason
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
    return call_with_schema(
        STEP_NAME,
        model=model,
        system=fragments.step_system(fragments.ESCALATE_INSTRUCTIONS),
        user_text=build_user_text(thread, category, queue),
        schema=EscalateResult,
        client=client,
    )
