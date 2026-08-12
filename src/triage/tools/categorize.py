"""Categorize step: one taxonomy category per thread, with rationale (R4, R9).

Pure, explicitly-parameterized step function (KTD2): takes the thread and a
model id, returns a Pydantic result — no hidden state, no framework.
"""

from __future__ import annotations

from typing import Any

from triage.ingest.reconstruct import Thread
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import build_user_text, degenerate_reason
from triage.tools.schemas import CategorizeResult, OutputFailure

STEP_NAME = "categorize"


def categorize(
    thread: Thread, *, model: str, client: Any = None
) -> CategorizeResult | OutputFailure:
    """Assign one category label from the single-sourced taxonomy (R4).

    Degenerate threads short-circuit to General Inquiry with an
    insufficient-content rationale before any LLM call (R28).
    """
    reason = degenerate_reason(thread)
    if reason is not None:
        return CategorizeResult(
            label=fragments.GENERAL_INQUIRY,
            rationale=fragments.insufficient_content_reason(reason),
        )
    return call_with_schema(
        STEP_NAME,
        model=model,
        system=fragments.step_system(fragments.CATEGORIZE_INSTRUCTIONS),
        user_text=build_user_text(thread),
        schema=CategorizeResult,
        client=client,
    )
