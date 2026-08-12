"""Draft step: a reply draft for every thread, never sent (R6, R9).

Pure, explicitly-parameterized step function (KTD2). Every thread gets a
draft — including escalated and degenerate ones; the result's ``status``
field is always ``never_sent`` because all output is for human review.
"""

from __future__ import annotations

from typing import Any

from triage.ingest.reconstruct import Thread
from triage.prompts import fragments
from triage.tools.llm import call_with_schema
from triage.tools.retrieval import degenerate_reason, render_thread
from triage.tools.schemas import CategorizeResult, DraftResult, OutputFailure, RouteResult

STEP_NAME = "draft"


def draft(
    thread: Thread,
    category: CategorizeResult | OutputFailure | None = None,
    queue: RouteResult | OutputFailure | None = None,
    *,
    model: str,
    client: Any = None,
) -> DraftResult | OutputFailure:
    """Write a reply draft for human review; nothing is ever sent (R6).

    Degenerate threads get a deterministic ask-for-details draft with no LLM
    call (R28): a valid result, still never sent.
    """
    if degenerate_reason(thread) is not None:
        return DraftResult(draft=fragments.DEGENERATE_DRAFT)
    category_label = category.label if isinstance(category, CategorizeResult) else None
    queue_label = queue.queue if isinstance(queue, RouteResult) else None
    user_text = fragments.thread_block(render_thread(thread))
    prior = fragments.prior_outputs_block(category=category_label, queue=queue_label)
    if prior:
        user_text = f"{user_text}\n\n{prior}"
    return call_with_schema(
        STEP_NAME,
        model=model,
        system=fragments.step_system(fragments.DRAFT_INSTRUCTIONS),
        user_text=user_text,
        schema=DraftResult,
        client=client,
    )
