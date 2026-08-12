"""Pipeline state: a TypedDict flowing through the LangGraph StateGraph (R3, KTD4).

One state per thread run. Each node writes its own result key (partial state
update); ``failed_steps`` accumulates the names of steps whose result was a
typed :class:`OutputFailure` (R27), so the pipeline result can surface which
step failed without losing the others.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from triage.ingest.reconstruct import Thread
from triage.tools import (
    CategorizeResult,
    DraftResult,
    EscalateResult,
    OutputFailure,
    RouteResult,
)


class PipelineState(TypedDict, total=False):
    """State for one thread's run through the four-step graph (R3).

    ``thread``, ``model``, and (optionally) ``client`` are inputs; the four
    result keys are written one-per-node; ``failed_steps`` accumulates across
    nodes via the ``operator.add`` reducer.
    """

    thread: Thread
    model: str
    client: Any  # injectable for tests; None -> real client in the tool layer
    category: CategorizeResult | OutputFailure
    queue: RouteResult | OutputFailure
    escalation: EscalateResult | OutputFailure
    draft: DraftResult | OutputFailure
    failed_steps: Annotated[list[str], operator.add]
