"""Shared tool layer: one implementation, every surface (R10, KTD2).

Framework-free, pure, explicitly-parameterized step functions — the LangGraph
pipeline, the MCP server, and the CLI all adapt these same implementations:

- :func:`categorize` — category label + rationale (R4)
- :func:`route` — support-queue label + rationale (R4)
- :func:`draft` — reply draft, always ``never_sent`` (R6)
- :func:`escalate` — binary needs-human-attention decision + reason (R5)

Signature shape: ``fn(thread, prior_outputs..., *, model, client=None)`` —
prior-step outputs are always passed explicitly, never via hidden state
(KTD2). Each returns a Pydantic result or a typed :class:`OutputFailure`
(never raises for bad model output, R27); degenerate threads short-circuit
before any LLM call (R28).
"""

from __future__ import annotations

from triage.tools.categorize import categorize
from triage.tools.draft import draft
from triage.tools.escalate import escalate
from triage.tools.route import route
from triage.tools.schemas import (
    CategorizeResult,
    DraftResult,
    EscalateResult,
    OutputFailure,
    RouteResult,
)

__all__ = [
    "CategorizeResult",
    "DraftResult",
    "EscalateResult",
    "OutputFailure",
    "RouteResult",
    "categorize",
    "draft",
    "escalate",
    "route",
]
