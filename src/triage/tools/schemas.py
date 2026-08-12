"""Result schemas for the tool layer (R9 shapes, R4 taxonomy, R27 typed failure).

The label types are built from the single-sourced taxonomy in
``triage.prompts.fragments`` as ``Literal`` types, so an off-taxonomy label
fails Pydantic validation inside the schema-enforced call (KTD4) instead of
leaking into results.

``OutputFailure`` is the typed result a tool returns when the model's output
is still malformed, off-taxonomy, or refused after the schema-enforced
retries — the tool layer never raises for bad model output and never excludes
a thread (R27).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from triage.prompts.fragments import CATEGORY_LABELS, QUEUE_LABELS

# Literal types built at runtime from the single-sourced label tuples (KTD7):
# validation is the taxonomy gate.
CategoryLabel = Literal[CATEGORY_LABELS]  # runtime-built Literal
QueueLabel = Literal[QUEUE_LABELS]  # runtime-built Literal

# Draft results are recommendations only; nothing is ever sent (R6).
NEVER_SENT = "never_sent"


class CategorizeResult(BaseModel):
    """Category label plus rationale (R4, R9)."""

    label: CategoryLabel = Field(description="Exactly one category label from the label set.")
    rationale: str = Field(description="Brief rationale grounded in the thread text.")


class RouteResult(BaseModel):
    """Support-queue label plus rationale (R4, R9)."""

    queue: QueueLabel = Field(description="Exactly one support-queue label from the label set.")
    rationale: str = Field(description="Brief rationale grounded in the thread text.")


class DraftResult(BaseModel):
    """Reply draft for human review; carries the explicit never-sent status (R6, R9)."""

    draft: str = Field(description="The reply draft, in the brand's public support voice.")
    status: Literal["never_sent"] = Field(
        default=NEVER_SENT,
        description="Always 'never_sent': drafts exist only for human review.",
    )


class EscalateResult(BaseModel):
    """Binary needs-human-attention decision with a stated reason (R5)."""

    escalate: bool = Field(description="True when this thread needs human attention.")
    reason: str = Field(description="The stated reason for the decision.")


# Failure kinds (R27).
FAILURE_MALFORMED = "malformed"
FAILURE_OFF_TAXONOMY = "off_taxonomy"
FAILURE_REFUSAL = "refusal"
FAILURE_KINDS = (FAILURE_MALFORMED, FAILURE_OFF_TAXONOMY, FAILURE_REFUSAL)


class OutputFailure(BaseModel):
    """Typed failure result after schema-enforced retries are exhausted (R27)."""

    step: str = Field(description="The tool step that failed (e.g. 'categorize').")
    kind: Literal[FAILURE_KINDS] = Field(  # runtime-built Literal
        description="Why the output was unusable: malformed, off_taxonomy, or refusal."
    )
    attempts: int = Field(description="Total model calls made (initial attempt + retries).")
    detail: str = Field(description="Detail from the last failed attempt.")
