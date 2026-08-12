"""Prompt fragments — the ONE place label sets and step instructions live (KTD7, R4).

Everything a prompt is built from is defined here exactly once:

- The predefined, generic, cross-industry taxonomy (R4). The same indicative
  six-label working set serves BOTH the category and the queue label sets; the
  sets are finalized in a later unit (U6) with changes logged, and any change
  happens here only.
- Per-step instructions for categorize / route / draft / escalate.
- Assembly helpers (system prompt, thread block, prior-outputs block) used by
  the pipeline steps now and by the single-prompt baseline later, so both
  surfaces see equal information (R8).

Nothing here performs I/O or imports the SDK; fragments are pure data and
pure string functions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Taxonomy (R4) — single source for category AND queue label sets.
# ---------------------------------------------------------------------------

TAXONOMY: tuple[tuple[str, str], ...] = (
    (
        "Billing/Payments",
        "Charges, refunds, invoices, payment methods, pricing, or anything about money.",
    ),
    (
        "Technical/Product",
        (
            "The product or service malfunctioning, errors, outages, bugs, or how-to "
            "questions about product functionality."
        ),
    ),
    (
        "Shipping/Delivery",
        (
            "Order status, delayed or missing deliveries, tracking, returns in transit, "
            "or logistics."
        ),
    ),
    (
        "Account/Access",
        "Login problems, password resets, account settings, verification, or account data.",
    ),
    (
        "Complaint/Escalation",
        (
            "Expressions of serious dissatisfaction, repeated unresolved contact, threats "
            "to leave, or demands for a manager/formal complaint."
        ),
    ),
    (
        "General Inquiry",
        (
            "Anything that fits no other label: general questions, feedback, praise, or "
            "threads with too little content to diagnose."
        ),
    ),
)

# The same six-label working set serves both steps (R4); U6 may split them, and
# would do so by editing this module only.
CATEGORY_LABELS: tuple[str, ...] = tuple(label for label, _ in TAXONOMY)
QUEUE_LABELS: tuple[str, ...] = CATEGORY_LABELS

GENERAL_INQUIRY = "General Inquiry"
assert GENERAL_INQUIRY in CATEGORY_LABELS  # R28 fallback must stay in-taxonomy


# ---------------------------------------------------------------------------
# Shared prompt pieces.
# ---------------------------------------------------------------------------

SYSTEM_PREAMBLE = (
    "You are a customer-support triage assistant working on public social-media "
    "support threads. You analyze one reconstructed conversation thread at a time. "
    "Nothing you produce is ever sent to a customer; every output is a recommendation "
    "for human review."
)


def taxonomy_block() -> str:
    """Render the label definitions shown to the model (and to the baseline)."""
    lines = ["Label set (use one of these labels exactly as written; no other label exists):"]
    lines.extend(f"- {label}: {definition}" for label, definition in TAXONOMY)
    return "\n".join(lines)


def step_system(step_instructions: str) -> str:
    """Assemble a step's system prompt from the shared pieces."""
    return f"{SYSTEM_PREAMBLE}\n\n{taxonomy_block()}\n\n{step_instructions}"


def thread_block(rendered_thread: str) -> str:
    """Wrap rendered thread text as the prompt's conversation block."""
    return f"Conversation thread:\n{rendered_thread}"


def prior_outputs_block(
    category: str | None = None,
    queue: str | None = None,
    escalate: bool | None = None,
) -> str:
    """Render prior-step outputs for explicit parameter passing (KTD2).

    Returns an empty string when no prior outputs are available (e.g. an
    upstream step returned a typed failure), so prompts degrade gracefully.
    """
    lines = []
    if category is not None:
        lines.append(f"- Assigned category: {category}")
    if queue is not None:
        lines.append(f"- Assigned support queue: {queue}")
    if escalate is not None:
        lines.append(f"- Needs human attention: {'yes' if escalate else 'no'}")
    if not lines:
        return ""
    return "Prior triage step outputs:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-step instructions.
# ---------------------------------------------------------------------------

CATEGORIZE_INSTRUCTIONS = (
    "Task — categorize: assign exactly one category label from the label set to this "
    "thread, describing what the customer's issue is about. Give a brief rationale "
    "grounded in the thread text."
)

ROUTE_INSTRUCTIONS = (
    "Task — route: assign exactly one support-queue label from the label set, naming "
    "the team best placed to resolve this thread. Give a brief rationale grounded in "
    "the thread text."
)

DRAFT_INSTRUCTIONS = (
    "Task — draft: write a reply draft to the customer in the brand's public support "
    "voice: concise, empathetic, concrete about the next step, and suitable for the "
    "same channel as the thread. Draft a reply even if the thread needs human "
    "attention. The draft is never sent; it exists only for human review, so its "
    "status is always 'never_sent'."
)

ESCALATE_INSTRUCTIONS = (
    "Task — escalate: decide whether this thread needs human attention (true or "
    "false) and state the reason for your decision. Escalate for angry or churn-risk "
    "customers, legal/safety issues, repeated unresolved contact, or anything a "
    "support macro cannot resolve."
)


# ---------------------------------------------------------------------------
# Degenerate-thread handling (R28).
# ---------------------------------------------------------------------------

def insufficient_content_reason(detail: str) -> str:
    """The stated insufficient-content reason attached to degenerate results."""
    return f"Insufficient content: {detail}"


DEGENERATE_DRAFT = (
    "Hi, thanks for reaching out! We'd like to help, but we couldn't find enough "
    "detail in this thread to know what's going on. Could you share a bit more — "
    "what you were trying to do, what happened instead, and any error message you "
    "saw? We'll pick it up from there."
)
