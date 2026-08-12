"""Prompt fragments — the ONE place label sets and step instructions live (KTD7, R4).

Everything a prompt is built from is defined here exactly once:

- The two predefined, generic, cross-industry label sets (R4), finalized in U6
  and changed here only. They are deliberately disjoint vocabularies on
  different axes: CATEGORY names what the issue is about, QUEUE names which
  team owns it. Sharing one set (as the indicative planning set did) made route
  a restatement of categorize rather than a distinct decision.
- Per-step instructions for categorize / route / draft / escalate.
- Assembly helpers (system prompt, thread block, prior-outputs block) used by
  the pipeline steps now and by the single-prompt baseline later, so both
  surfaces see equal information (R8).

Nothing here performs I/O or imports the SDK; fragments are pure data and
pure string functions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Category taxonomy (R4) — what the customer's issue is about.
# ---------------------------------------------------------------------------

CATEGORY_TAXONOMY: tuple[tuple[str, str], ...] = (
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
        "Complaint/Dispute",
        (
            "The grievance itself is the subject: general service dissatisfaction, a "
            "threat to leave, or a formal complaint. Prefer the specific functional "
            "label when the grievance is about a concrete billing, shipping, technical, "
            "or access issue. Says nothing about urgency — that is decided separately."
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


# ---------------------------------------------------------------------------
# Queue taxonomy (R4) — which team owns the thread. Ownership only: a queue
# naming escalation would be near-deterministically escalate=true, duplicating
# that decision into the label space.
# ---------------------------------------------------------------------------

QUEUE_TAXONOMY: tuple[tuple[str, str], ...] = (
    (
        "Tier-1 General",
        "Front-line handling: self-serve answers, simple questions, deflection.",
    ),
    (
        "Billing Ops",
        "Finance-authorized actions: refunds, adjustments, invoice corrections.",
    ),
    (
        "Technical Support",
        "Diagnosis requiring product or engineering knowledge.",
    ),
    (
        "Logistics",
        "Carrier contact, shipment tracing, warehouse and returns handling.",
    ),
    (
        "Account Security",
        "Identity verification, lockouts, suspected account compromise.",
    ),
    (
        "Trust & Safety",
        "Fraud, abuse, threats, legal or reputational exposure.",
    ),
)

CATEGORY_LABELS: tuple[str, ...] = tuple(label for label, _ in CATEGORY_TAXONOMY)
QUEUE_LABELS: tuple[str, ...] = tuple(label for label, _ in QUEUE_TAXONOMY)

# R28 fallbacks — each must stay inside its own step's label set.
GENERAL_INQUIRY = "General Inquiry"
DEGENERATE_QUEUE = "Tier-1 General"
assert GENERAL_INQUIRY in CATEGORY_LABELS
assert DEGENERATE_QUEUE in QUEUE_LABELS
assert set(CATEGORY_LABELS).isdisjoint(QUEUE_LABELS)


# ---------------------------------------------------------------------------
# Shared prompt pieces.
# ---------------------------------------------------------------------------

SYSTEM_PREAMBLE = (
    "You are a customer-support triage assistant working on public social-media "
    "support threads. You analyze one reconstructed conversation thread at a time. "
    "Nothing you produce is ever sent to a customer; every output is a recommendation "
    "for human review."
)


def _label_block(heading: str, taxonomy: tuple[tuple[str, str], ...]) -> str:
    lines = [f"{heading} (use one of these labels exactly as written; no other label exists):"]
    lines.extend(f"- {label}: {definition}" for label, definition in taxonomy)
    return "\n".join(lines)


def category_block() -> str:
    """Category label definitions — what the issue is about."""
    return _label_block("Category label set", CATEGORY_TAXONOMY)


def queue_block() -> str:
    """Queue label definitions — which team owns the thread."""
    return _label_block("Support queue label set", QUEUE_TAXONOMY)


def step_system(step_instructions: str, *label_blocks: str) -> str:
    """Assemble a step's system prompt from the shared pieces.

    A step receives only the label space it must choose from; showing route the
    category definitions re-anchors it on content.
    """
    parts = [SYSTEM_PREAMBLE, *label_blocks, step_instructions]
    return "\n\n".join(parts)


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
    "the team that should own this thread. Decide by which team's authority and skills "
    "the resolution actually needs — not by restating what the issue is about. A "
    "billing question a front-line agent can answer belongs to Tier-1 General, not "
    "Billing Ops; an account problem showing signs of compromise belongs to Account "
    "Security; fraud, abuse, or legal exposure belongs to Trust & Safety whatever the "
    "topic. Do not choose a queue on urgency alone — urgency is decided separately. "
    "Give a brief rationale grounded in the thread text."
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
