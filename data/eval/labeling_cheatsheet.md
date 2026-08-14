# Labeling cheat sheet (U6, R24)

The annotator's desk reference. Definitions are copied verbatim from
`src/triage/prompts/fragments.py`, the single source both the models and this
sheet work from — if they ever disagree, `fragments.py` wins.

**The one rule that prevents most mistakes:** label the *topic of the
customer's problem*, not their mood. An angry thread about a smashed package
is Shipping/Delivery. A polite thread about a wrong charge is
Billing/Payments. Mood is irrelevant to category and queue; urgency is
decided only in the escalate pass.

## Pass 1 — category: what is the issue about?

| # | Label | Definition (verbatim) | Quick examples |
|---|-------|-----------------------|----------------|
| 1 | Billing/Payments | Charges, refunds, invoices, payment methods, pricing, or anything about money. | "charged twice"; "where's my refund"; "why did the price go up" |
| 2 | Technical/Product | The product or service malfunctioning, errors, outages, bugs, or how-to questions about product functionality. | "app crashes on login"; "error 105"; "how do I turn on captions" |
| 3 | Shipping/Delivery | Order status, delayed or missing deliveries, tracking, returns in transit, or logistics. | "package 2 weeks late"; "tracking says delivered but nothing came" |
| 4 | Account/Access | Login problems, password resets, account settings, verification, or account data. | "locked out"; "someone registered with my email"; "can't verify my number" |
| 5 | Complaint/Dispute | The grievance itself is the subject: general service dissatisfaction, a threat to leave, or a formal complaint. Prefer the specific functional label when the grievance is about a concrete billing, shipping, technical, or access issue. Says nothing about urgency — that is decided separately. | "you people are the worst, switching providers" (no concrete issue named) |
| 6 | General Inquiry | Anything that fits no other label: general questions, feedback, praise, or threads with too little content to diagnose. | "love the new ad!"; "what time do you open"; brand chatter |

**Tie-breaks:**
- Angry + concrete issue -> the functional label (1-4), never 5.
- 5 only when no functional label fits and the dissatisfaction IS the story.
- 6 only when there is no issue at all. If the customer wants something
  fixed, it is not General Inquiry.

## Pass 2 — queue: which team should own it?

| # | Label | Definition (verbatim) | Rule of thumb |
|---|-------|-----------------------|---------------|
| 1 | Tier-1 General | Front-line handling: self-serve answers, simple questions, deflection. | Anyone with a macro could answer it |
| 2 | Billing Ops | Finance-authorized actions: refunds, adjustments, invoice corrections. | Someone must be allowed to move money |
| 3 | Technical Support | Diagnosis requiring product or engineering knowledge. | Someone must debug or know internals |
| 4 | Logistics | Carrier contact, shipment tracing, warehouse and returns handling. | Someone must chase a physical package |
| 5 | Account Security | Identity verification, lockouts, suspected account compromise. | Someone must verify who the customer is |
| 6 | Trust & Safety | Fraud, abuse, threats, legal or reputational exposure. | Someone must protect people or the platform |

Queue is ownership, not topic: a billing *question* answerable from a help
page is Tier-1 General; a refund that needs authorization is Billing Ops.
Category and queue legitimately diverge — that divergence is one of the
things this eval measures, so judge each pass fresh.

## Pass 3 — escalate: does this need a human now?

`true` when a human should intervene promptly: the customer is blocked with
no self-serve path, there is account compromise, fraud, safety, legal or
reputational risk, repeated failed contact, or visible severe frustration.
`false` for routine matters a queue can work through normal channels.
