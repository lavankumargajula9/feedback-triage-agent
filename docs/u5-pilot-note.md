# U5 functional pilot — note

**Date:** 2026-08-12 · **Verdict: GO** · Requirement: R20 · Gate: functional, not metric.

## What ran

Eight threads end-to-end through the four-step pipeline on the **dev** profile
(`claude-haiku-4-5`), plus one repeat on the **measurement** profile
(`claude-opus-5`). Every invocation exited 0; no unhandled exceptions.

| # | Thread | Shape exercised | Category | Queue | Escalate |
|---|---|---|---|---|---|
| 1 | store tweet 1 | resolved 4-tweet thread | Technical/Product | Technical/Product | false |
| 2 | store tweet 10 | truncated — parent 999 absent | Technical/Product | Technical/Product | true |
| 3 | store tweet 15 | cycle-flagged (15 ↔ 16) | Billing/Payments | Billing/Payments | false |
| 4 | store tweet 20 | tree-shaped, customer 100005 branch | Shipping/Delivery | Shipping/Delivery | false |
| 5 | store tweet 22 | sibling customer 100006, same root | Shipping/Delivery | Shipping/Delivery | false |
| 6 | store tweet 30 | single tweet, diagnosable content | General Inquiry | General Inquiry | false |
| 7 | inline | non-English (Spanish) + supervisor demand | Billing/Payments | Complaint/Escalation | true |
| 8 | inline | DM deflection, undiagnosable | General Inquiry | General Inquiry | true |
| 9 | store tweet 1, measurement profile | model-pair smoke | Technical/Product | Technical/Product | false |

## Pathological-case findings

- **Truncation, cycles, and sibling-customer trees** all reconstructed to the
  expected member sets and triaged without incident. Thread 5 resolved to the
  *other* customer's branch off the shared root, confirming the entry-point
  rule under R31.
- **R28/AE7 holds.** Thread 8 short-circuited to General Inquiry + escalate=true
  with an insufficient-content reason, and still carried a `never_sent` draft
  (R6). Thread 6 did *not* short-circuit: a single tweet with diagnosable
  content ("the new bowl is great") is triageable, so degeneracy is decided by
  content, not tweet count. This is the intended reading of R28 — AE7 qualifies
  the single-tweet case with "no diagnosable content."
- **Non-English** (thread 7) was handled without a language guard: the model
  read Spanish, categorized it, and escalated on the supervisor demand. Per
  R28 this is "whatever the model produces," disclosed in limitations rather
  than engineered.
- **Profile agreement.** Thread 9 matched thread 1 on all four steps; Opus 5's
  rationales were longer and cited more thread evidence, but no label moved.

## Observation carried forward to U11

**Queue equalled category on 7 of 8 threads.** Only the non-English thread
diverged (Billing/Payments → Complaint/Escalation, on the supervisor demand).
This is a small, non-random fixture and proves nothing on its own, but it is the
first evidence bearing on whether the route step earns its keep as a decision
distinct from categorize. The measured eval must report category/queue
divergence rate; if it stays near zero, that belongs in the README's error
analysis (R23), not hidden.

## Scope limitation of this pilot

R20 asks for ~15-20 threads. This pilot ran 8, because the Kaggle download and
its redistribution-license verdict (R2/R29) are still open, so no real `twcs`
threads are in the store — six came from the committed test fixture and two were
authored inline to reach the non-English and DM-deflect shapes. **Every
pathological shape R20 names is covered**; the shortfall is volume and
real-traffic messiness, not case coverage. Re-run this pilot over ~15 real
threads once the dataset gate clears, before U6 labeling begins.

## Cost

Estimated, not metered — the CLI `run` path does not capture usage (that was
added to the eval runner only):

- 8 dev threads × 4 steps = 32 Haiku 4.5 calls, minus 4 short-circuited by
  thread 8 ≈ 28 calls × ~1,500 in / ~300 out at $1/$5 per MTok ≈ **$0.08**
- 4 Opus 5 calls × ~1,500 in / ~300 out at $5/$25 per MTok ≈ **$0.06**
- Pilot total ≈ **$0.15** against the $100 ceiling.
