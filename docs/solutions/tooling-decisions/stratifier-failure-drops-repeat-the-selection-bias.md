---
title: A stratifier that drops what its own classifier failed on repeats the selection bias
date: 2026-08-13
category: tooling-decisions
module: triage.evals
problem_type: tooling_decision
component: testing_framework
severity: high
applies_when:
  - A model pass decides which items enter a sample, and that pass can fail
  - Items whose classification call errors, refuses, or fails validation are skipped
  - A retry budget is exhausted and the item is dropped rather than parked
  - The classifier's failure rate is assumed to be random with respect to content
  - The headline result is a gap between two arms rather than one absolute score
tags:
  - eval-set
  - stratified-sampling
  - construct-validity
  - sampling-bias
  - error-handling
  - measurement
---

# A stratifier that drops what its own classifier failed on repeats the selection bias

## Context

[Eval-pool selection rules must not correlate with the difficulty being
measured](keyword-stratification-biases-eval-pool.md) settled that a keyword
stratifier is a construct-validity defect: it admits only threads whose class is
lexically obvious, both arms inflate, and the measured lift compresses toward
zero. The dev-model stratifier was chosen precisely because it does not filter on
vocabulary.

The stratifier shipped in `794ee86`. It scores every scanned thread on three
isolated passes — category, queue, escalation — through the shared
schema-enforced wrapper, which retries twice and then returns a typed
`OutputFailure` rather than raising (`src/triage/tools/llm.py:32`,
`src/triage/tools/schemas.py:63-66`). The first implementation did the obvious
thing with that failure:

```python
if isinstance(result, OutputFailure):
    failures[f"{CRITERION_ROUGH_FAILED}:{pass_.name}"] += 1
    failed.add(thread_id)          # thread is dropped from the pool
    continue
```

That is the same bug the earlier document is about, reached from a completely
different direction, and it survived the original implementation, a
three-reviewer simplification pass, and a test suite written specifically to
protect this module's invariants. Code review caught it.

## Guidance

**Admission must not correlate with the classifier's success, only with the
classifier's answer.** The earlier rule said the admission criterion must not
correlate with per-item difficulty. The generalisation this episode forces: that
includes *whether the stratifier managed to produce an answer at all*. A model
pass has two outputs — a label, and a success/failure — and only the first is a
judgement about the item's class. Routing on the second is a filter nobody
designed.

**Park failures, do not drop them.** The fix keeps an unclassified thread
eligible without inventing a stratum for it. `rough_classify` returns the ids it
could not classify (`src/triage/evals/pool.py:363`), and `stratified_select`
accepts them as `top_up_only` candidates — drawable during top-up, never counted
toward a quota (`src/triage/evals/pool.py:419-428`, `:472`). The thread can still
reach the eval set, where a human assigns the real label; it just cannot fill a
class floor on the strength of a guess that was never made.

The alternative — assigning a fallback stratum such as `General Inquiry` — was
rejected. It admits the thread but pollutes a specific class's support with items
that were never classified into it, which corrupts the quota accounting the
floors exist to guarantee. Parked-but-eligible keeps admission and stratification
independent, which is the property that was violated in the first place.

**Count the failures and surface them.** `pool_stats.json` records per-pass
failure counts and `rough_unclassified`, and the CLI prints the unclassified
count. A failure rate that is quietly zero and a failure rate that is quietly
fifteen percent look identical from the outside otherwise.

## Why This Matters

The intuition that makes dropping look safe: schema-enforced calls with two
retries fail rarely, and when they do it is transient — a timeout, a malformed
token stream. Random noise at a low rate does not bias a sample. Drop them.

The rate premise is probably right. The randomness premise is wrong, and it is
the only one that matters.

The three failure kinds are `malformed`, `off_taxonomy`, and `refusal`
(`src/triage/tools/schemas.py:63-66`). None of them is content-neutral:

- **Refusal** correlates directly with content. A support corpus contains abuse,
  threats, slurs, and self-harm adjacent messages. Those are exactly the threads
  a model declines to process — and exactly the threads that should be labelled
  `Trust & Safety` and `escalate=true`.
- **Off-taxonomy and malformed output** cluster on inputs that are hard to parse
  as a support request at all: non-English threads, heavy emoji or markup,
  fragmentary text, unusual formatting.

So the dropped set is not a random sample of the corpus. It is enriched for
abusive content, non-English content, and messy formatting — a systematic
subtraction from the pool, aimed at precisely the rare and difficult classes the
stratification exists to guarantee support for. `Trust & Safety` is the class
most likely to trigger a refusal and among the hardest to reach its floor
anyway.

From there the mechanism is identical to the keyword case. The pool loses its
hard instances. Both arms inflate. The baseline inflates at least as much,
because it had more headroom. The measured gap compresses. Per-class support
still meets the floor, because top-up rounds ran until it did. Nothing in the
eval output says "the abusive threads were removed before you started."

The failure is one-directional and quiet, in the same way and for the same
reason: it understates the pipeline's value, and understatement does not attract
the scrutiny that a surprising result does.

## The meta-lesson

The earlier document was written because a rejected proposal came back a second
time with a mitigation attached — the reasoning had not been legible enough. This
episode is the payoff and the limit of that write-up in one:

- **The payoff.** The rule was stated generally ("the admission rule must not
  correlate with the per-item difficulty the metric measures") rather than as its
  conclusion ("use a model, not keywords"). That generality is what made a
  reviewer able to recognise an unrelated-looking line of error handling as the
  same defect. A document that had only recorded *use a model instead of keywords*
  would have been fully complied with by the buggy code.
- **The limit.** Nobody applied the rule while writing the code. It was applied
  in review, by someone reading with the failure family in mind. Writing a
  learning down does not make the next instance self-evident, because the next
  instance does not look like the last one. This one arrived disguised as
  ordinary error handling — a `continue` in an exception branch, the least
  suspicious construct in the module.

The practical consequence: when a learning names a *class* of defect, the review
prompt for related work should name the class explicitly rather than trusting the
author to re-derive it. That is what surfaced this.

## When to Apply

**Any pipeline stage that both filters and can fail.** The trigger is the
conjunction. A stage that only filters gets scrutinised as a filter; a stage that
only fails gets scrutinised as reliability. When one component does both, the
failure path silently becomes a second filter with no design behind it.

**Error handling in selection code specifically.** `except: continue`,
`if result is None: skip`, `filter(lambda x: x.ok, items)` — inside a sampling,
enrichment, or admission step these are selection rules wearing the costume of
robustness. Ask what is correlated with the error, not just how often it fires.

**Any retry budget in front of a sample.** Exhausting retries is a decision to
exclude. If the retries can be exhausted more readily by one kind of content, the
budget is a content filter with a numeric threshold.

**Not applicable** where the failed item is genuinely unusable by every consumer
— the degenerate-thread exclusion at
`src/triage/evals/pool.py:screen_degenerate` passes this test, because a thread
with no diagnosable customer content is unanswerable for *both* arms rather than
merely hard for one, and its exclusion is counted per reason rather than silent.

## Related

- [keyword-stratification-biases-eval-pool.md](keyword-stratification-biases-eval-pool.md)
  — the rule this generalises. Read that one first; this is the second instance
  of its failure family, reached through error handling rather than through a
  cost-saving shortcut.
- `CONCEPTS.md` — **Stratifier**: "Its one real power is deciding which threads a
  human ever sees, which is why the rule it must satisfy is that its admission
  criterion must not correlate with the per-item difficulty the measurement is
  trying to detect."
- `src/triage/evals/pool.py` — `rough_classify` and `stratified_select` carry the
  parked-not-dropped behaviour; `pool_stats.json` carries the disclosure.
