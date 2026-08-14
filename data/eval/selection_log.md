# Eval set selection and labeling log (U6)

The record behind the README's methodology section. R11 requires the cleaning
rules to be stated rather than silent: the "real data" claim survives only if
what was filtered out is disclosed.

Sections marked **PENDING** are filled in as labeling proceeds.

## 1. Label set changes from the plan's indicative sets (R4)

The plan carried one six-label set described as indicative, serving both the
category and queue steps. U6 finalizes them as two disjoint vocabularies.
Shipped in `749a0db`.

### Change 1 — queue became its own vocabulary

`QUEUE_LABELS` was literally `CATEGORY_LABELS` (the same tuple object), and both
steps received the same label definitions, while route additionally saw
categorize's answer. Route could not be anything but a restatement of
categorize. The U5 pilot's queue==category rate of 7 of 8 threads measured that
design, not the data.

| Axis | Question it answers | Labels |
|---|---|---|
| category | What is the issue about? | Billing/Payments, Technical/Product, Shipping/Delivery, Account/Access, Complaint/Dispute, General Inquiry |
| queue | Which team should own it? | Tier-1 General, Billing Ops, Technical Support, Logistics, Account Security, Trust & Safety |

### Change 2 — Complaint/Escalation became Complaint/Dispute

The original label named escalation, which is already a separate field in the
gold schema (`thread_id,category,queue,escalate`) and a separate pipeline step.
A category naming escalation duplicates that judgment into the label space.
Renamed so escalation lives in the `escalate` field alone.

For the same reason the queue set contains no "Escalations" team: such a queue
would be near-deterministically `escalate=true`, rebuilding the duplication one
field over. Queue is ownership only; urgency is `escalate` only.

### Tie-break rule for Complaint/Dispute

`Complaint/Dispute` overlaps the functional categories — an angry billing
complaint matches both it and `Billing/Payments`. With a single annotator (R24)
a written rule is the only consistency defense:

> Use the specific functional category when the grievance is *about* a concrete
> billing, shipping, technical, or access issue. Use `Complaint/Dispute` only
> when the grievance itself is the subject and no functional category fits —
> general service dissatisfaction, or a threat to leave.

This rule is also embedded in the label's definition in
`src/triage/prompts/fragments.py`, so the model and the annotator work from the
same wording.

## 2. Cleaning rules (R11)

- **Degenerate threads are excluded.** A thread with no diagnosable customer
  content after cleaning — empty after stripping mentions and links, under three
  content words, or a pure deflect-to-DM exchange — is dropped from the
  candidate pool. `triage label` reports the excluded count on every run rather
  than filtering silently. The predicate is `degenerate_reason` in
  `src/triage/tools/retrieval.py`; degenerate threads remain fully supported at
  runtime via R28, they are simply not eval-set material.
- **Threads are the unit, not tweets.** Reconstruction follows R31: root-to-leaf
  branch containing the requested tweet plus direct brand replies, truncation
  and cycles flagged, sibling replies from other customers excluded.
- **Stratification.** Target: >=12 per class across queues and both escalation
  outcomes, at N=80 (the cut-ladder floor). Built by `triage pool` (U6, KTD8) in
  four stages, all seeded and reproducible from `pool_stats.json`'s manifest:

  1. *Structural prefilter* — one grouped SQL scan over the full corpus,
     class-neutral criteria only (a customer tweet exists; raw customer text is
     at least `MIN_DIAGNOSABLE_WORDS` characters). The floor is derived so it
     can only over-admit relative to `degenerate_reason`; SQL never becomes a
     second definition of degeneracy.
  2. *Seeded uniform sample* down to scan size. The volume cut is random, not a
     content rule — nothing about a thread's wording changes its odds of being
     scanned.
  3. *Degeneracy screen* using `degenerate_reason` itself, rejections counted
     per reason (see the table below when filled).
  4. *Stratifier* (CONCEPTS.md sense): rough classification on the DEV profile
     (`claude-haiku-4-5`, R30 — its spend is outside the measurement ceiling),
     three isolated per-field passes so the rough queue guess is never anchored
     on the rough category guess, then largest-deficit-first selection toward
     the floors. Rough labels are written to `pool_rough.csv`, never to the
     membership file the labeling path reads; the annotator cannot see a model
     guess.

  A thread whose rough classification fails after retries stays eligible with
  no stratum (top-up only, counted toward no quota). Dropping it would make
  admission depend on the model succeeding, and refusals correlate with
  content — the second instance of the selection-bias family recorded in
  `docs/solutions/tooling-decisions/stratifier-failure-drops-repeat-the-selection-bias.md`.

  **Disclosed residual risks, both bounded (this is the disclosure-is-correct
  case, not the disclosure-as-remedy error):**

  - *Rare-class stratum legibility.* Candidates for a class's quota come from
    threads the dev model itself predicted as that class. If the dev model has
    a systematic blind spot — subtle unauthorized-access threads it fails to
    read as Account Security — that stratum skews toward model-legible cases.
    Probabilistic rather than categorical (unlike the rejected keyword rule,
    every scanned thread is scored and misreads still enter the pool in another
    stratum), but not zero, and inherent to any imperfect stratifier short of
    labeling the whole corpus.
  - *Resume nondeterminism.* The LLM cache stores successful parses only, so an
    interrupted-and-resumed scan re-rolls previously failed calls and can reach
    a slightly different classified/unclassified split than an uninterrupted
    run. Seeds do not cover this. The realized failure counts below bound how
    much this could matter.

  **Achieved numbers** (from `pool_stats.json`, scan frozen 2026-08-14;
  rough passes on `claude-haiku-4-5`, the dev profile per R30):

  | Funnel stage | Threads |
  |---|---:|
  | Corpus | 901,648 |
  | Structurally eligible | 901,560 |
  | Seeded uniform sample | 4,000 |
  | Survived degeneracy screen | 3,944 |
  | Rough-classified | 3,944 |
  | Selected (target 80) | **80** |

  Per-criterion rejections: 87 `no_customer_tweet`, 1
  `customer_text_below_floor` (structural prefilter); 53
  `degenerate:below_word_minimum`, 3 `degenerate:empty_after_cleaning`
  (degeneracy screen); 897,560 `not_sampled` (the seeded volume cut); 3,864
  `not_selected_by_stratification`.

  **Rough-classification failures after retries: 0** — no thread entered the
  no-stratum top-up path, and the resume-nondeterminism risk above is bounded
  at zero realized failures. (The scan was interrupted twice — a process
  death on 2026-08-13 and API-credit exhaustion on 2026-08-14 — and resumed
  from the call cache both times.)

  Achieved support per bucket (floor >= 12), no shortfalls:

  | Bucket | Support |
  |---|---:|
  | category: Account/Access | 13 |
  | category: Billing/Payments | 12 |
  | category: Complaint/Dispute | 16 |
  | category: General Inquiry | 12 |
  | category: Shipping/Delivery | 12 |
  | category: Technical/Product | 15 |
  | queue: Account Security | 12 |
  | queue: Billing Ops | 13 |
  | queue: Logistics | 16 |
  | queue: Technical Support | 14 |
  | queue: Tier-1 General | 13 |
  | queue: Trust & Safety | 12 |
  | escalate: true | 64 |
  | escalate: false | 16 |

  The escalation skew (64/16) is the dev model's rough guess over
  quota-driven picks, not a claim about the corpus; the gold labels decide.
  Top-up rounds during labeling: none yet — recorded here if labeling
  triggers any.

## 3. Labeling protocol — three independent passes

One annotator labels all three fields (R24). Labeling them together anchors each
field on the previous one: queue drifts toward the team whose name echoes the
category just chosen, and the divergent cases the taxonomy split exists to
capture get under-labeled. That is the same collapse removed from the pipeline,
reappearing in the gold labels — where it is invisible, because the labels are
what everything else is measured against.

So each field is a separate full sweep over the whole set, and the isolation is
structural rather than a display choice:

- `LabelItem` carries only `(thread_id, text)`. No field exists on it for a
  label, so no prior answer has an in-memory channel into a later pass.
- Each pass reads and writes exactly one file, derived from the pass itself. No
  function accepts another pass's path.
- `merge_passes` is the only function that opens more than one pass file, and it
  refuses unless all three passes cover exactly the same thread ids.

`tests/test_label_helper.py::TestPassIsolation::test_a_pass_opens_only_its_own_file`
instruments `builtins.open` and asserts a pass touches nothing else. It was
verified to fail when a cross-pass read is deliberately injected, so the
guarantee is tested rather than asserted.

**No model pre-classification is ever shown during labeling.** A dev-model pass
is used only to stratify the candidate pool (KTD8); displaying its guess would
be the strongest available anchor and would reduce the eval to measuring the
model's agreement with itself.

### Pass-1 redo (2026-08-14)

The first category sweep was discarded and redone in full. A post-pass check
found its distribution collapsed into the two catch-all labels (41 General
Inquiry + 23 Complaint/Dispute of 80, zero Account/Access) with at least one
confirmed slip, a pattern consistent with mood-based labeling and menu
mistypes rather than topic judgments. The redo followed
`labeling_cheatsheet.md` (definitions copied verbatim from the single-sourced
`fragments.py`). The discarded sweep is kept at `pass_category.attempt1.csv`;
no per-thread disagreement list was shown to the annotator — the check used
only aggregate counts plus one thread already discussed in the open, so the
redo stays anchored to definitions, not to model guesses.

A second start (`pass_category.attempt2.csv`, 10 threads) was also set aside:
its labels agreed with the annotator's own first-attempt labels on only 2 of
the 10 shared threads, a test-retest instability that fails the reliability
bar regardless of any model comparison. Labeling resumes fresh when it can be
given unhurried attention; both discarded files stay in the repo as the
honest record.

### Pass order seeds

Each pass shuffles under its own fixed seed, so by the second sweep threads
arrive in an order that defeats recall of the first sweep's answer. Published
from `label_helper.manifest()` so a reader can rerun the ordering and check it.

| Pass | Seed | Output file |
|---|---|---|
| category | 20260812 | `pass_category.csv` |
| queue | 778301 | `pass_queue.csv` |
| escalate | 4419907 | `pass_escalate.csv` |

Commands:

```
triage label --pass category    # then queue, then escalate — each a full sweep
triage label --pass queue
triage label --pass escalate
triage label --merge            # writes gold_labels.csv
```

## 4. Measurement this protocol enables

Because the passes are independent, the **category/queue divergence rate in the
human labels** is measurable. That is the honest ceiling on what the route step
can contribute: if the annotator diverges on only 5% of threads, route cannot
add more than that, and the README reports the number instead of the pipeline
quietly collapsing for unexplained reasons. Record the observed rate here once
the merge completes.

**PENDING — observed divergence rate.**

## 5. Judge subsample and anchor stock (R14)

- **PENDING — agreement subsample:** 20-30 drafts scored by hand with a one-line
  critique per score. Used only for judge-agreement computation.
- **PENDING — anchor stock:** a separate held-out set of labeled drafts supplying
  the judge's few-shot anchors. Drawn from outside the eval set's 80-150 range,
  never carved from the agreement subsample, never counted in agreement.

## 6. Freeze

**PENDING.** `gold_labels.csv` must be committed (frozen) before any measured run
begins; git history is the evidence for that ordering.
