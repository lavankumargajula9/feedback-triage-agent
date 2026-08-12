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
- **PENDING — stratification.** Target: >=12 per class across queues and both
  escalation outcomes, at N=80 (the cut-ladder floor). Record the achieved
  per-class support and any top-up rounds here.

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
