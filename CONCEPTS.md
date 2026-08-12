# Concepts

Shared domain vocabulary for this project — entities, named processes, and status concepts
with project-specific meaning. Seeded with core domain vocabulary, then accretes as
ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only,
not a spec or catch-all.

This first seed covers the **measurement and eval-set area**. Terms from other areas of the
project (ingestion internals, the MCP surface, the pipeline's node structure) are not yet
defined here and are candidates for a later pass.

## Corpus and threads

### Thread
One customer's conversation with one brand, reconstructed from individual tweets by walking
reply parentage. A thread is the unit everything else operates on: the unit that gets
classified, labeled, retrieved, and scored.

A thread's identity is the pair (root tweet, customer), not a stored surrogate — so a thread
whose root shifts, because a previously-missing parent tweet arrives in a later ingest, is a
*different* thread by identity even though it describes the same conversation. This is why
re-ingesting is guarded rather than free once labels exist. Two threads can legitimately
share tweets: a brand reply that several customers respond to belongs to each of their
threads.

### Degenerate thread
A thread with no diagnosable customer content — an immediate deflect-to-DM, an empty
after-cleaning shell, a lone tweet with no grievance. Degenerate threads are excluded from
the eval set by rule, and the exclusion count is reported rather than applied silently,
because they are unanswerable for every system under test rather than easy for all of them.

## Taxonomy

### Category
What the customer's problem *is* — the subject-matter classification of a thread.

### Queue
Which internal team *owns* the thread. Deliberately a disjoint vocabulary from Category:
no label is valid for both, and that disjointness is asserted structurally rather than
maintained by convention. Category and Queue answer different questions and a thread's
answers to them need not correspond.

### Escalation
Whether a thread requires handling beyond the routine queue path. A binary outcome
classified as its own step, and one the eval set is stratified across so that both outcomes
carry real support.

### Divergence rate
How often a thread's Category and Queue disagree in the *human* labels. It is the honest
ceiling on what the routing step can contribute: if a human assigns a queue that the
category alone would not have predicted only rarely, then routing cannot add more value than
that, and the number is reported rather than left to look like unexplained pipeline
underperformance.

## Measurement

### Baseline arm
The single-prompt system: one model call given the same information the pipeline receives,
producing the same outputs. It exists to be beaten, and its informational parity with the
pipeline is structural — both draw their prompt text from one shared source — so a measured
gap cannot be an artifact of one arm simply knowing more.

### Pipeline arm
The multi-step system: separate classification, routing, escalation, and drafting steps.

### Measured lift
The difference between the pipeline arm's scores and the baseline arm's scores. This is the
project's deliverable — not either arm's absolute score. Because it is a difference, it is
more fragile than either level it is computed from, and anything that shifts the eval set's
item mix toward easier instances compresses it while leaving both absolute numbers looking
healthy.

### Judge
A third model that scores both arms' drafts against a rubric. Required to be a different
model from the one that generated the drafts, so a system is never grading its own output,
and required to be comparably capable, so that removing self-preference bias does not
introduce a capability-mismatch confound in its place. The constraint is asserted at load
time rather than left to convention.

### Variance pass
A repeat run of the same arm over the same items, used to observe genuine run-to-run spread.
Its purpose is to derive the noise tolerance the regression check gates on, since the
determinism levers that older model generations offered are no longer available — spread is
measured rather than suppressed.

### Reference run
The recorded artifact a later run is compared against. It pins the environment that produced
the numbers — model identities, prompt fingerprints, and the eval set's own fingerprint —
alongside per-thread outputs. A new one is recorded only by explicit command, never as a side
effect of running an eval.

### Environment drift
The condition where a current run's pinned environment no longer matches the reference run's.
Reported as its own distinct warning rather than as a metric failure, because a changed
prompt or model invalidates the comparison itself — the numbers are no longer describing the
same system, which is a different problem from a metric getting worse.

## Eval-set construction

### Eval set
The hand-labeled set of threads every reported number is computed over. Small enough to label
by hand, deliberately not a uniform random draw from the corpus, and frozen before the first
measured run so that the labels cannot drift under the results.

### Candidate pool
The over-sampled set of threads a human is asked to label from. Larger than the eval set,
because labeling rounds continue until per-class support is met and some candidates are
rejected on inspection.

### Stratifier
The pass that predicts each candidate thread's class so the pool can be sampled toward the
per-class floor rather than inheriting the corpus's natural skew.

The stratifier only *predicts* — it never labels. Its predictions are structurally hidden
from the annotator, because showing a model's guess would be the strongest available anchor
and would reduce the eval to measuring the model's agreement with itself. Its one real power
is deciding which threads a human ever sees, which is why the rule it must satisfy is that
its admission criterion must not correlate with the per-item difficulty the measurement is
trying to detect.

### Per-class floor
The minimum number of labeled threads required for each class, across queues and both
escalation outcomes, below which per-class precision and recall stop being meaningful.

Checked twice at two different times, and the distinction matters: the stratifier's
predictions are checked against the floor when the pool is built, and the *human* labels are
re-checked against it after merge. A class that cannot reach its floor is reported as a real
data-sparsity finding with a non-zero exit, never quietly backfilled with weaker matches —
"this class is genuinely rare in this corpus" is a result, not a defect.

### Gold label
A human-assigned, authoritative label for a thread. Gold labels are frozen in version control
before the first measured run, so the ordering — labels first, numbers second — is evidenced
rather than asserted.

### Three-pass labeling
The protocol for producing gold labels: category, queue, and escalation are labeled in
separate passes rather than all three per thread in one sitting.

The isolation is structural, not presentational — later passes are unable to reach earlier
passes' answers, rather than merely not shown them. Labeling all three fields per thread in
order would anchor the queue toward whichever team the category implies, reproducing in the
human labels the exact collapse the taxonomy split was made to remove.

## Cost and model roles

### Dev profile
The model-role assignment used for all iteration, debugging, and pipeline development. Cheap
by design and never used for any number that ships, which is what lets development spend sit
outside the project's cost ceiling entirely.

### Measurement profile
The model-role assignment whose outputs are reported. Kept deliberately separate from the dev
profile and centralized in one place, so that switching what gets measured is a visible
decision rather than a silent default; tests pin the exact assignment for the same reason.

### Cut ladder
The pre-agreed order in which scope is surrendered if the schedule slips, decided before the
pressure arrives rather than during it. It names both what gets cut first and what is
protected from cutting at all, and it carries a hard floor on eval-set size below which the
measurement stops meaning anything.

### Cost lever
A way of reducing spend, classified by what it changes. Levers that change how each item is
processed — payload size, model tier, caching — are safe to pull freely. Levers that change
*which* items are measured — sample size, sample selection — are where validity lives, and a
saving from that family is never evaluated on price alone.

## Plan vocabulary

### Requirement
A numbered statement of what the system must do, governing behavior. Requirements outrank
everything else: a technical decision may choose the mechanism, but only within the
requirement it cites.

### Key Technical Decision
A numbered choice of mechanism, each one citing the requirements it governs. Decisions cannot
override requirements, and implementation units override neither.

## Flagged ambiguities

- **Category and Queue** were once the same list — the routing step was handed the category
  vocabulary and asked to pick a team from it, which made a high queue-equals-category rate
  a measurement of the taxonomy's design rather than a finding about the data. They are now
  disjoint vocabularies and the disjointness is asserted in code.
- **"Stratify" never means "label."** The stratifier's output is a prediction used only to
  shape the candidate pool; the gold label is always human. Text that reads as though the
  stratifier assigns labels is wrong.
- **Disclosure versus remedy.** Writing a limitation down is the correct treatment for a
  bounded fact a reader can price in, and is *not* a treatment for a method that biases the
  reported number. If a reader who fully believes the disclosure still cannot recover the
  right number, the method needs fixing rather than documenting.
