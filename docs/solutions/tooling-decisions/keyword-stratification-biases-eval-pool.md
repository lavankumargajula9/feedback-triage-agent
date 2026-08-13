---
title: Eval-pool selection rules must not correlate with the difficulty being measured
date: 2026-08-12
category: tooling-decisions
module: triage.evals
problem_type: tooling_decision
component: testing_framework
severity: high
applies_when:
  - Stratifying a sample by a predicted class rather than a known ground-truth label
  - Swapping a model pass for regex or keyword matching to save API spend
  - Both arms of a comparison draw their instances from the same candidate pool
  - A sampling shortcut's known bias would be disclosed in a log rather than removed
  - The headline result is a gap between two arms rather than one absolute score
tags:
  - eval-set
  - stratified-sampling
  - construct-validity
  - sampling-bias
  - measurement
  - cost-vs-validity
---

# Eval-pool selection rules must not correlate with the difficulty being measured

## Context

`feedback-triage-agent` exists to support a single quantitative claim: that a multi-step
triage pipeline produces better classifications and better draft replies than a
single-prompt baseline given the same information. Everything else in the repository — the
LangGraph pipeline, the prompt-fragment single-sourcing, the SQLite call cache — is
scaffolding around that one measured number.

The eval set is drawn from a large real corpus rather than authored.
`docs/data-verification.md:24` records the ingest as **2,811,774 tweets into 901,648
threads across 108 distinct brand accounts**. That is far more than can be hand-labeled and
the class distribution in the wild is heavily skewed, so the eval set cannot be a uniform
random draw. It has to be stratified.

That is what the stratifier is for. KTD8
(`docs/plans/2026-08-11-001-feat-feedback-triage-agent-plan.md:194`) spells out the recipe:
"Rough-classify a candidate pool (dev model), stratified-sample toward ≥12 per class, label
in rounds until support is met, log per-criterion rejection counts for the README
methodology section, and freeze gold labels in a commit before the first measured run."
`data/eval/selection_log.md:66-68` states the target concretely: ">=12 per class across
queues and both escalation outcomes, at N=80 (the cut-ladder floor)."

Stratification was not in the original design — it entered as a round-1 `ce-doc-review`
finding that R11 should carry explicit exclusion and stratification criteria (session
history). That origin matters: the step exists *because* someone asked how the eval set
avoids being quietly unrepresentative.

**Two checks at two different times, and this is the part that makes the stratifier feel
low-stakes when it is not.** The stratifier only *predicts*. The ≥12 floor is re-checked
against the final *human* labels after merge, with top-up rounds until it is met. Because
the second check is honest, it is tempting to conclude the first one can be sloppy — a bad
stratifier just means more top-up rounds. That inference is wrong in a specific way, and
this document is about why.

The stratifier is also firewalled from the annotator. `data/eval/selection_log.md:94-97`:
"**No model pre-classification is ever shown during labeling.** A dev-model pass is used
only to stratify the candidate pool (KTD8); displaying its guess would be the strongest
available anchor and would reduce the eval to measuring the model's agreement with itself."
Its only power is over pool composition — which turns out to be enough power to matter.

The friction that produced this decision was budget. The project runs under a $100
stop-condition ceiling; the itemized estimate at plan line 176 puts all measured runs at
"≈ $61 against the $100 stop-condition ceiling," and spend to date is $0.15
(`docs/u5-pilot-note.md:70`). Someone scanning for savings noticed the stratifier makes one
LLM call per candidate thread and proposed replacing it with literal pattern matching.

Two implementations were on the table:

- **(a) The dev-model stratifier.** A classification pass on `claude-haiku-4-5` — the DEV
  profile model at `src/triage/config.py:61-65`, whose comment reads "never used for
  measured results" — predicting each thread's class from the opening customer message
  alone. Roughly 200 tokens in and 10 out per call. At the Haiku 4.5 prices in
  `src/triage/evals/cache.py:46-50` that is about $0.00025 per call, so a 4,000-thread scan
  costs about one dollar.
- **(b) A regex/keyword stratifier.** Literal pattern matching against per-class keyword
  lists — the concrete proposal was `refund|charge` → billing, `login|password` → account,
  all-caps or profanity or "supervisor" → escalation signal, plus thread length and brand
  (session history). No model call at all. Proposed for exactly one reason: to save that
  ~$1-2.

Option (b) was proposed, rejected, then **reopened a second time with a mitigation
attached** — a "disclosed limitation" paragraph to be written into
`data/eval/selection_log.md` — and was rejected again and settled on 2026-08-12. The
disclosure paragraph was reverted; the working tree is clean at `e58bee4` and the
stratification entry in the selection log is again the plain PENDING note at lines 66-68.
(This repository is local-only with no remote and no pull requests, so commits are
referenced by SHA and no merge-state claim is made.)

The rule is written down because the proposal came back a second time with a
plausible-sounding fix. A decision rejected once may be a judgment call. A decision that has
to be rejected twice, the second time against a mitigation that sounds responsible, is a
decision whose *reason* was not legible. That is what this document fixes.

## Guidance

**When selecting items for an eval set, the admission rule must not correlate with the
per-item difficulty that the metric is supposed to measure.** The question to ask of a
candidate selection mechanism is not "how much of the population does this admit?" but
"does the thing that gets an item admitted also make that item easier for the systems under
test?" If yes, the mechanism is a construct-validity defect regardless of how cheap, fast,
or simple it is.

Three concrete instructions follow.

**Use the dev-model stratifier, not a keyword stratifier.** A model pass predicts from the
whole opening message — its phrasing, its implied situation, its affect — and therefore
admits threads that belong to a class without containing that class's vocabulary. A keyword
rule can only admit threads that say the expected words.

**When cost pressure hits a method, cut the payload before you cut the method.** This is the
constructive half of the lesson and the thing that actually resolved the episode. The
stratifier does not need the full thread — it needs the opening customer message and a bare
label back:

| Shape | Tokens | Cost/call |
|---|---|---|
| Full thread (as originally budgeted) | ~1,500 in / ~300 out | ~$0.003 |
| Opening message, label only | ~200 in / ~10 out | ~$0.00025 |

Twelve times cheaper for the *same method*, which bought a **deeper scan — 4,000 candidate
threads instead of 1,000** — with no loss of selection validity (session history). The
budget objection was real; it just had a better answer than the one first proposed. Look for
that answer before trading away validity.

**Disclosure is not a remedy for a selection rule that compresses the measured effect.**
Writing "the candidate pool was built by keyword matching, which may bias toward lexically
obvious cases" into `data/eval/selection_log.md` does not repair the number; it annotates a
wrong number. Disclosure is the correct treatment for a *known, bounded* limitation — a fact
a reader can reason about and adjust for. It is not a treatment for systematic deflation of
the headline metric, because the reader cannot invert it: nobody, including the author,
knows how much lift was erased.

The distinction is visible in what this repository already discloses correctly. R11 (plan
line 89) requires cleaning rules to be stated because "the raw-data claim survives only if
filtering is stated, not silent." `docs/data-verification.md:44` discloses that 4,387 of
901,648 threads (0.49%) are truncated, and that cycle-flagged threads number zero with the
instruction not to claim the cycle guard as a handled real-world case. Section 4 of the
selection log (`data/eval/selection_log.md:120-129`) discloses that the three-pass protocol
makes the category/queue divergence rate measurable — "the honest ceiling on what the route
step can contribute." Each names a bound the reader can price in. A selection rule that
makes both arms look better is categorically different, and reaching for the disclosure
hammer on it is the error this document names.

## Why This Matters

The intuition that makes a keyword stratifier feel safe: the stratifier only builds the
*candidate* pool; humans label everything afterwards; the ≥12 floor is enforced against
those human labels; therefore any stratifier error shows up as extra top-up rounds, not as a
wrong number. Every clause is true and the conclusion still does not follow.

A keyword stratifier admits a thread into class *C*'s stratum when the thread contains class
*C*'s vocabulary. That is its entire criterion. So the threads reaching the pool for class
*C* are systematically the ones where the class is **lexically obvious** — where the
customer used the word a taxonomy designer would have used. Threads that belong to *C* but
describe the situation in other words are never admitted, and no amount of downstream human
labeling recovers them, because a human never sees them. The labeling pass is honest about
the threads it is given and has no visibility into the threads it was never given. The ≥12
floor gets met, on schedule, entirely with easy instances. Every check passes.

Now consider what a pool of lexically obvious instances does to the comparison. The
pipeline's whole claim to value is that decomposition helps where the right answer is *not*
obvious from the surface of the text — where you must work out what the customer's situation
actually is before categorizing, routing, and deciding escalation. On a thread whose class
is announced by its own vocabulary, that work is unnecessary. The baseline gets it right in
one shot. The pipeline also gets it right, at four times the cost.

So a lexically-easy pool inflates **both arms**, and inflates the baseline *at least as
much*, because the baseline had more headroom to gain — the pipeline was already near
ceiling on easy cases. The measured gap is the difference of two inflated numbers, and it
compresses toward zero. The eval reports a small lift, or no lift, for a pipeline that would
have shown a real lift on a representative pool.

That gap is the deliverable. Spending a dollar less on pool construction in exchange for a
smaller measured effect is not a saving; it is paying a dollar less to buy a worse answer to
the only question being asked.

The failure is also **undetectable from the eval output alone**. Per-class support meets the
floor, because top-up rounds ran until it did. Per-class accuracy tables look healthy. The
paired bootstrap confidence intervals are computed correctly and are honestly narrow — the
resampling faithfully describes the variance *of the pool it was given* and says nothing
about whether that was the right pool. Judge agreement is unaffected; the judge scores the
drafts it is shown. Nothing in the artifact says "these threads were easy." A reader —
including the author six months later — sees a well-instrumented eval reporting a modest
lift and concludes the pipeline is not worth much.

That asymmetry is the deepest reason for the rule. Most measurement errors are two-sided:
they add noise, or inflate a number a skeptic can challenge, or show up as an anomaly
someone investigates. This one is one-sided and quiet. It **understates** the pipeline's
value, and understatement does not attract scrutiny the way overstatement does. Nobody
audits a disappointing result as hard as a surprising one. A selection rule that silently
shrinks your effect can end a project's central claim without ever announcing itself.

And it cannot be fixed after the fact. Once `gold_labels.csv` is frozen — which KTD8
requires before the first measured run, with git history as the evidence of ordering — the
pool is the pool. Re-selecting means re-labeling, which is the schedule's critical path and
the author's own hand-labeling time.

## When to Apply

**Any A/B or lift measurement.** When the output is a *difference* rather than an absolute
level, the difference is more fragile than either level. Anything shifting the item mix
toward the easy end compresses it even when both absolute numbers look fine, and the
mechanism is invisible in per-arm metrics.

**Any sampling, filtering, deduplication, or stratification step upstream of a comparison.**
The trigger is structural: if a step decides *which items* reach the measurement, it is in
scope however mundane it looks. Candidate-pool construction is the obvious case, but so are
a "skip threads longer than N tokens" guard, a language filter, a dedupe pass keyed on
near-duplicate text, a "drop items the parser failed on" step, and a cache-warming pass that
quietly determines which items get evaluated. Each has a defensible operational reason and
each can correlate with difficulty: long threads are usually harder, parse failures cluster
on messy inputs, near-duplicates cluster on templated easy traffic.

The exclusions this repository already documents at `data/eval/selection_log.md:56-62` —
degenerate threads with no diagnosable content — are in scope by this test and **pass** it,
because "has no diagnosable customer content" excludes items unanswerable for *both* arms
rather than easy for both, and the excluded count is reported on every `triage label` run
rather than filtered silently.

**Whenever a cost lever touches sample selection rather than payload size or model tier.**
The most useful trigger in practice, because cost pressure generates these proposals and
they arrive framed as engineering, not methodology. When someone says "we can save money
here," classify *which* lever before evaluating the saving.

**When a rejected proposal returns with a disclosure paragraph attached.** Treat
mitigation-by-documentation as a signal to re-examine the original objection, not as a
resolution of it. The sort is: *does the limitation change the number, or does it change how
to read the number?*

The rule does **not** apply to a filter whose admission criterion is orthogonal to
difficulty — sampling by date range, by brand account, by thread ID hash, or an even random
draw. Those cost sample size, a known and priceable loss, and this repository already prices
it: the cut ladder permits shrinking the eval set toward the floor precisely because uniform
shrinkage is a legible trade, pinning a hard minimum around 80 "below which precision/recall
stops being meaningful."

## Examples

### The two stratifiers on one thread

Consider a thread of the kind `docs/data-verification.md` describes as having "specific,
non-templated grievances":

> "someone in another state has been ordering things through your app on my card since
> tuesday and i never gave anyone my details, i can see the orders in my history and i
> cannot get anyone to call me back"

Ground truth: queue is Account Security, category is Account/Access, `escalate` is true.
This is exactly the sort of thread the eval set most needs, because it is where a decomposed
pipeline should beat a single-shot answer — the right routing depends on working out that
unauthorized access has occurred, which the customer never says.

A keyword stratifier for Account Security would match `hacked`, `password`, `2FA`, `locked
out`, `compromised`, `phishing`, `unauthorized`. This thread contains none of them. It is
never admitted, never labeled, never in the eval set.

A `claude-haiku-4-5` pass over the opening message predicts Account Security from the
situation rather than the vocabulary and admits it to the pool, where a human then applies
the real label.

The same keyword rule fails in the opposite direction, which compounds it. A thread reading
"the password for the wifi in your store doesn't work" contains `password` and would be
admitted to the Account Security stratum, where a human correctly labels it Technical
Support. That cell of the pool ends up simultaneously **thinner in the hard cases it most
needs** and **padded with easy false positives a top-up round has to churn through**. The
floor still gets met. The pool is still wrong.

Note where the two checks land. The stratifier's mistake happens at pool-construction time
and is silent. The ≥12 floor is re-checked later against merged human labels and passes,
because it counts labeled threads per class and says nothing about how they were found. The
honest second check cannot see the dishonest first one.

### The cost arithmetic

Prices are on file at `src/triage/evals/cache.py:46-50`. Stratifier call on
`claude-haiku-4-5` at $1.00 in / $5.00 out per million tokens, at ~200 input and ~10 output
tokens:

- input: 200 × $1.00 / 1,000,000 = $0.00020
- output: 10 × $5.00 / 1,000,000 = $0.00005
- **per call: $0.00025**

A 4,000-thread scan is **$1.00**. Doubling the pool to 8,000 costs $2.00. That is the entire
saving the keyword stratifier was proposed to capture.

Against that: the plan's estimate for all measured runs is ≈$61 against a $100 ceiling, and
plan line 176 adds the decisive qualifier — **"dev iteration excluded by R30 (Haiku
4.5/mocks)."** The stratifier runs on the DEV profile. Its spend is therefore *outside* the
stop-condition ceiling by design. The trade being offered was: accept an unmeasurable,
one-directional deflation of the project's headline metric in order to save one dollar that
was never being counted against the budget in the first place.

Writing the arithmetic out is what makes the answer obvious, and that is a general technique.
"Save an LLM call per item" sounds like disciplined engineering in the abstract. "Save one
dollar of uncounted spend" does not survive contact with "at the cost of the number the
repository exists to produce." Price the lever before debating the principle; the price often
settles the debate without needing the principle.

### The cost-lever taxonomy

| Lever | What it changes | Safe to cut? | In this repo |
|---|---|---|---|
| **Payload size** | Tokens per call | Yes, freely — subject to keeping both arms informationally equal | The stratifier reads the opening message only; KTD7 single-sources prompt fragments so trimming applies identically to both arms |
| **Model tier** | Which model runs a role | Yes, for anything unreported | The DEV/MEASUREMENT split at `src/triage/config.py:58-73` is exactly this lever |
| **Sample size** | How many items are measured | Only with a floor, and disclose it | The cut ladder permits shrinking the eval set first, pinning a hard minimum around 80 |
| **Sample selection** | Which items are measured | **No** — this is where validity lives | The keyword stratifier, rejected twice and settled 2026-08-12 |

The first two rows change how each item is processed and leave the item mix alone; a mistake
there is visible, because it shows up as degraded output on items you can inspect. The last
two change the population; a mistake there is invisible, because the items it damages are the
ones not in front of you. Caching belongs to the safe family, which is why
`src/triage/evals/cache.py` is the right place to have spent effort on cost: it makes
repeated evaluation nearly free without changing which threads are evaluated or how they are
scored, and it tracks the difference honestly — a replayed call adds zeros to the money
fields (`src/triage/evals/cache.py:257`) so reported totals describe money actually spent.

### The disclosure that would not have worked

The second proposal's mitigation was a paragraph along the lines of: *"The candidate pool was
stratified by keyword matching rather than a model pass. This may under-represent threads
whose class is not lexically explicit."*

The same file discloses correctly elsewhere — the divergence-rate note at
`data/eval/selection_log.md:120-129` names a bound, gives the reader a number, and makes the
result more interpretable. The keyword disclosure has none of those properties. It names no
bound, gives no quantity, and critically the reader cannot invert it — there is no way to
look at a reported lift of, say, four points and recover what it would have been on an
unbiased pool. It would have converted "we published a wrong number" into "we published a
wrong number and said so," which is more honest but not more correct, and correctness is what
a measurement is for.

**The test to carry forward: if a reader who fully believes your disclosure still cannot
recover the right number, the disclosure is not a mitigation — it is a confession, and you
should fix the method instead.**

This project had already established that standard one layer down, hours earlier. When the
three-pass labeling protocol was designed, a *display-level* fix — simply not showing pass 1's
answer during pass 2 — was rejected in favour of structural isolation: separate output files
per pass until final merge, so later passes are "structurally unable to access" earlier ones
rather than merely not shown them (session history; the protocol is recorded at
`data/eval/selection_log.md:94-97`). A bias you have written down is still in the sample, for
the same reason an answer you have merely hidden is still in the session.

## Related

- [stratifier-failure-drops-repeat-the-selection-bias.md](stratifier-failure-drops-repeat-the-selection-bias.md)
  — the second instance of this failure family, found in the shipped stratifier on
  2026-08-13. The dev-model pass was chosen as this document recommends, and then dropped
  every thread whose classification call failed after retries. Refusals and malformed
  output correlate with abusive, non-English, and messy content, so admission again
  correlated with difficulty — this time through error handling rather than vocabulary.
  Evidence that stating the general rule, rather than the "use a model" conclusion, is what
  made the second instance recognisable.
- `data/eval/selection_log.md` — the operational record this rule governs. Section 2's
  stratification entry is still PENDING; it should state the dev-model rule when filled in.
- `docs/plans/2026-08-11-001-feat-feedback-triage-agent-plan.md` — KTD8 (line 194) is the
  settled construction recipe, R11 (line 89) the requirement it governs, R30 (line 100) the
  dev/measurement decoupling that puts stratifier spend outside the ceiling, and line 176 the
  itemized cost estimate.
- `docs/data-verification.md` — the sibling claim-integrity record; the same reasoning family
  (a claim survives only if its construction is honest) applied to the corpus rather than the
  sample.
