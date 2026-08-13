---
title: The LLM cache is single-threaded by construction, so batch passes are serial
date: 2026-08-13
category: tooling-decisions
module: triage.evals
problem_type: tooling_decision
component: llm_cache
severity: medium
applies_when:
  - A batch of independent LLM calls is running serially and the wall clock hurts
  - Someone proposes a thread pool over calls that go through CachingClient
  - Estimating how long a scan, eval run, or variance pass will take
  - Deciding whether to wait out a long run or parallelise it first
tags:
  - llm-cache
  - concurrency
  - sqlite
  - performance
  - batch-execution
---

# The LLM cache is single-threaded by construction, so batch passes are serial

## Context

Every LLM call in this system goes through `CachingClient`
(`src/triage/evals/cache.py`), which sits at the wrapper seam so both eval arms,
the pipeline steps, and the pool stratifier share one cache. The module says so
plainly at line 19: the backoff is "deliberately simple, no concurrency
machinery (per the plan)."

That reads like a style note. It is a hard constraint, and it sets the wall
clock for every batch operation in the project.

The concrete instance: the candidate-pool stratifier makes three isolated calls
per scanned thread. On the real corpus that is **11,832 calls at ~0.9 s each,
about 2.8 hours**, on an embarrassingly parallel workload where bounded
concurrency of 8–16 would cut it to minutes. The question "why not just thread
it?" has a specific answer worth writing down, because it will be asked again of
the eval runner and the variance passes, which are larger.

## Guidance

**Do not add a thread pool over `CachingClient` without changing `CallCache`
first.** It will not merely be slow or racy — it raises immediately:

- `CallCache.__init__` does `sqlite3.connect(path)` with no
  `check_same_thread=False` (`src/triage/evals/cache.py:139`). The stdlib default
  is `True`, so the connection is legal only on the thread that created it. A
  worker thread touching `get`/`put`/`delete` raises
  `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used
  in that same thread` on the first cache access. This is a crash, not a subtle
  race.
- Independently, the usage tally is unguarded. `_record_usage`
  (`src/triage/evals/cache.py:248-265`) does read-modify-write on plain dict
  fields (`self.usage["calls"] += 1` and friends). Concurrent workers lose
  updates, so the reported call count, token count, and dollar total silently
  under-report. For a project whose budget discipline is a stated requirement,
  quietly wrong spend figures are worse than a slow run.

**The minimal safe change, if it is ever worth making:**

1. `sqlite3.connect(path, check_same_thread=False)`.
2. A `threading.Lock` around the bodies of `get`, `put`, and `delete` — held for
   the SQLite call and commit only.
3. In `_CachingMessages.parse`, release that lock before the network call and
   re-acquire only for the `put` and the usage update. Holding it across the
   request would serialise the very thing being parallelised.
4. Guard `_record_usage`'s increments with the same lock.

Small and mechanical. The reason it has not been done is not difficulty.

**Prefer waiting to parallelising, unless the run is on the critical path.**
`CallCache` is shared with the eval runner and the regression check — the
machinery that produces the numbers the project exists to publish, currently
covered by 363 passing tests. Introducing concurrency there means re-verifying
cache-key salting (`run_id` for variance passes), R32 crash-safety (the
commit-per-write that makes a run resumable), and R27 retry semantics under
concurrent access. That is a real verification burden to shave hours off a
one-time job.

**Budget the wall clock instead.** Before starting a batch pass, multiply calls
by ~0.9 s and decide whether to start it now. A 2.8-hour job started in the
morning is free; the same job started at the end of a session is a lost day.

## Why This Matters

The trap is that the serial constraint is invisible at the call site. Nothing in
`call_with_schema` or in a step function hints that the client underneath cannot
be shared across threads. A future contributor sizing up a slow batch sees a loop
over independent items — the textbook `ThreadPoolExecutor` shape — and has no
signal that the obvious change crashes on the first cache hit.

There is a second, subtler cost. Because the constraint is undocumented at the
point of use, the *reason* a long run is long gets rediscovered each time, and
the natural response to "this is taking 2.8 hours" is to reach for the method
rather than the schedule — trimming the scan size, cutting a pass, or sampling
less. Those are exactly the levers that damage measurement validity, as the
sibling learning on
[eval-pool selection](keyword-stratification-biases-eval-pool.md) records at
length. Knowing that the slowness is a fixed, deliberate property of the cache —
and not a signal that the method is too expensive — keeps the pressure off the
part of the system where cutting corners is unrecoverable.

## When to Apply

**Before proposing concurrency anywhere downstream of `CachingClient`.** That is
the pool stratifier, the eval runner's per-thread loop, the judge pass, and the
variance passes.

**When estimating a run.** Serial is the planning assumption. There is no
concurrency to discover.

**When a batch is interrupted.** Serial execution plus commit-per-write means a
killed run keeps everything it had cached; a rerun replays those and continues.
Note the one gap: only *successful* parses are stored, so a resumed run re-rolls
previously failed calls and can reach a slightly different end state than an
uninterrupted one.

**Not applicable** to work that never touches the cache — SQL over the store,
metrics computation, and the deterministic selection logic are all pure local
compute and parallelise or vectorise freely if they ever matter.

## Related

- `src/triage/evals/cache.py` — the constraint itself, at line 19 (the design
  note), line 139 (the connection), and lines 248-265 (the unguarded tally).
- [stratifier-failure-drops-repeat-the-selection-bias.md](stratifier-failure-drops-repeat-the-selection-bias.md)
  — the other half of the same episode: what the stratifier does when a call
  fails, as opposed to how fast the calls run.
- `docs/plans/2026-08-11-001-feat-feedback-triage-agent-plan.md` — U7 (line 379)
  describes the runner as having "bounded concurrency"; the shipped
  implementation is serial, and this document is the record of that divergence
  and why it stands.
