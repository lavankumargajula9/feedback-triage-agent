# Customer Feedback Triage Agent

A four-step LangGraph triage pipeline over real customer-support threads, exposed
through an MCP server and measured against a hand-labeled eval set with a
single-prompt baseline.

## Why this repo exists

My production agentic work is LangGraph orchestration with real operational
metrics, and none of it is inspectable — it is client and employer code. A
resume claim nobody can check reads the same as one that is untrue. This project
is the inspectable counterpart: the same orchestration claim, on public data,
plus the two things that production work does not demonstrate at all —
protocol-level tool exposure (MCP) and formal quantified evaluation of an LLM
system.

The triage domain is a real problem in its own right. Raw multi-brand support
traffic is messy: fragmented threads, vague complaints, mixed intents. Deciding
*what is this about*, *who should own it*, and *does a human need to see it now*
is exactly the work a single prompt handles poorly and a measured multi-step
system can be shown to handle better — if you actually measure it.

## Status

**In progress. No measured comparison exists yet, and this README will not carry
one until it does.**

| Component | State |
|---|---|
| Ingestion + thread reconstruction | Working — 2,811,774 tweets → 901,648 threads, 108 brands |
| Tool layer (categorize / route / draft / escalate) | Working — schema-enforced, typed failures |
| LangGraph pipeline + CLI | Working — pilot-verified over real threads (15/15 clean) |
| MCP server | Working — structural parity tests against the same tool functions |
| Eval harness (cache, runner, metrics, judge, regression) | Built and tested; batch mode smoke-verified against the live Batch API |
| Candidate pool for the eval set | **Frozen** — 80 threads, every category/queue bucket ≥ 12, zero shortfalls ([selection log](data/eval/selection_log.md)) |
| Gold labels | In progress — hand-labeling underway, three isolated passes |
| Measured baseline-vs-pipeline results | **Not produced** — runs the moment labels freeze |
| Terminal recording (R19) | Cut, per the plan's agreed cut ladder (first item cut; the eval core is never cut) |

The eval harness is complete *before* any results exist, which is deliberate:
the agreement protocol, tolerances, and metric set are frozen in code so they
cannot be tuned after seeing a number.

## Architecture

```
Kaggle twcs CSV
      |
      v
  ingest  ->  SQLite store (threads, reconstruction flags)
                  |
                  +--> triage run ----> LangGraph pipeline ----> JSON result
                  |                     categorize -> route
                  |                     -> draft -> escalate
                  |
                  +--> triage-mcp ----> same tool functions over MCP
                  |
                  +--> triage pool --> frozen candidate pool
                            |               |
                            |         triage label (3 isolated passes)
                            |               |
                            |         gold_labels.csv
                            |               |
                            +--------> triage eval
                                      pipeline arm vs single-prompt baseline
                                      at equal information
```

The pipeline and the baseline assemble their prompts from the same fragments
module, so "4-step pipeline vs single prompt" is a comparison at equal
information by construction rather than by assertion.

The eval-set construction path is deliberately **separate** from the runtime
pipeline and runs on a different model profile. Conflating them is the specific
error that would invalidate the measurement.

## Quickstart

Requires Python 3.12+ and an Anthropic API key.

```bash
python -m venv .venv
.venv/Scripts/activate          # POSIX: source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-...  # Windows: set ANTHROPIC_API_KEY=sk-...

pytest                           # full suite, no network — every LLM call is mocked
```

Get the data (see *Data and licensing* below for why it is not committed):

```bash
triage ingest                    # downloads via kagglehub, reconstructs threads
triage ingest --sample           # or: build a fixture store, no Kaggle account needed
```

Run one thread end to end:

```bash
triage run <tweet_id> --profile dev
```

Any tweet id belonging to a thread resolves to the whole reconstructed thread.
`--profile dev` uses the cheap development model; measurement models are reserved
for recorded runs only.

### Commands

| Command | Purpose |
|---|---|
| `triage ingest` | Kaggle CSV → reconstructed threads in SQLite |
| `triage run` | One thread or a batch → structured JSON result |
| `triage pool` | Build and freeze the stratified candidate pool |
| `triage label` | Hand-label the eval set, one field per pass |
| `triage eval` | Run both arms over the gold set; cache-first, resumable |
| `triage-mcp` | Serve the tool layer over MCP |

`triage eval --dry-run` prints the planned call count and cost estimate and
executes nothing. `triage pool --dry-run` does the same for pool construction.

`triage eval --batch` executes through the Messages Batch API at 50% of list
price: threads run against a collecting client that replays the shared call
cache and records misses, each wave submits the misses as one batch and
validates results back into the same cache, so dependent pipeline steps become
successive waves and an interrupted run resumes without re-paying — the
in-flight batch id is persisted and re-attached. Retry semantics and typed
failure kinds match the sync path exactly; per-call latency is the one metric
batch mode cannot record, so latency comparisons come from sync runs only.

## Data and licensing

Source: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(`twcs`), uploaded 2017. The raw CSV is hosted on Kaggle; no Twitter/X API is
involved.

The dataset is **CC BY-NC-SA 4.0**. Redistribution of a labeled subset would be
permitted, but this repo ships **IDs and labels only** — not thread text. Two
reasons, recorded in [`docs/data-verification.md`](docs/data-verification.md):
the non-commercial clause is ambiguous for a portfolio repo, and redistributing
tweet text carries its own constraint independent of the Kaggle license.

The practical consequence: reproducing the eval requires a Kaggle download and a
local rebuild. That path is documented rather than hidden.

No dataset text is committed. The test fixtures are hand-authored rows in the
dataset's schema, not scraped records.

## Methodology

The parts most worth reading, because they are where an eval usually goes wrong:

**Baseline at equal information.** Both arms build prompts from one shared
fragments module. The baseline is a genuine single-prompt attempt at the same
task with the same label definitions, not a strawman.

**Gold labels are collected in three isolated passes.** One annotator labels
category, then queue, then escalation — each a separate full sweep in its own
seeded order. Labeling all three fields per thread anchors each field on the
previous one, and that collapse would be invisible, because gold labels are what
everything else is measured against. The isolation is structural: a pass cannot
read another pass's file.

**Pool selection must not correlate with difficulty.** A keyword stratifier was
proposed and rejected twice: it admits only threads whose class is lexically
obvious, so both arms inflate and the measured lift compresses toward zero.
Disclosure was tried as a remedy and reverted. The reasoning is written up in
[`docs/solutions/tooling-decisions/keyword-stratification-biases-eval-pool.md`](docs/solutions/tooling-decisions/keyword-stratification-biases-eval-pool.md).

**Development and measurement models are decoupled.** All iteration runs on a
cheap dev profile or mocks; only recorded runs use the measurement pair. The
config asserts the judge model differs from the models it grades.

**Cleaning rules are stated, not silent.** Threads with no diagnosable content
are excluded from the eval set, and the excluded count is reported on every run.
The full record is in [`data/eval/selection_log.md`](data/eval/selection_log.md).

Project vocabulary is defined in [`CONCEPTS.md`](CONCEPTS.md).

## Results

Not yet measured. This section will carry the baseline-vs-pipeline comparison —
per-class precision/recall, confusion matrix, paired bootstrap confidence
intervals, draft-quality scores with judge-human agreement, and per-arm cost —
once the reference run is recorded, with every number traceable to the
committed run artifact.

Everything upstream of the numbers is already frozen: the 80-thread pool, its
[selection log](data/eval/selection_log.md) with achieved stratification
counts, the [labeling protocol](data/eval/labeling_cheatsheet.md), and the
measurement harness (smoke-verified end to end against the live Batch API).
The remaining input is the hand-labeling itself, which is deliberately slow:
the labels are the answer key, and this repo would rather show an empty
results table than an unreliable one.

## Limitations

Stated now rather than after the numbers land:

- **Escalation labels have no ground-truth proxy.** The dataset carries no
  escalation signal, so every escalation label is a human judgment call.
- **A single annotator produced all gold labels** and the judge-validation
  subsample. There is no inter-annotator agreement figure, and there cannot be.
- **The judge shares a provider with the models it grades.** A cross-provider
  judge would force a second API key on anyone reproducing this.
- **Rare-class strata are shaped by an imperfect classifier.** Candidates for a
  rare class come from threads a cheap model already believed were that class, so
  those strata skew toward cases that model finds legible. Bounded, and disclosed
  rather than assumed away.
- **Non-English threads** are present in the corpus and are not handled
  specially.

## Reproduction

1. Clone, create the venv, `pip install -e ".[dev]"`.
2. Set `ANTHROPIC_API_KEY`.
3. `triage ingest` (needs a Kaggle token at `~/.kaggle/kaggle.json`).
4. Rebuild the eval set from the shipped IDs and labels.
5. `triage eval --dry-run` to see planned calls and cost, then run it.

`pytest` requires none of the above — the suite is fully mocked and makes no
network calls.

## License

Code in this repo is MIT licensed — see [`LICENSE`](LICENSE).

The dataset is CC BY-NC-SA 4.0 and is **not** redistributed here. The MIT grant
covers this repo's code only; it does not extend to the `twcs` data, which you
obtain from Kaggle under its own terms.
