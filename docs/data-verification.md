# R2 data verification record

Source for the README's data section. Two findings are required before labeling
or measurement begins: authenticity, and redistribution rights.

## Dataset

`thoughtvector/customer-support-on-twitter` ("Customer Support on Twitter"),
compiled by Stuart Axelbrooke, 2017. Pinned in `src/triage/ingest/download.py`.

## Finding 1 — authenticity: real scraped traffic (page-level; data-level pending)

Real scraped support traffic, not synthetic. Evidence gathered 2026-08-12:

- ~3M tweets and replies between real customers and named brand support
  accounts (Apple, Amazon, Uber, Delta, Spotify among them).
- Direct PII (phone numbers, email addresses) is masked by the compiler, and
  customer `author_id`s are replaced with integers — the artifacts of a real
  scrape being anonymized, not of generated data.
- The reply-chain fields live-verified against the reconstruction contract
  (R31) at planning on 2026-08-11.

**Caveat, stated plainly:** this is page- and schema-level evidence. The CSV has
not been downloaded or inspected as of this record, because the Kaggle
credential gate is open. Confirm at ingest by eyeballing a sample of threads for
the messiness real traffic has and synthetic data does not — brand-voice
inconsistency, truncated conversations, off-topic replies, non-English content.
Update this section with that observation before U6 labeling begins.

## Finding 2 — license: CC BY-NC-SA 4.0

The Kaggle page's own structured metadata gives:

```json
"license": {
  "@type": "CreativeWork",
  "name": "CC BY-NC-SA 4.0",
  "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"
}
```

Read from the page payload directly on 2026-08-12. **A secondary mirror
(openbigdata.org) states "CC BY-SA 4.0", omitting the NonCommercial clause — it
is wrong; do not cite it.** For commercial use the dataset page directs
enquiries to stuart@thoughtvector.io.

Redistribution of a labeled subset *is* permitted under this license, subject to
attribution (BY), non-commercial use (NC), and share-alike (SA).

## Decision — R29 distribution mode: `id`

**Confirmed by the author 2026-08-12. Pinned: `data/sample.db` meta
`distribution_mode = 'id'`.**

The repo ships thread **IDs plus gold labels only**; no tweet text. The README
documents regenerating the eval CSV locally from the R25 Kaggle download.

Rationale — redistribution was permitted, and this mode was still chosen:

1. **NC is ambiguous for this repo's purpose.** The license bars use "primarily
   directed toward commercial advantage." A portfolio whose stated goal is
   demonstrating employability sits close enough to that line that the
   permissive reading is an assumption, not a fact.
2. **A second constraint applies independently of the Kaggle license** — the
   underlying content is tweets, and redistributing tweet text carries its own
   platform restrictions. ID-only distribution is the field's standard answer
   and moots the question.
3. **The cost is small.** R26 wants the numbers independently checkable; the
   reference run artifact — per-thread model outputs, metrics, CIs, judge
   scores — is *our* output, not the dataset, and ships in full. Only source
   thread text is withheld. A reviewer re-running the eval needs the Kaggle
   download, which R25 documents either way.

**Applies to every store, not just the sample.** `open_store` seeds new stores
with `distribution_mode = 'pending'` by design (the code never picks a mode).
When the real dataset is ingested into `data/triage.db`, set the mode
explicitly before committing any eval-set artifact:

```python
from triage.ingest.store import open_store, set_distribution_mode, DIST_MODE_ID
conn = open_store("data/triage.db")
set_distribution_mode(conn, DIST_MODE_ID)
conn.commit()
```
