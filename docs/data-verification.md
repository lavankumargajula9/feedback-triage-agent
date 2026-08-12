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

**Data-level confirmation — CLOSED 2026-08-12.** The CSV was downloaded
(169 MB) and ingested: **2,811,774 tweets into 901,648 threads**, across 108
distinct brand accounts. A random sample of multi-tweet threads carries the
texture generated data does not have:

- **Agent sign-off initials** on brand replies (`^PS`, `^mm`, `^FR`, `^Monica`) —
  a real support-desk convention for attributing a shared account to a person.
- **Character-limit artifacts**: replies manually split as `1/2`, `2/2`.
- **Unmoderated customer affect**: all-caps escalation ("I HAVE BEEN REPLYING
  SINCE TWO DAYS"), profanity, emoji, hashtags.
- **Third-party intrusions**: uninvolved customers replying into another
  customer's thread — exactly the sibling replies R31 excludes.
- **Specific, non-templated grievances**: a bag lost JFK→Heathrow→Bucharest, a
  £5 credit offered after a failed grocery delivery before a dinner party.
- **Partial handle masking**: brands appear both as names (`@AmazonHelp`) and as
  numeric ids (`@115830`, `@116035`) where the mention was not linked, alongside
  the numeric customer ids — the signature of an anonymization pass over real
  scraped text.

Two reconstruction statistics worth carrying into the README's limitations:

- **Truncated threads: 4,387 of 901,648 (0.49%)** — the dataset is a slice, so a
  small fraction reference parents outside it.
- **Cycle-flagged threads: 0.** Reply cycles do not occur in this data; the
  cycle guard in R31 is defensive only, exercised by a synthetic fixture rather
  than by real traffic. Do not claim it as a handled real-world case.

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
