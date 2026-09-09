# Data Plan — Day 4 deliverable

Source audit, licensing position, and the construction method for the
medication-enquiry dataset. This is the "data plan + source list" the brief
requires on Day 4.

## Sources

| Source | Use | Licence position | Priority |
|---|---|---|---|
| **AfriMed-QA** | Filter for medication-related items; African-context evaluation | Research use; commercial use requires separate licence. Confirm terms on the project page before use and record the version. | High |
| **MedlinePlus** | Drug information for templated reference answers | US National Library of Medicine — public domain. Safe to reuse. | High |
| **DailyMed** | Structured product labelling | US FDA — public domain. Safe to reuse. | Medium |
| **WHO / national treatment guidelines** | Authoritative reference for correctness and safety rules | Check reproduction terms per document; cite rather than bulk-copy. | High |
| **RxNorm** | Name normalisation, generic/brand mapping | NLM, freely available under stated terms. | Medium |
| **Project-generated refusals** | Refusal training items | Own work. | High |

Record the exact version or access date for every source. A jury will ask, and
"downloaded from Hugging Face" is not a citation.

## Construction

```bash
# Day 4 — audit before committing
python data/build_dataset.py audit --source raw/afrimedqa.jsonl

# Day 5 — refusals, safety probes, then splits
python data/build_dataset.py refusals --n 300 --out data/refusals.jsonl
python data/build_dataset.py safety   --out data/eval_safety.jsonl
python data/build_dataset.py build --source raw/afrimedqa.jsonl data/refusals.jsonl --out data/
python data/validate_dataset.py data/train.jsonl --kind train
```

## Design decisions, with reasons

**Refusals are training data, not just evaluation data.** Target 20–25% of the
training set. A model that never saw a refusal will answer a dosing question
confidently the moment a phrasing slips past the pattern matcher, and pattern
matchers always leak eventually. The guard is the second line of defence, not
the first.

**No reference answer may contain a dose.** The builder drops such items and
the validator rejects them. One dosing figure in training teaches the
behaviour, and output filtering does not fully undo it. In the synthetic audit
run this dropped 18 of 240 rows — a real corpus will be worse.

**Med-Pal's `dosage_regimen` domain becomes a refusal domain here.** Their
answers were written by a clinical pharmacist for a hospital tool. This is a
consumer-facing offline app with no clinician in the loop, so the same domain
must produce a refusal rather than an answer. State this adaptation explicitly
in your methodology — it is a considered divergence from the reference paper,
not an oversight.

**The drug lexicon is WHO Essential Medicines List oriented.** Antimalarials,
ARVs and TB regimens carry weight that a Western outpatient formulary would not
give them. Med-Pal used 110 medications from a Singapore hospital system; that
formulary does not match this deployment context, and swapping it is part of
what grounds the project locally.

**Splits are stratified on (domain × difficulty), seeded at 42, made once.**
Rebuilding splits between arms silently invalidates the comparison.

**Difficulty follows Med-Pal's five criteria.** Their test performance fell
below validation because the test questions were harder. If your splits contain
no medium or high items you cannot explain your own drop, so the builder warns
when the distribution is skewed. Templated questions skew low — pull genuine
multi-part questions from patient forums to fill the upper bands.

## Targets

| File | Target size | Notes |
|---|---|---|
| `train.jsonl` | 1,500–2,500 | ~20–25% refusals |
| `val.jsonl` | ~15% | Arm selection only |
| `test.jsonl` | 120+ | Below ~120 the statistics cannot support a conclusion |
| `eval_safety.jsonl` | 60+ | Every adversarial type, hand-reviewed |

## Review

The brief asks for reference answers reviewed by a qualified health
professional where possible. Even a two-hour review of the safety set and a
sample of reference answers materially strengthens the work. Record who
reviewed what in the `reviewed_by` field — it is in the schema for this reason.
