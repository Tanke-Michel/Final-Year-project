# Offline Health AI-ASSIST

Comparative quantization-aware fine-tuning of small language models for an
offline Android medication-enquiry chatbot.

Built to the supervisor's brief; evaluation instrument and fixed training
configuration adopted from Elangovan et al. 2025 (*PLOS Digital Health*
4(9):e0000961, CC0).

---

## Start here

```bash
pip install -r requirements.txt

# 1. Prove the guard works (no ML dependencies needed — run this first)
python src/safety/test_guard.py

# 2. Prove the stats pipeline works on synthetic data
python src/stats.py

# 3. Validate a dataset before any training job
python data/validate_dataset.py data/train.jsonl --kind train

# 4. Check a training config without training
python src/train.py --model qwen05 --method lora --dry-run

# 5. Smoke-test the ENTIRE evaluation chain — no GPU, no API key
python src/generate.py --mock --arms fp16,lora,qat4,qlora4,ptq4
python src/score.py --mock-judge
python src/stats.py results/score_results.jsonl
```

## Layout

```
configs/base.yaml        Fixed hyperparameters. DO NOT vary between arms.
configs/models.yaml      Candidate shortlist + 3 GB RAM budget.
data/seed/               Schema examples for all three dataset files.
data/build_dataset.py    Days 4-5: audit, refusals, safety probes, splits.
data/drug_lexicon.json   WHO EML-oriented, sub-Saharan weighted.
data/domains.json        Question domains, templates, difficulty rules.
data/DATA_PLAN.md        Day 4 deliverable: sources + licensing.
data/validate_dataset.py Run before every training job.
src/train.py             ONE script, --method flag, six arms.
src/generate.py          Batch inference. --mock runs with no GPU.
src/quantize.py          GGUF export + the scheme-match check.
src/benchmark.py         Day 13 on-device measurement.
src/eval_guard.py        Block rate by adversarial type + false-positive rate.
src/figures.py           Day 14 figures and tables (PDF + LaTeX + markdown).
src/score.py             SCORE judge harness + human agreement sampling.
src/stats.py             Kruskal-Wallis, Dunn's post hoc, Fleiss' kappa.
src/safety/guard.py      Deterministic on-device safety layer.
src/safety/rules.json    Rule file — loaded unchanged by the Dart port.
mobile/NOTES.md          Day 3 toolchain spike.
docs/DAY1_CHECKLIST.md   What to do today.
docs/PIPELINE.md         How the pieces connect + day mapping.
docs/THESIS_OUTLINE.md   Chapter structure, word budgets, figure slots.
paper/                   Conference paper skeleton (compiles today).
```

## Three rules that keep the experiment valid

**1. Only `--method` varies.** Dataset, splits, seed, epochs, learning rate,
prompt template and generation settings stay fixed across every arm. That is
what makes it a controlled comparison rather than a collection of runs.

**2. The QAT simulation must match the deployed format.** `configs/base.yaml`
sets `gguf_q4_0`, group size 32, symmetric. If you simulate one scheme and
ship another, you have not measured quantization-aware training — you have
measured QAT-for-a-different-target followed by an unrelated post-training
quantization. Verify this end-to-end on Day 6, on one checkpoint, before the
real runs.

**3. The guard never calls the model.** A sub-billion-parameter model cannot
police itself. Everything in `src/safety/` is string and regex matching, fails
closed, and returns a rule id for every intervention so the results chapter can
report which rules fired and how often.

## Hard constraint: under 0.5B parameters

Every candidate must be below 0.5 billion total parameters. SmolLM2-1.7B was
removed for breaching this by 3.4x. State in the thesis whether you count total
or non-embedding parameters — they diverge sharply at this scale.

## Expected results

Med-Pal's combined safety-and-accuracy figures: Mistral-7B 71.9%,
TinyLlama-1.1B 39.4%, Danube-1.8B 18.2%. Every model here is a QUARTER the
size of Danube-1.8B or smaller.

At this scale memory stops being the binding constraint — all candidates are
110-400 MB at 4-bit, comfortably inside the ~1.4 GB budget. Quality and safety
become the constraint instead, which reframes the contribution from "can it
fit on the phone" to "is it good enough to be safe on the phone".

Plan for low absolute quality. The project is not "we built a good chatbot" —
it is an investigation of how far a medication model can be compressed before
quality and safety become unacceptable, and what must be handled
deterministically once the model can no longer be trusted. That framing
survives a poor result. The other one does not.

## Not a medical device

Research prototype. Public, non-identifiable data only. No dosing advice, no
diagnosis, human referral by design.
