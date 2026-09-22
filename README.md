# Offline Health AI-ASSIST

Comparative quantization-aware fine-tuning of small language models for an
offline Android medication-enquiry chatbot.

Built to the supervisor's brief; evaluation instrument and fixed training
configuration adopted from Elangovan et al. 2025 (*PLOS Digital Health*
4(9):e0000961, CC0).

---

## Start here

```bash
make setup          # installs dependencies, restores execute bits
make test           # full smoke test, ~20s, no GPU / API key / device needed
```

Execute permissions do not survive a zip or a plain file copy, so
`./run_smoke_test.sh` fails on a fresh copy with "Permission denied". Use
`make` and the problem disappears.

Individual checks, all runnable on a fresh clone:

```bash
python3 src/safety/test_guard.py                                   # no dependencies at all
python3 src/stats.py                                               # self-test on synthetic data
python3 data/validate_dataset.py data/seed/train_sample.jsonl --kind train
python3 src/train.py --model qwen05 --method lora --dry-run        # validates config, trains nothing

# Whole evaluation chain on mock data — no GPU, no API key
python3 src/generate.py --mock --arms fp16,lora,qat4 --eval data/seed/eval_quality_sample.jsonl
python3 src/score.py --mock-judge
python3 src/stats.py results/score_results.jsonl
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
data/check_leakage.py    Exact + near-duplicate overlap across splits.
data/validate_dataset.py Run before every training job.
src/train.py             ONE script, --method flag, six arms.
src/generate.py          Batch inference. --mock runs with no GPU.
src/quant_sim.py         Exact llama.cpp Q4_0/Q8_0 quantizer: QAT + deployed evaluation.
src/deploy_specs.py      The one definition of how each arm is quantized.
src/quantize.py          GGUF export from the same spec, verified tensor by tensor.
src/benchmark.py         Day 13 on-device measurement.
src/eval_guard.py        Block rate by adversarial type + false-positive rate.
src/run_pipeline.py      Runs the whole sweep unattended. Resumable.
src/figures.py           Day 14 figures and tables (PDF + LaTeX + markdown).
src/agreement.py         Judge-vs-human agreement. Required for defensibility.
src/test_stats.py        Validates Dunn's + kappas against reference libraries.
Makefile                 Entry point: make setup / test / paper / check-mock.
run_smoke_test.sh        End-to-end check. No GPU, no API key, no device.
src/score.py             SCORE judge harness + human agreement sampling.
src/stats.py             Kruskal-Wallis, Dunn's post hoc, Fleiss' kappa.
src/safety/guard.py      Deterministic on-device safety layer.
src/safety/rules.json    Rule file — loaded unchanged by the Dart port.
mobile/NOTES.md          Day 3 toolchain spike.
docs/DAY1_CHECKLIST.md   What to do today.
docs/PIPELINE.md         How the pieces connect + day mapping.
docs/KAGGLE.md           Fine-tuning on Kaggle: setup, sessions, budget.
kaggle/                  01 environment check, 02 resumable sweep.
docs/THESIS_OUTLINE.md   Chapter structure, word budgets, figure slots.
paper/                   Conference paper skeleton (compiles today).
```

## Three rules that keep the experiment valid

**1. Only `--method` varies.** Dataset, splits, seed, epochs, learning rate,
prompt template and generation settings stay fixed across every arm. That is
what makes it a controlled comparison rather than a collection of runs.

**2. Every arm is scored as the phone will run it.** Quality is measured on
the quantized weights, never on the fp16 checkpoint — the benefit of
quantization-aware training only exists after quantization. One quantizer,
`src/quant_sim.py`, reproduces llama.cpp's Q4_0/Q8_0 arithmetic exactly
(bit-identical against the C reference and against llama.cpp's own gguf-py),
and is used for training, for evaluation, and to specify the exported GGUF, so
all three see the same weights. The fp16 arm scored at Q4_0 is the
post-training quantization baseline.

**3. The statistics are validated, not assumed.** Dunn's test, Cohen's kappa
and Fleiss' kappa are implemented by hand here so they can be explained under
examination. `python3 src/test_stats.py` cross-checks them against
scikit-learn, scikit-posthocs and the canonical Fleiss worked example, and
probes the degenerate cases real data produces. A wrong implementation would
make every p-value in the results chapter wrong, and nothing downstream would
flag it.

**4. Grading never silently drops a row.** A judge call that fails after
retries is written with `status: "failed"`, never skipped. Silently dropping
failures would give arms unequal n through non-random missingness — the
comparison would be biased and nothing downstream would show it. Rerunning
`src/score.py` resumes and retries only what is missing; `--verify` reports
completeness per arm.

**5. The guard never calls the model.** A sub-billion-parameter model cannot
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
