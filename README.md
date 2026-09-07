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
```

## Layout

```
configs/base.yaml        Fixed hyperparameters. DO NOT vary between arms.
configs/models.yaml      Candidate shortlist + 3 GB RAM budget.
data/seed/               Schema examples for all three dataset files.
data/validate_dataset.py Run before every training job.
src/train.py             ONE script, --method flag, six arms.
src/score.py             SCORE judge harness + human agreement sampling.
src/stats.py             Kruskal-Wallis, Dunn's post hoc, Fleiss' kappa.
src/safety/guard.py      Deterministic on-device safety layer.
src/safety/rules.json    Rule file — loaded unchanged by the Dart port.
mobile/NOTES.md          Day 3 toolchain spike.
docs/DAY1_CHECKLIST.md   What to do today.
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

## Expected results

Med-Pal's combined safety-and-accuracy figures: Mistral-7B 71.9%,
TinyLlama-1.1B 39.4%, Danube-1.8B 18.2%. Every model in this project is
smaller than TinyLlama-1.1B or close to it.

Plan for low absolute quality. The project is not "we built a good chatbot" —
it is an investigation of how far a medication model can be compressed before
quality and safety become unacceptable, and what must be handled
deterministically once the model can no longer be trusted. That framing
survives a poor result. The other one does not.

## Not a medical device

Research prototype. Public, non-identifiable data only. No dosing advice, no
diagnosis, human referral by design.
