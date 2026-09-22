# Fine-tuning on Kaggle

A sub-0.5B model uses a fraction of a T4's 15 GB, and the whole sweep fits in
well under one week's 30-hour GPU quota. What shapes the workflow is that a
session has a time limit, there is no persistent machine, and only
`/kaggle/working` survives — and only when a run finishes normally.

## 1. Get the complete project onto Kaggle

**Use the zip, not individual files.** The first attempt failed with
`No rule to make target 'test'` and `can't open file test_guard.py` because the
repository on GitHub was assembled file by file and some files were missing or
outdated. Every notebook now runs `scripts/verify_repo.py` first and stops,
naming the problem files, if the copy is incomplete.

**Recommended — Kaggle Dataset (no git, no tokens):**

1. kaggle.com → *Datasets* → *New Dataset* → upload `offline-health-ai-assist.zip`.
   Kaggle unpacks it.
2. In each notebook: *Add Input* → your dataset. Nothing to edit — the notebook
   finds the project wherever it sits inside the dataset.
3. When the project changes: open the dataset → *New Version* → upload the new zip.

**Alternative — GitHub:** unzip, then from inside the folder that contains
`src/` and `Makefile`:

```bash
git init && git add -A && git commit -m "project"
git branch -M main
git remote add origin https://github.com/<you>/offline-health-ai-assist.git
git push -u --force origin main
```

The repository's front page should list `src/`, `data/`, `Makefile` directly.
Set `GITHUB_URL` in the notebook; for a private repository add a
`GITHUB_TOKEN` secret (read-only, this repository only). The notebook removes
the token from the clone afterwards, because `/kaggle/working` is saved with
the output.

## 2. Secrets and settings

| Setting | Why |
|---|---|
| Accelerator **GPU T4 x2** | Tensor cores make fp16 training much faster than on the P100 |
| Internet **On** | Needs a phone-verified account |
| Secret `HF_TOKEN` | Gemma is gated: create a token, then accept the Gemma licence on its model page **with the same account** |
| Secret `ANTHROPIC_API_KEY` | Notebook 02 only — grading calls the API and uses credits |

## 3. Notebook 01 — environment check (~25 GPU-minutes)

Runs, in order: completeness check, pinned installs, the CPU smoke test, GPU
selection, every training arm for ten steps with its peak memory, a check that
checkpoints are ordinary loadable models, and the evaluation path in deployed
form on the real GPU. Paste back the summary table and any FAIL lines.

**Do not add cells that load a model.** The first attempt's arms all failed
with "CUDA out of memory" within seconds — QLoRA could not get even 20 MB —
which means the GPU was already occupied before training started. A model
loaded in a notebook cell stays on the GPU until the kernel restarts. The
notebook now checks free memory on both T4s, trains on the emptier one, and
stops with an explanation if neither has 12 GB free. Fix: *Run → Restart &
clear cell outputs*, then run the cells in order.

## 4. Notebook 02 — the sweep, across sessions

1. *Save Version → Save & Run All (Commit)*. Runs in the background.
2. The sweep stops itself after `BUDGET_HOURS` (default 7), cleanly, so the
   session always ends normally and its outputs are saved.
3. Open the notebook, *Add Input → Notebook output*, pick the version that ran.
4. Commit again. Completed stages are restored and never re-run.

It refuses to start without `data/train.jsonl`, `val.jsonl` and `test.jsonl`,
and refuses if train and test leak. Build the dataset locally, then upload a
new version of the project.

## What the sweep measures

Every arm is scored **in the form the phone will run it**. Weights are rounded
with `src/quant_sim.py`, an exact reimplementation of llama.cpp's Q4_0 and
Q8_0 arithmetic — verified bit-identical against both the C reference and
llama.cpp's own gguf-py — before a single answer is generated:

| Scored as | What it is |
|---|---|
| `fp16@none` | Unquantized upper bound |
| `fp16@q4_0` | Post-training quantization baseline: the same model, plainly rounded |
| `lora@q4_0`, `qlora4@q4_0` | Adapter methods, merged, then rounded |
| `qat4@q4_0` | **Main experiment**: trained against the exact Q4_0 rounding |
| `qat8@q8_0`, `qat_mixed@mixed` | The 8-bit and mixed-precision variants |

The headline comparison is `qat4@q4_0` against `fp16@q4_0` and `lora@q4_0`:
same deployed format, with and without quantization-aware training.

## Budget

| Phase | Estimate on a T4 |
|---|---|
| Notebook 01 | ~0.4 h |
| 15 training runs | ~3–5 h |
| 16 scored configurations, batched | ~1–2 h |
| Grading | API, not GPU |

Estimates. Notebook 01's per-arm timings replace them with real numbers.

## Settings that changed for the T4

- **fp16, never bf16.** `torch.cuda.is_bf16_supported()` returns True on a T4
  because bf16 can be *emulated* — slowly. The code checks compute capability.
- **fp32 master weights** for full-parameter arms, fp16 arithmetic via autocast.
- **Batch 2 × accumulation 8** instead of 4 × 4: the same effective batch of
  16, half the peak memory. With a 152k-token vocabulary the logits of four
  sequences alone need several GB.
- **Gradient checkpointing** on every arm: same maths, less memory.
- **No per-epoch checkpoints.** Each carried ~6 GB of optimizer state; three
  would exceed Kaggle's 20 GB output limit on one arm. Final models are saved
  in fp16 (~1 GB for 0.5B).

## Exporting the phone models (after the sweep)

Only for the configurations you benchmark. Conversion is memory-bound, so use a
CPU session or your own machine:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --target llama-quantize
pip install gguf
python3 src/quantize.py --run runs/qwen05_qat4 --llama-cpp ~/llama.cpp
```

The exporter passes every llama-quantize choice explicitly — including
`--tensor-type` overrides for the mixed arm — then reads the file back and
checks every tensor's type against the spec.

## If something fails

Each training run writes a full log (`/kaggle/working/smoke_<arm>.log` in
notebook 01, `runs/logs/` in the sweep). Bring back the lines the notebook
prints under the failure — most problems are version or memory issues, and
quick to fix once the actual error is visible.
