# Pipeline

```
raw source ──► data/build_dataset.py ──► data/{train,val,test}.jsonl
                  audit | refusals | safety | build
                            │
                            ▼
data/train.jsonl ──► src/train.py ──► runs/{model}_{method}/final
                        │ --method fp16|lora|qlora4|qat8|qat4|qat_mixed
                        ▼
                    src/quantize.py ──► runs/.../gguf/model-q4_0.gguf
                        │ SCHEME MATCH CHECK              │
                        ▼                                 ▼
data/eval_*.jsonl ─► src/generate.py            src/benchmark.py (Day 13)
                        │ --guard both                    │
                        ▼                                 ▼
              results/generations.jsonl        results/device_benchmark.json
                        │
                        ▼
                    src/score.py  (SCORE rubric + reproducibility)
                        │
                        ▼
              results/score_results.jsonl
                        │
                        ▼
                    src/stats.py  (KW + Dunn's + paired Wilcoxon)
                        │
                        ├── src/eval_guard.py ──► results/guard_results.json
                        ▼
                    src/figures.py ──► results/figures/*.pdf, *.tex, *.md
```

## Smoke test

```bash
./run_smoke_test.sh
```

Eight stages: guard tests, generated-asset sync, parity corpus, dataset tooling,
guard evaluation, training config, the full mock evaluation chain, and the paper
build. Roughly twenty seconds, and it is what catches a change in one place
breaking another across eighty-odd files.

## Validate the whole chain today — no GPU, no API key

```bash
python src/generate.py --mock --arms fp16,lora,qat4,qlora4,ptq4
python src/score.py --mock-judge
python src/stats.py results/score_results.jsonl
```

This is the Day 6 gate. If the pipeline runs on mock data, Day 11 is execution
rather than debugging. **Never report mock numbers** — both scripts print a
warning for a reason.

## Two analysis decisions already made for you

**Guard on and guard off are analysed separately.** They are a paired factor on
the same items, not independent arms. Putting all twelve into one Kruskal-Wallis
produces 66 pairwise comparisons, and Bonferroni then makes almost nothing
significant. Compare quantization arms *within* a guard condition, then use the
paired Wilcoxon signed-rank test to isolate what the guard contributes. That
delta answers "which component is doing the work" — a question you will be asked.

**Evaluation-set size drives everything.** At 5 items nothing reached
significance; at 130 the same arms separated cleanly. Med-Pal used 231
validation questions. Aim for at least 120 or the comparison cannot support a
conclusion. `stats.py` prints a power warning when the numbers get thin.

## Run the whole sweep unattended

```bash
make plan        # what will run
make rehearse    # full dry run on mock data, ~10s
make sweep       # the real thing — resumable
make status      # progress
```

Every stage writes a marker on success, so a crash costs that stage rather than
the night. `--force <key>` redoes one stage; `--only train` runs one phase.
Days 7 to 11 become one command plus monitoring.

Two things the orchestrator handles that hand-driving gets wrong: every arm
shares the same fixed configuration, which is what makes the comparison
controlled; and each QAT arm is exported to the scheme it actually simulated,
rather than to a single global target that can only be right for one of them.

## Day mapping

| Brief day | Command |
|---|---|
| 4 | `python data/build_dataset.py audit --source raw/afrimedqa.jsonl` |
| 5 | `python data/build_dataset.py refusals --n 300 --out data/refusals.jsonl` |
| 5 | `python data/build_dataset.py safety --out data/eval_safety.jsonl` |
| 5 | `python data/build_dataset.py build --source raw/... data/refusals.jsonl --out data/` |
| 5 | `python data/check_leakage.py data/train.jsonl data/val.jsonl data/test.jsonl data/eval_safety.jsonl` |
| 5 | `python data/validate_dataset.py data/train.jsonl --kind train` |
| 6 | mock chain above + `python src/train.py --model qwen05 --method lora --dry-run` |
| 6 | `python src/quantize.py --run runs/qwen05_qat4 --verify-only` |
| 7-11 | `make sweep` runs everything below unattended; the per-stage commands are kept for reference |
| 7 | `--method fp16`, `--method lora` |
| 8 | `--method qlora4` |
| 9 | `--method qat8`, `--method qat4` (exact llama.cpp rounding in the forward pass) |
| 10 | `--method qat_mixed`; the PTQ baseline is `fp16` scored at Q4_0 — no extra run |
| 11 | `src/generate.py` (real) → `src/score.py --workers 4` → `src/score.py --verify` |
| 11 | `src/score.py --sample-human`, grade by hand, then `src/agreement.py` |
| 11 | `src/stats.py results/score_results.jsonl` |
| 12 | `./mobile/sync_rules.sh`, then swap `MockLlmService` for `LlamaCppService` in `mobile/lib/main.dart` |
| 13 | `cd mobile && flutter run --profile`, run the Benchmark screen |
| 13 | `python src/benchmark.py --gguf runs/<run>/gguf/model-q4_0.gguf --collect` |
| 14 | `python src/eval_guard.py --safety data/eval_safety.jsonl --legit data/test.jsonl` |
| 14 | `python src/figures.py --scores results/score_results.jsonl --guard results/guard_results.json --bench results/device_benchmark.json --sizes '{"qwen05-qat4@q4_0":400,...}'` |
| 14 | `cd paper && make` (pulls the regenerated tables and figures automatically) |
| 14 | `cd paper && make check` — must pass before submission |
