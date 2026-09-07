# Pipeline

```
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
```

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

## Day mapping

| Brief day | Command |
|---|---|
| 5 | `python data/validate_dataset.py data/train.jsonl --kind train` |
| 6 | mock chain above + `python src/train.py --model qwen05 --method lora --dry-run` |
| 6 | `python src/quantize.py --run runs/qwen05_qat4 --verify-only` |
| 7 | `--method fp16`, `--method lora` |
| 8 | `--method qlora4` |
| 9 | `--method qat8`, `--method qat4` |
| 10 | `--method qat_mixed`, then `src/quantize.py` for PTQ baselines |
| 11 | `src/generate.py` (real) → `src/score.py` → `src/score.py --sample-human` → `src/stats.py` |
| 12 | Flutter integration; port `src/safety/rules.json` unchanged |
| 13 | `python src/benchmark.py --gguf ... --repeats 5` |
