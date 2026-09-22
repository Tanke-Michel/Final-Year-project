# Thesis Outline

Chapter structure for the final report, with word budgets, figure slots and the
things each section must contain. Word counts assume roughly 12,000–15,000
words total; scale proportionally once your supervisor confirms the expected
length (decision Q4 in the interim report).

Figures referenced here are the ones `src/figures.py` emits into
`results/figures/`. Regenerate rather than redraw.

---

## Chapter 1 — Introduction (~1,200 words)

| Section | Content | Words |
|---|---|---|
| 1.1 Background | Medication information needs; connectivity in sub-Saharan Africa; why a server-dependent assistant fails at the moment of need | 400 |
| 1.2 Problem statement | Compression is unavoidable on a low-end handset. Its effect on medication-safety behaviour is unmeasured below 0.5B | 300 |
| 1.3 Aim and objectives | General objective, then the six specific objectives from the brief | 200 |
| 1.4 Scope and exclusions | Medication information only. Dosing, diagnosis and prescribing explicitly out of scope | 150 |
| 1.5 Contributions | Numbered, one line each, each verifiable from Chapter 5 | 150 |

**Must appear:** the sub-0.5B constraint and which parameter convention you count.

---

## Chapter 2 — Literature Review (~2,500 words)

| Section | Content | Words |
|---|---|---|
| 2.1 LLMs in medicine | General landscape; where consensus sits on patient-facing use | 500 |
| 2.2 Lightweight medical models | Med-Pal in detail; BioMistral; Meerkat. State plainly that none exists below 0.5B | 700 |
| 2.3 Quantization | Post-training methods against quantization-aware training. Make the QLoRA distinction explicitly | 700 |
| 2.4 On-device inference | Engines, formats, the constraints a 3 GB device imposes | 400 |
| 2.5 The gap | Two sentences. Everything above exists to justify these | 200 |

**Must appear:** Med-Pal's own future-work statement calling for evaluation of
quantization technique. That sentence justifies the entire study, so quote its
substance and cite it precisely.

---

## Chapter 3 — Methodology (~2,800 words)

| Section | Content | Words |
|---|---|---|
| 3.1 Research design | Controlled comparison; what is held fixed and what varies | 300 |
| 3.2 Data | Sources, licences, filtering, domain and difficulty classification, stratified splits | 700 |
| 3.3 Model selection | Candidates with total and non-embedding counts; selection criteria | 400 |
| 3.4 Compression strategies | One paragraph per arm | 600 |
| 3.5 Scheme matching | Why the simulated scheme must match the deployed one | 300 |
| 3.6 Safety layer | Rules, priority, conjunctive matching, fail-closed, parity mechanism | 300 |
| 3.7 Evaluation | SCORE, judging protocol, human subsample, statistical plan | 200 |

**Must appear:**
- That every configuration is scored in its deployed, quantized form, using
  one quantizer shared by training, evaluation and export — and how it was
  verified against llama.cpp. This is what makes the quality numbers mean
  "quality on the phone".
- The divergence from Med-Pal on `dosage_regimen`, framed as a considered
  decision for a system with no clinician in the loop.
- Why refusals are in the training set, not only in evaluation.
- The prohibition on doses in reference answers and how it is enforced.
- The minimum test-set size and why.

---

## Chapter 4 — Implementation (~2,000 words)

| Section | Content | Words | Figure |
|---|---|---|---|
| 4.1 System architecture | Pipeline diagram; the three layers | 400 | architecture diagram (draw this) |
| 4.2 Training pipeline | One parameterised script; reproducibility measures | 400 | — |
| 4.3 Safety layer | Rule structure; worked example of an intervention | 400 | — |
| 4.4 Quantization and export | Format, group size, the scheme check | 300 | — |
| 4.5 Mobile application | Architecture, guarded turn, generated assets, privacy of the intervention log | 500 | app screenshots |

**Must appear:** that the model is unreachable except through the guard, and
that this is structural rather than procedural. Include a screenshot of an
intercepted response showing the rule identifier.

---

## Chapter 5 — Results (~2,500 words)

| Section | Content | Figure / Table |
|---|---|---|
| 5.1 Answer quality by arm | Median, IQR, combined safety+accuracy | `table1_scores.tex`, `fig1_score_box` |
| 5.2 Domain breakdown | Where each arm is strong and weak | `fig2_score_domains` |
| 5.3 Statistical comparison | Omnibus test, then surviving pairwise comparisons only | — |
| 5.4 Compression floor | The central result | `fig3_compression_floor` |
| 5.5 Reproducibility | The novel axis | `fig5_reproducibility` |
| 5.6 Safety layer | Block rate with false-positive rate; unwrapped probes; both defects found | `figA_guard_rates`, `figB_rule_fires` |
| 5.7 Guard contribution | Paired test | `fig4_guard_effect` |
| 5.8 On-device benchmark | Size, memory, latency, throughput; medians with ranges | `table3_benchmark.md` |

**Rules for this chapter:**
- Lead each subsection with the finding, then the procedure.
- Never report a pairwise comparison without the omnibus test that licenses it.
- Never report a block rate without the false-positive rate beside it.
- Report the agreement between automated and manual grading. Without it, the
  automated evaluation is not defensible.
- Report the pre-registered threshold and confirm it was fixed before results
  were seen.

---

## Chapter 6 — Discussion (~2,000 words)

| Section | Content | Words |
|---|---|---|
| 6.1 Interpretation | What the comparison shows; be specific about what did and did not survive correction | 500 |
| 6.2 The reframing | Memory is not binding below 0.5B; quality and safety are | 400 |
| 6.3 Role of the safety layer | If fine-tuning cannot produce reliable refusal at this scale, the deterministic layer is the mechanism of safety, not an accessory | 400 |
| 6.4 Comparison with Med-Pal | Methodological comparison only. Their best model is far larger and ran on a workstation GPU | 300 |
| 6.5 Limitations | Written properly, not defensively | 400 |

**Limitations that must appear:**
- Single evaluation language.
- Automated grading with a limited human subsample.
- Templated adversarial probes are not a red team.
- A single target device.
- No prospective clinical evaluation.
- Training corpora under-represent the population the system is argued to
  serve. Given the stated motivation, this matters more than usual and should
  be stated plainly rather than softened.

---

## Chapter 7 — Conclusion and Future Work (~800 words)

What was compared, what was found, what it implies for deployment, what should
follow. Claim nothing the results do not support. Do not describe the system as
clinically usable.

---

## Front and back matter

- Abstract (250 words, written last)
- Acknowledgements
- List of figures and tables
- Abbreviations
- **Ethics statement** — public non-identifiable data; not a medical device;
  human referral is architectural; the demographic limitation above
- **Reproducibility statement** — repository, fixed seed, exact model
  revisions, judge model and version
- References
- Appendices: rule base with cited guideline sources; safety evaluation set;
  hyperparameters; full results tables

---

## Writing order

Do not write Chapter 1 first. Write in this order:

1. **Chapter 3** (Methodology) — you can write it today; nothing is pending
2. **Chapter 4** (Implementation) — also complete
3. **Chapter 5** (Results) — the day the analysis finishes, while it is fresh
4. **Chapter 2** (Literature) — after Chapter 5, so the gap is framed by what
   you actually found
5. **Chapter 6** (Discussion)
6. **Chapter 1** (Introduction) — the contributions list must match Chapter 5
7. **Chapter 7**, then the abstract

Chapters 3 and 4 are roughly 4,800 words you can write before any model
finishes training. That is a third of the thesis, available now.
