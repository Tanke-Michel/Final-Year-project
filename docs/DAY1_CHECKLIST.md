# Day 1 — today

Brief says: read 5-10 papers, produce literature notes and an initial model
list. Do that. These run alongside it and de-risk the rest of the fortnight.

## Before anything else (30 min)

- [ ] `python src/safety/test_guard.py` — should report 18/18 blocked, 0 false positives
- [ ] `python src/stats.py` — should print a full Kruskal-Wallis + Dunn's table
- [ ] `git init && git add -A && git commit -m "scaffold"` — commit history is evidence of your own work

## Downloads to start now, not later

- [ ] **`MedPal_FineTuning_Data_Sample.xlsx`** from
      `github.com/MedPal-Project/MedPal---Lightweight-Medical-Enquiry-Chatbot-`
      This is your dataset template. Knowing its shape changes how you spend Days 4-5.
- [ ] arXiv:2407.12822 (the Med-Pal preprint — sometimes carries detail trimmed in publication)
- [ ] AfriMed-QA from Hugging Face — check the licence terms while it downloads
- [ ] Kaggle or Colab account confirmed working with GPU enabled

## Three decisions to raise with your supervisor today

These cascade. Late answers cost days.

1. **Inference engine** (llama.cpp/GGUF recommended). Determines the deployable
   quantization format, which determines what QAT must simulate on Day 9.
2. **Target device** — exact phone model, Android version, chipset. An emulator
   gives meaningless latency numbers, and Day 13 needs real ones.
3. **Judge access** — an API key for the SCORE harness. Without it Day 11 is
   not survivable. The phone never calls it, so the offline claim is unaffected.

## Set expectations in writing

Send your supervisor one short note today saying you expect low absolute
quality scores given the parameter sizes, citing Med-Pal's TinyLlama-1.1B
(39.4%) and Danube-1.8B (18.2%) figures, and that the study is framed as
locating the compression floor. Getting this agreed on Day 1 rather than
defended on Day 14 is the difference between a finding and a disappointment.

## Do NOT do today

- Do not open Flutter. Brief §11 is explicit and correct: model first.
- Do not start collecting the dataset before you have looked at the Med-Pal
  schema. You would only rework it.
