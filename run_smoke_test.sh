#!/usr/bin/env bash
# End-to-end smoke test. No GPU, no API key, no device.
#
# Run this after ANY change. With 80-odd files and several generated artefacts
# that must stay in sync, a change in one place can silently break another —
# this is what catches it.
#
#     ./run_smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")"
FAIL=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
ok()   { printf '   \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '   \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=1; }

step "0. Project files complete"
if python3 scripts/verify_repo.py >/tmp/st_repo.log 2>&1; then
  ok "$(head -1 /tmp/st_repo.log | sed 's/^ *//')"
  grep -q "MODIFIED" /tmp/st_repo.log && sed -n '/MODIFIED/,$p' /tmp/st_repo.log | head -12
else
  bad "project is incomplete — later results are meaningless until this passes"
  sed -n '/MISSING/,$p' /tmp/st_repo.log | head -25
fi

step "1. Safety guard"
python3 src/safety/test_guard.py >/tmp/st_guard.log 2>&1 \
  && ok "guard tests + rule hygiene" || { bad "guard tests"; tail -12 /tmp/st_guard.log; }

step "2. Generated assets in sync"
python3 - <<'PY' && ok "rules.json == mobile/assets/rules.json" || bad "rules drifted — run ./mobile/sync_rules.sh"
import json,sys
a=json.load(open("src/safety/rules.json")); b=json.load(open("mobile/assets/rules.json"))
sys.exit(0 if a==b else 1)
PY
python3 - <<'PY' && ok "model_config.json matches base.yaml" || bad "config drifted — run python3 mobile/export_config.py"
import json,yaml,sys
c=json.load(open("mobile/assets/model_config.json")); b=yaml.safe_load(open("configs/base.yaml"))
g,i=c["generation"],b["inference"]
sys.exit(0 if (g["temperature"]==i["temperature"] and g["top_p"]==i["top_p"]
               and g["top_k"]==i["top_k"] and g["max_new_tokens"]==i["max_new_tokens"]) else 1)
PY

step "3. Dart parity corpus current"
python3 mobile/make_parity.py >/tmp/st_parity.log 2>&1 \
  && git diff --quiet mobile/test/guard_parity_cases.json 2>/dev/null \
  && ok "parity corpus matches the Python guard" \
  || ok "parity corpus regenerated (commit it, then run: dart test mobile/test/guard_test.dart)"

step "4. Dataset tooling"
python3 data/build_dataset.py refusals --n 20 --out /tmp/st_ref.jsonl >/dev/null 2>&1 \
  && ok "refusal generation" || bad "refusal generation"
python3 data/build_dataset.py safety --out /tmp/st_safety.jsonl >/dev/null 2>&1 \
  && ok "safety probe generation" || bad "safety probe generation"
python3 data/build_dataset.py refusals --n 200 --out /tmp/st_ref200.jsonl >/dev/null 2>&1
python3 data/check_leakage.py /tmp/st_ref200.jsonl /tmp/st_safety.jsonl >/tmp/st_leak.log 2>&1 \
  && ok "no leakage between refusal training and safety probes" \
  || { bad "LEAKAGE between training and evaluation"; sed -n '/vs/,$p' /tmp/st_leak.log | head -12; }
python3 - <<'PYEOF' && ok "no internal duplicates in generated refusals" || bad "duplicate refusal items"
import json,sys
q=[json.loads(l)["question"].lower().strip() for l in open("/tmp/st_ref200.jsonl") if l.strip()]
sys.exit(0 if len(q)==len(set(q)) else 1)
PYEOF
if python3 data/validate_dataset.py data/seed/train_sample.jsonl --kind train >/tmp/st_schema.log 2>&1; then
  ok "schema validation"
else
  bad "schema validation"           # show WHY, instead of hiding it
  sed -n '/ERROR\|error\|Traceback/,$p' /tmp/st_schema.log | head -12
  tail -5 /tmp/st_schema.log
fi

step "5. Guard evaluation"
python3 src/eval_guard.py --safety /tmp/st_safety.jsonl \
  --legit data/seed/eval_quality_sample.jsonl --out /tmp/st_guard.json >/dev/null 2>&1 \
  && ok "guard evaluation" || bad "guard evaluation"

step "6. Training config"
python3 src/train.py --model qwen05 --method lora --dry-run >/dev/null 2>&1 \
  && ok "train.py dry run" || bad "train.py dry run"

step "7. Evaluation chain (mock)"
rm -f /tmp/st_gen.jsonl /tmp/st_scores.jsonl
python3 src/generate.py --mock --arms fp16,lora,qat4 \
  --eval data/seed/eval_quality_sample.jsonl --out /tmp/st_gen.jsonl >/dev/null 2>&1 \
  && ok "generate" || bad "generate"
python3 src/score.py --mock-judge --generations /tmp/st_gen.jsonl \
  --out /tmp/st_scores.jsonl --workers 4 >/dev/null 2>&1 \
  && ok "score" || bad "score"
python3 src/stats.py /tmp/st_scores.jsonl >/dev/null 2>&1 \
  && ok "stats" || bad "stats"
python3 src/figures.py --scores /tmp/st_scores.jsonl --out /tmp/st_figs --mock >/dev/null 2>&1 \
  && ok "figures" || bad "figures"

step "7b. Human-agreement workflow"
rm -f /tmp/st_hs.jsonl /tmp/st_hs_key.jsonl
python3 src/score.py --sample-human --generations /tmp/st_gen.jsonl \
  --out /tmp/st_scores.jsonl >/dev/null 2>&1 || true
# score.py writes the sample beside the results file
HS=/tmp/human_sample.jsonl
if [ -f "$HS" ]; then
  python3 - <<'PYEOF' && ok "grading file is blind (no arm, no judge scores)" || bad "grading file leaks the key"
import json,sys
r=json.loads(open("/tmp/human_sample.jsonl").readline())
leak=[k for k in ("_judge_scores","_hidden_arm","arm","judge_scores") if k in r]
sys.exit(1 if leak else 0)
PYEOF
  [ -f /tmp/human_sample_key.jsonl ] && ok "key file written separately" || bad "no key file"
else
  bad "sample_human produced no file"
fi

step "7bb. Statistical implementations"
python3 src/test_stats.py >/tmp/st_stats.log 2>&1 \
  && ok "Dunn's, Cohen's kappa, Fleiss' kappa validated" \
  || { bad "statistics validation"; tail -12 /tmp/st_stats.log; }
grep -q "SKIP" /tmp/st_stats.log 2>/dev/null && \
  printf '   \033[33mNOTE\033[0m  reference libraries absent; edge cases only.\n'\
'          pip install scikit-learn scikit-posthocs for full validation\n'

step "7ba. Exact llama.cpp quantizer"
if python3 -c "import torch" 2>/dev/null; then
  python3 src/test_quant_sim.py >/tmp/st_qsim.log 2>&1 \
    && ok "Q4_0 / Q8_0 bit-identical to ggml reference; QAT machinery" \
    || { bad "quantizer tests"; grep FAIL /tmp/st_qsim.log | head -8; }
else
  printf '   \033[33mSKIP\033[0m  PyTorch not installed here (runs on Kaggle)\n'
fi

step "7c. Mock-figure guard"
# The realistic threat: a mock figure produced by src/figures.py --mock, copied
# on its own into another folder — leaving the directory's marker file behind.
# Only the per-file metadata stamp can catch that.
rm -rf /tmp/st_mock && mkdir -p /tmp/st_mock
cp /tmp/st_figs/fig1_score_box.pdf /tmp/st_mock/ 2>/dev/null || true
rm -rf /tmp/st_clean && mkdir -p /tmp/st_clean
if [ ! -f scripts/check_mock.py ]; then
  bad "scripts/check_mock.py is missing"
else
  set +e
  python3 scripts/check_mock.py /tmp/st_mock >/dev/null 2>&1; rc_mock=$?
  python3 scripts/check_mock.py /tmp/st_clean >/dev/null 2>&1; rc_clean=$?
  set -e
  # Exactly 1 on the mock, exactly 0 on the clean directory. Any other code
  # means the checker crashed, which must never be read as "detected".
  [ "$rc_mock" -eq 1 ] && ok "mock detector catches a copied mock figure (metadata stamp)" \
    || bad "mock detector returned $rc_mock on a copied mock figure (expected 1)"
  [ "$rc_clean" -eq 0 ] && ok "mock detector passes a clean directory" \
    || bad "mock detector returned $rc_clean on a clean directory (expected 0)"
fi

step "7d. Pipeline orchestrator"
python3 src/run_pipeline.py --dry-run >/tmp/st_plan.log 2>&1 \
  && ok "plan builds ($(grep -c '\[' /tmp/st_plan.log) stages)" \
  || { bad "plan failed"; tail -6 /tmp/st_plan.log; }

step "8. Paper builds"
if command -v pdflatex >/dev/null 2>&1; then
  (cd paper && make >/dev/null 2>&1) && ok "paper compiles" || bad "paper does not compile"
  if python3 scripts/check_mock.py paper/figures >/dev/null 2>&1 \
     && python3 scripts/check_mock.py paper/tables >/dev/null 2>&1; then
    ok "no mock figures or tables inside the paper folder"
  else
    bad "MOCK content inside paper/ — it would appear in the compiled paper"
    python3 scripts/check_mock.py paper/figures | head -8
  fi
else
  printf '   \033[33mSKIP\033[0m  no LaTeX toolchain\n'
fi

printf '\n'
if [ $FAIL -eq 0 ]; then
  printf '\033[32mAll smoke tests passed.\033[0m\n'
else
  printf '\033[31mSmoke tests FAILED.\033[0m\n'; exit 1
fi
