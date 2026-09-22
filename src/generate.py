"""
Batch generation for evaluation.

Produces results/generations.jsonl, which src/score.py consumes.

    # smoke-test the whole pipeline today, no GPU, no torch
    python src/generate.py --mock --arms fp16,lora,qlora4,qat4,ptq4

    # real run
    python src/generate.py --run runs/qwen05_qat4 --eval data/eval_quality.jsonl

Two design points that matter for your results chapter.

GUARDED vs UNGUARDED
    --guard both runs each question twice: once with the deterministic safety
    layer active, once without. You need both. The unguarded numbers measure
    what fine-tuning achieved; the difference between them measures what the
    guard contributes. Reporting only the guarded numbers hides which component
    is doing the work, and a jury will ask.

REPEATS
    n_repeats from configs/base.yaml (default 3). Sampling is on, so the same
    question produces different answers. That is what SCORE's Reproducibility
    domain measures, and it is the axis nobody has connected to quantization.
    One extra inference pass, one extra results section.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety.guard import Guard  # noqa: E402


# --------------------------------------------------------------------------- #
# Mock backend — lets you validate generate -> score -> stats before any model
# exists. Run this on Day 6 so that Day 11 is execution, not debugging.
# --------------------------------------------------------------------------- #

_MOCK_GOOD = [
    "{med} is generally used to help manage the condition it was prescribed for. "
    "It works by acting on specific processes in the body. Common effects include "
    "mild stomach upset and headache. Speak to a pharmacist or doctor about how it "
    "applies to your own situation.",
    "{med} belongs to a recognised class of medicines. It is usually taken with food "
    "to reduce irritation. Tell a health professional if you notice a rash, swelling "
    "or difficulty breathing, as these may indicate an allergic reaction.",
]
_MOCK_DEGRADED = [
    "{med} is a medicine. It is used for things. Ask a doctor.",
    "{med} {med} {med} is used for the condition and is used for the condition and "
    "is used for the condition.",
    "Take 500 mg of {med} twice a day.",          # unsafe: should be caught by OUT-01
    "ok",                                          # degenerate: caught by OUT-04
]

# Degradation probability per arm — deliberately ordered so the mock produces a
# plausible-looking gradient and you can confirm the stats pipeline separates it.
_MOCK_DEGRADE_RATE = {
    "fp16": 0.10, "lora": 0.15, "qat4": 0.28,
    "qlora4": 0.35, "qat8": 0.22, "qat_mixed": 0.25, "ptq4": 0.60,
}


def mock_generate(question: str, med: str, arm: str, rng: random.Random) -> str:
    rate = _MOCK_DEGRADE_RATE.get(arm, 0.3)
    pool = _MOCK_DEGRADED if rng.random() < rate else _MOCK_GOOD
    return rng.choice(pool).format(med=med or "this medicine")


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #

def load_model(run_dir: Path, deploy: str = "q4_0"):
    """Load a trained checkpoint IN THE FORM THE PHONE WILL RUN IT.

    This is the step that makes the evaluation measure the research question.
    The earlier version scored the full-precision checkpoint, never the
    quantized one, so every arm was compared in fp16 — and the whole point of
    quantization-aware training only shows up after quantization. Now:

      1. adapter checkpoints (lora, qlora4) are merged into the base, exactly
         as src/quantize.py merges them before export;
      2. the weights are loaded in fp16, matching the f16 GGUF that
         llama-quantize starts from;
      3. every weight is rounded with llama.cpp's own arithmetic under the
         arm's deployment spec (src/quant_sim.py).

    deploy="none" skips step 3, for the unquantized upper bound.
    """
    import json as _json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ROOT / "src"))
    from quant_sim import simulate_deployment

    ckpt = run_dir / "final"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}. Train it first.")

    on_gpu = torch.cuda.is_available()
    bf16 = on_gpu and torch.cuda.get_device_capability(0)[0] >= 8
    # bf16 only where it runs in hardware; is_bf16_supported() says True on a
    # T4 because bf16 can be emulated, slowly.
    dtype = torch.bfloat16 if bf16 else (torch.float16 if on_gpu else torch.float32)
    where = {"": 0} if on_gpu else None

    tok = AutoTokenizer.from_pretrained(ckpt)
    adapter_cfg = ckpt / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        base_id = _json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map=where)
        model = PeftModel.from_pretrained(base, str(ckpt)).merge_and_unload()
        print(f"  merged adapter into {base_id}")
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype, device_map=where)

    counts = simulate_deployment(model, deploy)
    if deploy == "none":
        print("  evaluating UNQUANTIZED weights (upper bound)")
    else:
        print(f"  simulated deployment {deploy}: " +
              ", ".join(f"{n} tensors at {f}" for f, n in sorted(counts.items())))
        if not counts:
            raise SystemExit(f"Deployment spec {deploy} quantized nothing — refusing to "
                             "report unquantized results under a quantized label.")
    model.eval()
    return model, tok


def real_generate(model, tok, question: str, system_prompt: str, icfg: dict) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=icfg["max_new_tokens"],
            temperature=icfg["temperature"],
            do_sample=icfg["do_sample"],
            top_p=icfg["top_p"],
            top_k=icfg["top_k"],
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def real_generate_batch(model, tok, questions: list[str], system_prompt: str,
                        icfg: dict, n_repeats: int, batch_size: int) -> list[list[str]]:
    """Batched generation. Returns one list of n_repeats answers per question.

    Batching matters on a quota. One question at a time, a T4 spends most of
    each step idle; batching fills it. Rough figures for this project: about
    13 GPU-hours unbatched across the sweep, under 2 batched.

    Decoder-only models must be LEFT-padded for batched generation, or the
    shorter prompts get padding between the prompt and the generated text and
    the output degrades. The tokenizer's padding side is set here rather than
    trusted, because several small models ship with right-padding by default.
    """
    import torch

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True) for q in questions]

    # Repeat each prompt n_repeats times so every sample is an independent draw.
    flat = [(qi, p) for qi, p in enumerate(prompts) for _ in range(n_repeats)]
    out: list[list[str]] = [[] for _ in questions]

    total = len(flat)
    for start in range(0, total, batch_size):
        chunk = flat[start:start + batch_size]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=icfg["max_new_tokens"],
                temperature=icfg["temperature"],
                do_sample=icfg["do_sample"],
                top_p=icfg["top_p"],
                top_k=icfg["top_k"],
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        for (qi, _), seq in zip(chunk, gen):
            out[qi].append(tok.decode(seq[prompt_len:], skip_special_tokens=True).strip())
        done = min(start + batch_size, total)
        if done % (batch_size * 5) == 0 or done == total:
            print(f"    generated {done}/{total}")
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory, e.g. runs/qwen05_qat4")
    ap.add_argument("--arm", help="arm label for output rows (defaults to run dir name)")
    ap.add_argument("--eval", default="data/seed/eval_quality_sample.jsonl")
    ap.add_argument("--out", default="results/generations.jsonl")
    ap.add_argument("--guard", choices=["on", "off", "both"], default="both")
    ap.add_argument("--mock", action="store_true", help="no model — validate the pipeline")
    ap.add_argument("--arms", default="fp16,lora,qlora4,qat4,ptq4", help="mock mode only")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="generation batch size. 16 suits a 0.5B model on a 16 GB T4; "
                         "reduce if you hit out-of-memory")
    ap.add_argument("--deploy", default="q4_0", choices=["none", "q4_0", "q8_0", "mixed"],
                    help="deployment spec to simulate before generating. Scores the "
                         "model the phone will run. 'none' = unquantized upper bound.")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    icfg = cfg["inference"]
    n_repeats = icfg.get("n_repeats", 3)

    eval_path = Path(a.eval) if Path(a.eval).is_absolute() else ROOT / a.eval
    items = [json.loads(l) for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not items:
        raise SystemExit(f"No items in {eval_path}")

    out_path = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    guard = Guard()
    guard_modes = ["off", "on"] if a.guard == "both" else [a.guard]

    if a.mock:
        arms = [s.strip() for s in a.arms.split(",") if s.strip()]
        model = tok = None
    else:
        if not a.run:
            raise SystemExit("--run is required unless --mock is set")
        run_dir = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
        arms = [a.arm or f"{run_dir.name}@{a.deploy}"]
        model, tok = load_model(run_dir, a.deploy)

    mode = "a" if a.append else "w"
    written = 0
    t0 = time.time()

    with out_path.open(mode, encoding="utf-8") as fh:
        for arm in arms:
            rng = random.Random(f"{cfg['seed']}:{arm}")

            # GENERATE ONCE, THEN APPLY BOTH GUARD CONDITIONS TO THE SAME OUTPUTS.
            #
            # The earlier version sampled separately for guard-on and guard-off.
            # With sampling enabled those are different model outputs, so the
            # paired test meant to isolate the guard's effect was measuring the
            # guard PLUS sampling noise — and the pairing was not a true pairing.
            # It also doubled GPU time, which on a 30-hour weekly quota matters.
            #
            # Now the raw outputs are shared. Guard-off shows them as generated;
            # guard-on passes the same text through the guard. Any difference
            # between the two conditions is attributable to the guard alone.
            questions = [item["question"] for item in items]
            if a.mock:
                raw_all = [[mock_generate(q, it.get("medication", ""), arm, rng)
                            for _ in range(n_repeats)]
                           for q, it in zip(questions, items)]
            else:
                raw_all = real_generate_batch(model, tok, questions, cfg["system_prompt"],
                                              icfg, n_repeats, a.batch_size)

            for item, raws in zip(items, raw_all):
                pre = guard.check_input(item["question"])

                for gmode in guard_modes:
                    answers, interventions = [], []
                    for raw in raws:
                        if gmode == "on":
                            if pre.blocked:
                                # The deployed system never shows the model this
                                # question. The raw output exists only because
                                # the same sample serves the unguarded condition.
                                answers.append(pre.text)
                                interventions.append({"stage": "input", "rule_id": pre.rule_id})
                                continue
                            post = guard.check_output(raw)
                            answers.append(post.text)
                            interventions.append(
                                {"stage": "output", "rule_id": post.rule_id} if post.blocked else {})
                        else:
                            answers.append(raw)
                            interventions.append({})

                    fh.write(json.dumps({
                        "arm": f"{arm}+guard" if gmode == "on" else arm,
                        "base_arm": arm,
                        "guard": gmode,
                        "item_id": item["id"],
                        "question": item["question"],
                        "reference": item.get("reference", item.get("answer", "")),
                        "domain": item.get("domain"),
                        "difficulty": item.get("difficulty"),
                        "answers": answers,
                        "interventions": [i for i in interventions if i],
                        "n_repeats": n_repeats,
                        "shared_raw_outputs": True,
                    }) + "\n")
                    written += 1

            print(f"  {arm}: {len(items)} items x {n_repeats} samples, "
                  f"shared across {len(guard_modes)} guard condition(s)")

    print(f"\nWrote {written} rows to {out_path} in {time.time()-t0:.1f}s")
    if a.mock:
        print("MOCK DATA — pipeline validation only. Do not report these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
