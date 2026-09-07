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

def load_model(run_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = run_dir / "final"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}. Train it first.")

    tok = AutoTokenizer.from_pretrained(ckpt)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype, device_map="auto")
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
        arms = [a.arm or run_dir.name]
        model, tok = load_model(run_dir)

    mode = "a" if a.append else "w"
    written = 0
    t0 = time.time()

    with out_path.open(mode, encoding="utf-8") as fh:
        for arm in arms:
            rng = random.Random(f"{cfg['seed']}:{arm}")
            for item in items:
                for gmode in guard_modes:
                    answers, interventions = [], []

                    for _ in range(n_repeats):
                        if a.mock:
                            raw = mock_generate(item["question"], item.get("medication", ""), arm, rng)
                        else:
                            raw = real_generate(model, tok, item["question"], cfg["system_prompt"], icfg)

                        if gmode == "on":
                            pre = guard.check_input(item["question"])
                            if pre.blocked:
                                answers.append(pre.text)
                                interventions.append({"stage": "input", "rule_id": pre.rule_id})
                                continue
                            post = guard.check_output(raw)
                            answers.append(post.text)
                            interventions.append(
                                {"stage": "output", "rule_id": post.rule_id} if post.blocked else {}
                            )
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
                    }) + "\n")
                    written += 1

            print(f"  {arm}: {len(items)} items x {len(guard_modes)} guard mode(s)")

    print(f"\nWrote {written} rows to {out_path} in {time.time()-t0:.1f}s")
    if a.mock:
        print("MOCK DATA — pipeline validation only. Do not report these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
