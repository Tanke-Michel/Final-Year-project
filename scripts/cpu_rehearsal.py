"""
CPU rehearsal: run the REAL training and evaluation code end to end, on a tiny
model, with no GPU and no internet.

    python3 scripts/cpu_rehearsal.py

Needs torch, transformers, trl, peft and datasets. It builds a tiny
Qwen2-architecture model and tokenizer locally, then for every arm:

  1. trains it with src/train.py for a few steps,
  2. checks the saved checkpoint is an ordinary, loadable model,
  3. scores it with src/generate.py in its deployed form (simulated Q4_0,
     Q8_0 or mixed), through the safety guard.

The numbers it produces mean nothing — the model is random and tiny. What it
proves is that every code path runs, in the same library versions as Kaggle,
before any GPU time is spent. The QLoRA arm needs bitsandbytes on a CUDA GPU
and is skipped on CPU; the Kaggle environment check covers it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["lora", "fp16", "qat4", "qat8", "qat_mixed"]


def build_tiny_model(dest: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    texts = []
    for f in ("data/seed/train_sample.jsonl", "data/seed/eval_quality_sample.jsonl"):
        for line in (ROOT / f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                texts += [r.get("question", ""), r.get("answer", ""), r.get("reference", "")]
    texts.append((ROOT / "configs" / "base.yaml").read_text())

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=640, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))

    # ChatML, the same layout Qwen2.5 uses.
    template = ("{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content']"
                " + '<|im_end|>\n'}}{% endfor %}"
                "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", chat_template=template)
    tok.save_pretrained(dest)

    # Every linear width is a multiple of 32, as the quantizer requires.
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=2048, tie_word_embeddings=True,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(dest)


def run(cmd: list[str], cwd: Path) -> tuple[bool, str, float]:
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode == 0, r.stdout + "\n" + r.stderr, time.time() - t0


def main() -> int:
    try:
        import peft, torch, transformers, trl  # noqa: F401,E401
    except ImportError as e:
        print(f"  [SKIP] {e.name} not installed — this rehearsal needs the training stack")
        return 0

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
          f"trl {trl.__version__} | peft {peft.__version__}")

    work = Path(tempfile.mkdtemp(prefix="ohai_rehearsal_"))
    for d in ("src", "data", "configs"):
        shutil.copytree(ROOT / d, work / d)
    model_dir = work / "tiny-model"
    build_tiny_model(model_dir)

    (work / "configs" / "models.yaml").write_text(json.dumps({"candidates": [{
        "key": "tiny", "hf_id": str(model_dir), "params_total": 1,
        "role": "primary", "target_modules": ["q_proj", "v_proj"]}]}))

    sys.path.insert(0, str(work / "src"))
    from deploy_specs import deploy_spec_for_arm

    py = sys.executable
    rows, failed = [], []

    for arm in ARMS:
        ok, log, dt = run([py, "src/train.py", "--model", "tiny", "--method", arm,
                           "--data", "data/seed/train_sample.jsonl", "--max-steps", "3"], work)
        final = work / "runs" / f"tiny_{arm}" / "final"
        saved = any((final / f).exists() for f in
                    ("model.safetensors", "adapter_model.safetensors", "pytorch_model.bin"))
        clean = True
        if ok and saved and (final / "model.safetensors").exists():
            from safetensors import safe_open
            with safe_open(final / "model.safetensors", "pt") as fh:
                clean = not any("parametrizations" in k for k in fh.keys())
        train_ok = ok and saved and clean
        rows.append((f"train {arm}", train_ok, dt))
        if not train_ok:
            failed.append((f"train {arm}", log))
            continue

        specs = (["none"] if arm == "fp16" else []) + [deploy_spec_for_arm(arm)]
        for spec in specs:
            out = work / "results" / "gen.jsonl"
            ok, log, dt = run([py, "src/generate.py", "--run", f"runs/tiny_{arm}",
                               "--deploy", spec, "--eval", "data/seed/eval_quality_sample.jsonl",
                               "--out", str(out), "--append", "--batch-size", "8"], work)
            label = f"tiny_{arm}@{spec}"
            got = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
            mine = [g for g in got if g["base_arm"] == label]
            gen_ok = ok and len(mine) == 10 and all(len(g["answers"]) == 3 for g in mine)
            rows.append((f"score {label}", gen_ok, dt))
            if not gen_ok:
                failed.append((f"score {label}", log))

    print()
    print("=" * 62)
    print(f"{'stage':<40}{'result':<8}{'time':>8}")
    print("-" * 62)
    for name, ok, dt in rows:
        print(f"{name:<40}{'PASS' if ok else 'FAIL':<8}{dt:>7.1f}s")
    print("=" * 62)

    for name, log in failed:
        print(f"\n--- {name}: last lines ---")
        print("\n".join(log.strip().splitlines()[-15:]))

    shutil.rmtree(work, ignore_errors=True)
    n_ok = sum(ok for _, ok, _ in rows)
    print(f"\n{n_ok}/{len(rows)} stages passed on CPU. QLoRA needs a CUDA GPU and is "
          f"covered by the Kaggle environment check.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
