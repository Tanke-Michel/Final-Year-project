"""
Single parameterized training script. ONE script, one --method flag.

This is the Day 6 deliverable in the brief ("reproducible training pipeline").
Six separate scripts will cost you Days 7-10 in debugging; this costs you an
afternoon. Everything except --method comes from configs/base.yaml and is
identical across arms, which is what makes the comparison controlled.

    python src/train.py --model qwen05 --method lora
    python src/train.py --model qwen05 --method qat4

Methods
-------
  fp16      Full-parameter fine-tune (fp32 weights, fp16 arithmetic on a T4).
            Evaluated twice: unquantized (upper bound) and rounded to Q4_0
            with no training awareness (the post-training quantization baseline).
  lora      LoRA adapters on an fp16 base. Practical baseline.
  qlora4    Base loaded in 4-bit NF4, LoRA trained on top.
            NOTE: this is a memory-efficient TRAINING method. Merging adapters
            afterwards yields an fp16 model that you then quantize separately.
            It is NOT the same as QAT. Say so explicitly in your methods
            chapter — the brief warns about conflating PTQ and QAT, and this
            is the same error one step over.
  qat8      Quantization-aware fine-tuning against Q8_0.
  qat4      Quantization-aware fine-tuning against Q4_0. Main experiment.
  qat_mixed QAT against Q4_0 with the first and last blocks at Q8_0.
  ptq       No training; the PTQ baseline is the fp16 arm evaluated at Q4_0.

The QAT arms fake-quantize with src/quant_sim.py, an exact reimplementation of
llama.cpp's Q4_0/Q8_0 arithmetic (verified bit-identical against both the C
reference and llama.cpp's own gguf-py). Training, evaluation and the exported
GGUF therefore all see the same weights. See src/deploy_specs.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
METHODS = ["fp16", "lora", "qlora4", "qat8", "qat4", "qat_mixed", "ptq"]


# --------------------------------------------------------------------------- #
# Run metadata — written before training starts so a crashed run is still
# reconstructible. Your results table is only defensible if every number can
# be traced to an exact configuration.
# --------------------------------------------------------------------------- #

@dataclass
class RunMeta:
    run_id: str
    model_key: str
    hf_id: str
    method: str
    seed: int
    config_snapshot: dict
    git_commit: str | None
    python: str
    platform: str
    started_at: float
    finished_at: float | None = None
    status: str = "running"
    error: str | None = None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def load_configs(model_key: str) -> tuple[dict, dict]:
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    entry = next((m for m in models["candidates"] if m["key"] == model_key), None)
    if entry is None:
        keys = [m["key"] for m in models["candidates"]]
        raise SystemExit(f"Unknown model key {model_key!r}. Available: {keys}")
    if entry.get("hf_id") in (None, "TBD"):
        raise SystemExit(f"Model {model_key!r} has no hf_id set yet (Day 2 task).")
    return base, entry


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def build_dataset(path: Path, system_prompt: str, tokenizer):
    """JSONL with question/answer -> conversational prompt-completion rows.

    LOSS ON THE ANSWER ONLY. The earlier version rendered the whole
    conversation into one text field, so the loss also covered the system
    prompt and the question. The system prompt is identical in every example,
    so the model memorises it within a few steps and roughly a third of each
    step's tokens then contribute nothing; worse, the model is also trained to
    produce users' questions. With prompt/completion rows, TRL masks the prompt
    and computes the loss on the assistant's answer alone. It also renders the
    prompt exactly as inference does (chat template with a generation prompt),
    so training and evaluation see the same format.
    """
    from datasets import Dataset

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"No rows in {path}")

    return Dataset.from_list([{
        "prompt": [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": r["question"]}],
        "completion": [{"role": "assistant", "content": r["answer"]}],
    } for r in rows])


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #

def native_bf16() -> bool:
    """True only where bf16 runs in hardware (compute capability 8.0+).

    torch.cuda.is_bf16_supported() is the wrong test: on a T4 it returns True
    because bf16 can be EMULATED, several times slower than fp16. Kaggle's T4
    reported exactly that. Trusting it would have trained every arm in slow
    emulated precision.
    """
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def device_map():
    """One GPU if there is one, otherwise CPU. A single device on purpose: a
    0.5B model gains nothing from being split across Kaggle's two T4s. The
    notebook picks the GPU with the most free memory via CUDA_VISIBLE_DEVICES,
    so device 0 is always the right one."""
    import torch
    return {"": 0} if torch.cuda.is_available() else None


# --------------------------------------------------------------------------- #
# Model construction, one branch per method
# --------------------------------------------------------------------------- #

def build_model(method: str, hf_id: str, cfg: dict, target_modules: list[str]):
    import torch
    from transformers import AutoModelForCausalLM

    bf16 = native_bf16()
    on_gpu = torch.cuda.is_available()

    # Two dtypes, for two different reasons.
    #
    # FROZEN-BASE arms (lora, qlora4): the base never receives gradients, so it
    # can sit in half precision. PEFT keeps the adapter weights in fp32.
    #
    # FULL-PARAMETER arms (fp16, qat*): every weight is trained. On a GPU
    # without native bf16, the weights must stay fp32 ("master weights") while
    # autocast runs the arithmetic in fp16 — loading them in fp16 makes the
    # gradient scaler fail with "Attempting to unscale FP16 gradients". A 0.5B
    # model in fp32 with Adam is about 8 GB.
    half = torch.bfloat16 if bf16 else (torch.float16 if on_gpu else torch.float32)
    full = torch.bfloat16 if bf16 else torch.float32

    if method in ("fp16", "ptq"):
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=full, device_map=device_map())
        return model, None

    if method == "lora":
        from peft import LoraConfig
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=half, device_map=device_map())
        return model, LoraConfig(target_modules=target_modules, **_lora_kwargs(cfg))

    if method == "qlora4":
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=half,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map=device_map())
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        return model, LoraConfig(target_modules=target_modules, **_lora_kwargs(cfg))

    if method.startswith("qat"):
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=full, device_map=device_map())
        model = apply_qat(model, method)
        return model, None

    raise SystemExit(f"Unhandled method {method!r}")


def _lora_kwargs(cfg: dict) -> dict:
    lora = dict(cfg["lora"])
    lora.pop("target_modules", None)
    return lora


def apply_qat(model, method: str):
    """Quantization-aware training against the EXACT deployed quantizer.

    The fake quantizer is src/quant_sim.py, a reimplementation of llama.cpp's
    own Q4_0 / Q8_0 arithmetic. The forward pass therefore sees precisely the
    weights the phone will hold. An earlier version used torchao's symmetric
    int4, which rounds to a different grid — Q4_0 maps the largest-magnitude
    value to -8 with its sign kept, so its step is max/8, not max/7.5 — and
    whose import path changed between torchao releases. Dropping it removes
    both the mismatch and the dependency.

      qat4       every linear layer at Q4_0
      qat8       every linear layer at Q8_0
      qat_mixed  Q4_0, first and last transformer blocks at Q8_0
    """
    from quant_sim import deploy_spec_for_arm, prepare_qat

    spec = deploy_spec_for_arm(method)
    counts = prepare_qat(model, spec)
    if not counts:
        raise SystemExit("QAT attached no fake quantizers — nothing would be simulated.")
    print(f"  QAT ({spec}): " + ", ".join(f"{n} layers at {f}" for f, n in sorted(counts.items())))
    return model


def strip_fake_quant(model) -> int:
    """Remove the fake-quant wrappers before saving, keeping the trained latent
    weights, so the checkpoint is an ordinary model llama.cpp can convert."""
    from quant_sim import strip_qat
    return strip_qat(model)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="key from configs/models.yaml")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate config and exit without training")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after N optimizer steps. For environment smoke runs "
                         "on Kaggle: proves the arm runs on this GPU in minutes, "
                         "not the hour a full run takes. Never use for real results.")
    args = ap.parse_args()

    cfg, entry = load_configs(args.model)
    run_id = f"{args.model}_{args.method}"
    out_dir = Path(args.out or ROOT / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = RunMeta(
        run_id=run_id,
        model_key=args.model,
        hf_id=entry["hf_id"],
        method=args.method,
        seed=cfg["seed"],
        config_snapshot=cfg,
        git_commit=git_commit(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        started_at=time.time(),
    )
    (out_dir / "run_meta.json").write_text(json.dumps(asdict(meta), indent=2, default=str))
    print(f"[{run_id}] metadata written to {out_dir/'run_meta.json'}")

    if args.dry_run:
        print(f"[{run_id}] dry run OK — config valid, no training performed")
        return 0

    try:
        import torch
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        torch.manual_seed(cfg["seed"])

        tok = AutoTokenizer.from_pretrained(entry["hf_id"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        dataset = build_dataset(data_path, cfg["system_prompt"], tok)
        print(f"[{run_id}] {len(dataset)} training examples")

        if args.method == "ptq":
            print(f"[{run_id}] ptq is a post-hoc baseline — run src/quantize.py on an "
                  f"existing checkpoint instead of training here.")
            return 0

        model, peft_cfg = build_model(
            args.method, entry["hf_id"], cfg, entry.get("target_modules", ["q_proj", "v_proj"])
        )

        t = cfg["training"]
        import inspect
        import torch

        # fp16 on Kaggle: neither the T4 (sm_75) nor the P100 (sm_60) runs bf16
        # in hardware. See native_bf16() for why is_bf16_supported() is not used.
        use_bf16 = native_bf16()

        # Gradient checkpointing recomputes activations in the backward pass
        # instead of storing them. It does not change the maths, only memory
        # and time, so it cannot confound the comparison between arms. It is on
        # because the full-parameter arms are tight on a 15 GB T4: fp32 weights,
        # gradients and Adam state are about 8 GB before any activations, and a
        # 152k-token vocabulary makes the logits alone ~1 GB per sequence.
        grad_ckpt = bool(t.get("gradient_checkpointing", True))

        wanted = dict(
            output_dir=str(out_dir),
            seed=cfg["seed"],
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            num_train_epochs=t["num_train_epochs"],
            lr_scheduler_type=t["lr_scheduler_type"],
            warmup_ratio=t["warmup_ratio"],
            logging_steps=t["logging_steps"],
            save_strategy="no" if args.max_steps else t["save_strategy"],
            optim=t["optim"],
            adam_beta1=t["adam_beta1"],
            adam_beta2=t["adam_beta2"],
            adam_epsilon=t["adam_epsilon"],
            max_grad_norm=t["max_grad_norm"],
            bf16=use_bf16,
            fp16=(not use_bf16) and torch.cuda.is_available() and args.method != "qlora4",
            gradient_checkpointing=grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
            # Explicit, rather than relying on TRL's default for prompt-completion
            # data: the loss covers the assistant's answer only.
            completion_only_loss=True,
            report_to="none",
        )
        if args.max_steps:
            wanted["max_steps"] = args.max_steps

        # Library versions rename settings. Known renames are translated
        # EXPLICITLY; anything else unrecognised stops the run.
        #
        # An earlier version silently dropped unrecognised keys "for safety".
        # Run against transformers 5 it dropped warmup_ratio — every arm would
        # have trained with no learning-rate warmup, a silent change to the
        # fixed configuration. A config that cannot be applied must fail
        # loudly, not quietly become a different config.
        accepted = set(inspect.signature(SFTConfig.__init__).parameters)
        wanted["max_length" if "max_length" in accepted else "max_seq_length"] = t["max_seq_length"]
        if "warmup_ratio" not in accepted:
            # transformers 5: warmup_steps takes a float in [0, 1) as a ratio.
            wanted["warmup_steps"] = float(wanted.pop("warmup_ratio"))
        if wanted.get("gradient_checkpointing_kwargs") is None:
            wanted.pop("gradient_checkpointing_kwargs", None)

        unknown = sorted(k for k in wanted if k not in accepted)
        if unknown:
            raise SystemExit(
                f"[{run_id}] This TRL/transformers version does not accept {unknown}. "
                f"Add an explicit translation above rather than dropping them — "
                f"dropping a setting silently changes the experiment.")
        sft = SFTConfig(**wanted)
        precision = "bf16" if use_bf16 else ("fp16" if torch.cuda.is_available() else "fp32 (CPU)")
        print(f"[{run_id}] precision: {precision}   gradient checkpointing: {grad_ckpt}"
              f"{'  (smoke run: ' + str(args.max_steps) + ' steps)' if args.max_steps else ''}")

        trainer = SFTTrainer(
            model=model,
            args=sft,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=peft_cfg,
        )
        trainer.train()

        # QAT arms: remove the fake-quantize wrappers so the saved checkpoint is
        # an ordinary model that llama.cpp can convert. See strip_fake_quant.
        if args.method.startswith("qat"):
            n = strip_fake_quant(trainer.model)
            print(f"[{run_id}] stripped {n} fake-quant modules before saving")

        # Full-parameter checkpoints are saved in fp16. Evaluation loads them in
        # fp16 and llama.cpp converts from f16, so nothing downstream sees more
        # precision than this — and it halves the size (~1 GB instead of 2 GB
        # for 0.5B), which matters against Kaggle's 20 GB output limit.
        if peft_cfg is None and next(trainer.model.parameters()).dtype == torch.float32:
            trainer.model.to(torch.float16)

        trainer.save_model(str(out_dir / "final"))
        tok.save_pretrained(str(out_dir / "final"))

        if torch.cuda.is_available():
            # Peak memory is worth recording: it shows the headroom per arm on
            # the T4, and training cost is a legitimate secondary result.
            peak = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[{run_id}] peak GPU memory: {peak:.2f} GB of {total:.1f} GB")

        meta.status = "complete"

    except Exception as exc:
        meta.status = "failed"
        meta.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta.finished_at = time.time()
        (out_dir / "run_meta.json").write_text(json.dumps(asdict(meta), indent=2, default=str))

    print(f"[{run_id}] done in {meta.finished_at - meta.started_at:.0f}s -> {out_dir/'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
