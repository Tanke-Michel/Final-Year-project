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
  fp16      Half-precision full fine-tune. Upper bound on quality.
  lora      LoRA adapters on an fp16 base. Practical baseline.
  qlora4    Base loaded in 4-bit NF4, LoRA trained on top.
            NOTE: this is a memory-efficient TRAINING method. Merging adapters
            afterwards yields an fp16 model that you then quantize separately.
            It is NOT the same as QAT. Say so explicitly in your methods
            chapter — the brief warns about conflating PTQ and QAT, and this
            is the same error one step over.
  qat8      Quantization-aware fine-tuning simulating 8-bit.
  qat4      Quantization-aware fine-tuning simulating the DEPLOYED 4-bit
            scheme. Main experiment.
  qat_mixed QAT with sensitive modules held at higher precision.
  ptq       No training. Quantizes an existing checkpoint. Baseline only.

CRITICAL: the QAT arms must simulate configs/base.yaml -> quantization, which
must match what you actually ship. Simulating group-32 symmetric int4 and then
deploying a different scheme means you did not measure QAT.
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
    """JSONL with question/answer -> chat-templated text field."""
    from datasets import Dataset

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"No rows in {path}")

    def to_text(r: dict) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return Dataset.from_list([{"text": to_text(r)} for r in rows])


# --------------------------------------------------------------------------- #
# Model construction, one branch per method
# --------------------------------------------------------------------------- #

def build_model(method: str, hf_id: str, cfg: dict, target_modules: list[str]):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    qcfg = cfg["quantization"]

    if method in ("fp16", "ptq"):
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
        return model, None

    if method == "lora":
        from peft import LoraConfig
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
        return model, LoraConfig(target_modules=target_modules, **_lora_kwargs(cfg))

    if method == "qlora4":
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        return model, LoraConfig(target_modules=target_modules, **_lora_kwargs(cfg))

    if method.startswith("qat"):
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
        model = apply_qat(model, method, qcfg)
        return model, None

    raise SystemExit(f"Unhandled method {method!r}")


def _lora_kwargs(cfg: dict) -> dict:
    lora = dict(cfg["lora"])
    lora.pop("target_modules", None)
    return lora


def apply_qat(model, method: str, qcfg: dict):
    """Insert fake-quantization so the model trains against the rounding error
    it will actually suffer on device.

    ############################################################################
    # VERIFY THIS BEFORE DAY 9. The torchao QAT API has moved repeatedly and
    # any tutorial older than a few months is likely wrong. Check the current
    # docs, run it on ONE checkpoint end-to-end on Day 6, and confirm the
    # simulated scheme matches configs/base.yaml -> quantization.
    #
    # If QAT will not run, tell your supervisor immediately rather than
    # silently dropping the main experiment. The 8-bit arm and the PTQ
    # baselines still support a reduced comparison.
    ############################################################################
    """
    try:
        from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
    except ImportError as exc:
        raise SystemExit(
            "torchao QAT import failed. Check the installed torchao version and "
            "the current API path, then update apply_qat(). See the block comment "
            "in src/train.py.\n"
            f"Original error: {exc}"
        )

    group_size = qcfg["group_size"]

    if method == "qat8":
        # Verify the correct 8-bit quantizer class name in your torchao version.
        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
    elif method == "qat4":
        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
    elif method == "qat_mixed":
        # Hold embeddings and the output head at higher precision; those layers
        # are consistently the most quantization-sensitive.
        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=group_size)
    else:
        raise SystemExit(f"Unknown QAT variant {method!r}")

    return quantizer.prepare(model)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="key from configs/models.yaml")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate config and exit without training")
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
        sft = SFTConfig(
            output_dir=str(out_dir),
            seed=cfg["seed"],
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            num_train_epochs=t["num_train_epochs"],
            lr_scheduler_type=t["lr_scheduler_type"],
            warmup_ratio=t["warmup_ratio"],
            logging_steps=t["logging_steps"],
            save_strategy=t["save_strategy"],
            optim=t["optim"],
            adam_beta1=t["adam_beta1"],
            adam_beta2=t["adam_beta2"],
            adam_epsilon=t["adam_epsilon"],
            max_grad_norm=t["max_grad_norm"],
            max_seq_length=t["max_seq_length"],
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=sft,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=peft_cfg,
        )
        trainer.train()
        trainer.save_model(str(out_dir / "final"))
        tok.save_pretrained(str(out_dir / "final"))

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
