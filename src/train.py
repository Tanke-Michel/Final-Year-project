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

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Two different dtypes, for two different reasons.
    #
    # FROZEN-BASE arms (lora): the base weights never receive gradients, so they
    # can sit in half precision to save memory.
    #
    # FULL-PARAMETER arms (fp16, qat*): every weight is trained. On a T4 or P100,
    # which lack bf16, loading those weights in fp16 AND enabling fp16 mixed
    # precision makes PyTorch's gradient scaler fail with "Attempting to unscale
    # FP16 gradients". Correct fp16 training keeps fp32 master weights and lets
    # autocast run the arithmetic in fp16 — which is what "fp16 fine-tuning"
    # actually means. A 0.5B model in fp32 with Adam is about 8 GB, within a
    # 16 GB T4.
    half = torch.bfloat16 if bf16_ok else torch.float16
    full_param = torch.bfloat16 if bf16_ok else torch.float32
    dtype = half
    qcfg = cfg["quantization"]

    if method in ("fp16", "ptq"):
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=full_param, device_map={"": 0})
        return model, None

    if method == "lora":
        from peft import LoraConfig
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map={"": 0})
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
            hf_id, quantization_config=bnb, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(model)
        return model, LoraConfig(target_modules=target_modules, **_lora_kwargs(cfg))

    if method.startswith("qat"):
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=full_param, device_map={"": 0})
        model = apply_qat(model, method, qcfg)
        return model, None

    raise SystemExit(f"Unhandled method {method!r}")


def _lora_kwargs(cfg: dict) -> dict:
    lora = dict(cfg["lora"])
    lora.pop("target_modules", None)
    return lora


def apply_qat(model, method: str, qcfg: dict):
    """Insert fake quantization so the model trains against the rounding error
    it will suffer on device.

    WHAT IS SIMULATED, AND WHY. The deployed format is GGUF, quantized by
    llama.cpp. For Q4_0 that means WEIGHT-ONLY, SYMMETRIC int4 in groups of 32.
    So qat4 simulates exactly that. torchao's popular Int8DynActInt4 preset is
    the wrong target here: it also fake-quantizes activations per token, which
    is the ExecuTorch/XNNPACK recipe, not the llama.cpp one. Training against
    a representation you do not deploy measures nothing.

    (llama.cpp does quantize activations to 8-bit in blocks of 32 at runtime.
    The simulation does not model that. It is benign relative to the weight
    error and is a small limitation worth one sentence in the methods.)

    API: torchao's recommended interface is quantize_ with QATConfig. The legacy
    Quantizer classes are documented as likely to be removed, and the
    torchao.quantization.prototype.qat path this file originally used no longer
    exists. Both current routes are tried, newest first.
    """
    import torch

    bits = 8 if method == "qat8" else 4
    group = qcfg.get("group_size", 32)
    dtype = torch.int8 if bits == 8 else torch.int4

    # Route 1: current API.
    try:
        from torchao.quantization import quantize_
        from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig

        wcfg = IntxFakeQuantizeConfig(dtype, group_size=group, is_symmetric=True)

        if method == "qat_mixed":
            # Hold the most quantization-sensitive modules out of the simulation.
            # In Gemma-3-270M roughly 63% of parameters are embeddings, so this
            # choice protects most of that model. lm_head and embeddings are
            # not nn.Linear-with-fake-quant targets anyway; exclude the first and
            # last transformer blocks as well, which are consistently the most
            # sensitive.
            n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 0)
            protected = {f"layers.{i}." for i in (0, n_layers - 1)} if n_layers else set()

            def filt(mod, fqn):
                return (isinstance(mod, torch.nn.Linear)
                        and "lm_head" not in fqn
                        and not any(p in fqn + "." for p in protected))

            quantize_(model, QATConfig(weight_config=wcfg, step="prepare"), filter_fn=filt)
        else:
            quantize_(model, QATConfig(weight_config=wcfg, step="prepare"),
                      filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear)
                      and "lm_head" not in fqn)

        n = sum(1 for m in model.modules() if "FakeQuant" in type(m).__name__)
        print(f"  QAT prepared via QATConfig: int{bits} weight-only, group {group}, "
              f"symmetric -> {n} fake-quantized modules")
        if n == 0:
            raise SystemExit("QAT inserted no fake-quantized modules — nothing would "
                             "be simulated. Check the torchao version.")
        return model

    except ImportError as e1:
        route1 = e1

    # Route 2: legacy Quantizer class. Coarser: it has no weight-only symmetric
    # int4 option, so the simulated scheme differs from Q4_0. Record it.
    try:
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
        print("  WARNING: falling back to the legacy Int8DynActInt4 quantizer.")
        print("  It also fake-quantizes activations, which Q4_0 deployment does")
        print("  not match exactly. Upgrade torchao and state this if you keep it.")
        try:
            q = Int8DynActInt4WeightQATQuantizer(group_size=group)
        except TypeError:
            q = Int8DynActInt4WeightQATQuantizer(groupsize=group)
        return q.prepare(model)
    except ImportError as e2:
        raise SystemExit(
            "No usable torchao QAT API found.\n"
            f"  current API: {route1}\n"
            f"  legacy API:  {e2}\n"
            "Install a recent torchao:  pip install -U torchao")


def strip_fake_quant(model) -> int:
    """Replace every fake-quantized Linear with a plain nn.Linear carrying the
    trained weights. Returns the number replaced.

    Deliberately NOT torchao's convert step. Convert would quantize the weights
    into torchao's own int4 layout, which (a) is not what llama.cpp consumes and
    (b) needs recent-GPU kernels that Kaggle's T4 and P100 lack. What llama.cpp
    wants is ordinary high-precision weights that have LEARNED to survive
    rounding — which is exactly what QAT training produces. Export those, and
    let llama.cpp apply the matching scheme.
    """
    import torch

    replaced = 0
    for name, mod in list(model.named_modules()):
        if "FakeQuant" not in type(mod).__name__ or not isinstance(mod, torch.nn.Linear):
            continue
        plain = torch.nn.Linear(mod.in_features, mod.out_features,
                                bias=mod.bias is not None,
                                device=mod.weight.device, dtype=mod.weight.dtype)
        with torch.no_grad():
            plain.weight.copy_(mod.weight)
            if mod.bias is not None:
                plain.bias.copy_(mod.bias)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, plain)
        replaced += 1
    return replaced


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

        # fp16 on Kaggle. Neither the T4 (sm_75) nor the P100 (sm_60) supports
        # bf16, so mixed precision must be fp16 there.
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

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
            report_to="none",
        )
        if args.max_steps:
            wanted["max_steps"] = args.max_steps

        # TRL renamed max_seq_length to max_length. Pass whichever this installed
        # version accepts, and drop anything it does not recognise, so a TRL
        # upgrade on Kaggle does not break the run with an unexpected-keyword error.
        accepted = set(inspect.signature(SFTConfig.__init__).parameters)
        seq_key = "max_length" if "max_length" in accepted else "max_seq_length"
        wanted[seq_key] = t["max_seq_length"]
        dropped = sorted(k for k in wanted if k not in accepted)
        sft = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})
        if dropped:
            print(f"[{run_id}] note: this TRL version ignores {dropped}")
        print(f"[{run_id}] precision: {'bf16' if use_bf16 else 'fp16'}"
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
