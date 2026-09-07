"""
Export a trained checkpoint to a deployable quantized model, and produce the
post-training quantization baselines.

    python src/quantize.py --run runs/qwen05_qat4 --verify-only
    python src/quantize.py --run runs/qwen05_qat4 --llama-cpp ~/llama.cpp

THE ONE THING THIS SCRIPT EXISTS TO ENFORCE
-------------------------------------------
Quantization-aware fine-tuning only pays off when the scheme simulated during
training matches the scheme actually deployed. Simulate group-32 symmetric int4
and then ship Q4_K_M — a mixed k-quant with a different block structure — and
you have not measured QAT. You have measured QAT-for-a-different-target
followed by an unrelated post-training quantization, and your central claim
collapses under questioning.

--verify-only runs that check without touching a model. Run it on Day 6.

GGUF SCHEMES
------------
    Q4_0     group 32, symmetric        <- matches the QAT simulation
    Q4_K_M   mixed k-quant, per-block   <- better quality, does NOT match
    Q8_0     group 32, symmetric 8-bit  <- matches the qat8 simulation

Report Q4_0 as your primary result. If you also report Q4_K_M, label the
scheme mismatch explicitly rather than letting the reader assume comparability.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Scheme properties of each GGUF type, for the match check.
GGUF_SCHEMES = {
    "q8_0":   {"bits": 8, "group_size": 32, "symmetric": True,  "mixed": False},
    "q4_0":   {"bits": 4, "group_size": 32, "symmetric": True,  "mixed": False},
    "q4_1":   {"bits": 4, "group_size": 32, "symmetric": False, "mixed": False},
    "q4_k_m": {"bits": 4, "group_size": None, "symmetric": False, "mixed": True},
    "q4_k_s": {"bits": 4, "group_size": None, "symmetric": False, "mixed": True},
    "q5_k_m": {"bits": 5, "group_size": None, "symmetric": False, "mixed": True},
}

# Which training method simulates which deployment scheme.
METHOD_EXPECTS = {
    "qat4": "q4_0", "qat8": "q8_0", "qat_mixed": "q4_k_m",
    "fp16": None, "lora": None, "qlora4": None, "ptq": None,
}


def verify_scheme_match(method: str, target: str, qcfg: dict) -> bool:
    """Returns True when the deployment scheme matches what QAT simulated."""
    target = target.replace("gguf_", "").lower()
    scheme = GGUF_SCHEMES.get(target)

    print("=" * 72)
    print("SCHEME MATCH CHECK")
    print("=" * 72)
    print(f"  training method     {method}")
    print(f"  deployment target   {target}")

    if scheme is None:
        print(f"\n  UNKNOWN target format {target!r}. Add it to GGUF_SCHEMES.")
        return False

    print(f"  configured          bits={qcfg['bits']} group={qcfg['group_size']} "
          f"symmetric={qcfg['symmetric']}")
    print(f"  {target} provides    bits={scheme['bits']} group={scheme['group_size']} "
          f"symmetric={scheme['symmetric']} mixed={scheme['mixed']}")
    print()

    if not method.startswith("qat"):
        print(f"  {method} is not a QAT arm — no simulation to match. OK.")
        return True

    problems = []
    if scheme["bits"] != qcfg["bits"]:
        problems.append(f"bit width {scheme['bits']} != configured {qcfg['bits']}")
    if scheme["mixed"]:
        problems.append(f"{target} is a mixed k-quant; a uniform groupwise QAT "
                        f"simulation cannot reproduce its block structure")
    if scheme["group_size"] is not None and scheme["group_size"] != qcfg["group_size"]:
        problems.append(f"group size {scheme['group_size']} != configured {qcfg['group_size']}")
    if scheme["symmetric"] != qcfg["symmetric"]:
        problems.append(f"symmetric={scheme['symmetric']} != configured {qcfg['symmetric']}")

    expected = METHOD_EXPECTS.get(method)
    if expected and expected != target:
        problems.append(f"{method} normally pairs with {expected}, not {target}")

    if problems:
        print("  MISMATCH:")
        for p in problems:
            print(f"    - {p}")
        print()
        print("  Your QAT result will not mean what you claim it means. Either change")
        print("  configs/base.yaml -> quantization to match the deployment format, or")
        print("  deploy the format your QAT actually simulated. Do not proceed and")
        print("  explain it away in the limitations section.")
        return False

    print("  MATCH. The simulated and deployed schemes agree.")
    return True


def merge_adapters(run_dir: Path) -> Path:
    """LoRA and QLoRA runs save adapters, not a full model. Merge before export."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = run_dir / "final"
    if not (ckpt / "adapter_config.json").exists():
        return ckpt

    merged = run_dir / "merged"
    if merged.exists():
        print(f"  reusing {merged}")
        return merged

    cfg = json.loads((ckpt / "adapter_config.json").read_text())
    base_id = cfg["base_model_name_or_path"]
    print(f"  merging adapters into {base_id}")

    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto")
    model = PeftModel.from_pretrained(model, ckpt).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(ckpt).save_pretrained(merged)
    return merged


def run_cmd(cmd: list[str]) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)


def export(model_dir: Path, out_dir: Path, llama_cpp: Path, targets: list[str]) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    convert = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = next(
        (p for p in [llama_cpp / "llama-quantize", llama_cpp / "build" / "bin" / "llama-quantize"]
         if p.exists()), None
    )
    if not convert.exists():
        raise SystemExit(f"convert_hf_to_gguf.py not found under {llama_cpp}")
    if quantize_bin is None:
        raise SystemExit(f"llama-quantize binary not found under {llama_cpp}. Build llama.cpp first.")

    f16 = out_dir / "model-f16.gguf"
    if not f16.exists():
        run_cmd([sys.executable, convert, model_dir, "--outfile", f16, "--outtype", "f16"])

    sizes = {"f16": f16.stat().st_size}
    for t in targets:
        dst = out_dir / f"model-{t}.gguf"
        run_cmd([quantize_bin, f16, dst, t.upper()])
        sizes[t] = dst.stat().st_size
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--llama-cpp", default="~/llama.cpp")
    ap.add_argument("--targets", default="q4_0,q8_0", help="also add q4_k_m to report both")
    ap.add_argument("--verify-only", action="store_true", help="scheme check only, no export")
    a = ap.parse_args()

    run_dir = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    qcfg = cfg["quantization"]

    meta_path = run_dir / "run_meta.json"
    method = json.loads(meta_path.read_text())["method"] if meta_path.exists() else "unknown"

    ok = verify_scheme_match(method, qcfg["target_format"], qcfg)
    if not ok and not a.verify_only:
        raise SystemExit("\nRefusing to export on a scheme mismatch. Fix the config first.")
    if a.verify_only:
        return 0 if ok else 1

    print()
    print("=" * 72)
    print("EXPORT")
    print("=" * 72)
    model_dir = merge_adapters(run_dir)
    llama_cpp = Path(a.llama_cpp).expanduser()
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    sizes = export(model_dir, run_dir / "gguf", llama_cpp, targets)

    print()
    print(f"{'format':<12}{'MB':>10}{'vs f16':>10}")
    print("-" * 32)
    base = sizes["f16"]
    for k, v in sizes.items():
        print(f"{k:<12}{v/1e6:>10.1f}{v/base:>9.2f}x")

    budget = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["device_budget"]
    print(f"\n  Usable device RAM: ~{budget['usable_mb']} MB")
    print("  File size is NOT peak RSS — add the KV cache, which grows with context")
    print("  length. Measure on the real device with src/benchmark.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
