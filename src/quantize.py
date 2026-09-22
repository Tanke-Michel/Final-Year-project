"""
Export a trained arm to the GGUF file that goes on the phone.

    python3 src/quantize.py --run runs/qwen05_qat4 --verify-only     # print the plan
    python3 src/quantize.py --run runs/qwen05_qat4 --llama-cpp ~/llama.cpp

The deployment spec comes from src/deploy_specs.py — the same definition the
quantization-aware training trained against and the evaluation simulated:

    qat8 -> q8_0     qat_mixed -> mixed     every other arm -> q4_0

Every llama-quantize choice is passed explicitly (embedding and output types,
per-block overrides for the mixed spec) rather than left to llama.cpp's
defaults, which have changed between versions. After export the GGUF is read
back and every tensor's type is checked against the spec — the file on the
phone is verified, not assumed.

WHERE TO RUN THIS. Conversion is memory-bound, not GPU-bound. Use a Kaggle CPU
session (no GPU quota) or your own machine, for the configurations you will
benchmark on the phone. The quality evaluation does not need GGUF files: it
scores the same weights through src/quant_sim.py.

Needs llama.cpp built (llama-quantize) and, for verification, `pip install gguf`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deploy_specs import (EMBEDDING_FORMAT, SPECS, deploy_spec_for_arm,  # noqa: E402
                          linear_format, llama_quantize_args)


def merge_adapters(run_dir: Path) -> Path:
    """LoRA and QLoRA runs save adapters, not a model. Merge them into the base
    in fp16 — exactly what src/generate.py does before evaluating."""
    ckpt = run_dir / "final"
    if not (ckpt / "adapter_config.json").exists():
        return ckpt
    merged = run_dir / "merged"
    if (merged / "config.json").exists():
        return merged

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = json.loads((ckpt / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"  merging adapter into {base_id}")
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(ckpt)).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(ckpt).save_pretrained(merged)
    return merged


def n_layers_of(model_dir: Path) -> int:
    cfg = json.loads((model_dir / "config.json").read_text())
    n = cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers")
    if not n:
        raise SystemExit(f"cannot read num_hidden_layers from {model_dir / 'config.json'}")
    return int(n)


def expected_type(tensor_name: str, spec: str, n_layers: int) -> str | None:
    """The type a GGUF tensor must have under the spec; None = not checked
    (norms, biases and other 1-D tensors stay F32 by design)."""
    if tensor_name in ("token_embd.weight", "output.weight"):
        return EMBEDDING_FORMAT.upper()
    if tensor_name.startswith("blk.") and tensor_name.endswith(".weight"):
        i = tensor_name.split(".")[1]
        fmt = linear_format(spec, f"model.layers.{i}.x", n_layers)
        return fmt.upper() if fmt else None
    return None


def verify_gguf(path: Path, spec: str, n_layers: int) -> list[str]:
    """Read the GGUF back and compare every checked tensor's type with the
    spec. Returns a list of problems (empty = verified)."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return ["cannot verify: pip install gguf"]

    problems, counts = [], {}
    for t in GGUFReader(str(path)).tensors:
        actual = t.tensor_type.name
        counts[actual] = counts.get(actual, 0) + 1
        if len(t.shape) < 2 or int(t.shape[0]) % 32:
            continue                            # 1-D or not block-aligned: not quantized
        want = expected_type(t.name, spec, n_layers)
        if want and actual != want:
            problems.append(f"{t.name}: {actual}, expected {want}")
    print("  tensor types: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return problems


def run_cmd(cmd: list[str]) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/qwen05_qat4")
    ap.add_argument("--llama-cpp", default="~/llama.cpp")
    ap.add_argument("--spec", choices=[s for s in SPECS if s != "none"], default=None,
                    help="override the arm's deployment spec (not recommended: the "
                         "arm was trained and evaluated in its own spec)")
    ap.add_argument("--verify-only", action="store_true",
                    help="print the spec and the commands that would run, then stop")
    a = ap.parse_args()

    run_dir = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    meta = run_dir / "run_meta.json"
    method = json.loads(meta.read_text())["method"] if meta.exists() else run_dir.name.split("_", 1)[-1]
    spec = a.spec or deploy_spec_for_arm(method)
    if a.spec and a.spec != deploy_spec_for_arm(method):
        print(f"  WARNING: {method} was trained and evaluated as {deploy_spec_for_arm(method)}, "
              f"exporting as {a.spec}. Report the mismatch if you use this file.")

    print("=" * 72)
    print(f"EXPORT  {run_dir.name}   method={method}   spec={spec}")
    print("=" * 72)

    if a.verify_only:
        n = n_layers_of(run_dir / "final") if (run_dir / "final" / "config.json").exists() else 24
        base, flags = llama_quantize_args(spec, n)
        print(f"  llama-quantize {' '.join(flags)} model-f16.gguf model-{spec}.gguf {base}")
        print("  (plan only — nothing exported)")
        return 0

    model_dir = merge_adapters(run_dir)
    n = n_layers_of(model_dir)
    base, flags = llama_quantize_args(spec, n)

    llama = Path(a.llama_cpp).expanduser()
    convert = llama / "convert_hf_to_gguf.py"
    quant_bin = next((p for p in (llama / "llama-quantize", llama / "build" / "bin" / "llama-quantize")
                      if p.exists()), None)
    if not convert.exists() or quant_bin is None:
        raise SystemExit(f"llama.cpp not found (or not built) under {llama}.\n"
                         "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && "
                         "cmake -B build && cmake --build build -j --target llama-quantize")

    out_dir = run_dir / "gguf"
    out_dir.mkdir(exist_ok=True)
    f16 = out_dir / "model-f16.gguf"
    if not f16.exists():
        # f16, not f32: the evaluation loaded the weights in fp16 before
        # rounding, so quantizing from f16 reproduces exactly what was scored.
        run_cmd([sys.executable, convert, model_dir, "--outfile", f16, "--outtype", "f16"])
    dst = out_dir / f"model-{spec}.gguf"
    run_cmd([quant_bin, *flags, f16, dst, base])

    print()
    problems = verify_gguf(dst, spec, n)
    if problems:
        print(f"  VERIFICATION FAILED ({len(problems)}):")
        for p in problems[:15]:
            print(f"    - {p}")
        print("  The file does not match what was trained and evaluated. Do not")
        print("  benchmark it; check the llama.cpp version's quantize flags.")
        return 1
    print(f"  verified: every tensor matches spec {spec}")
    print(f"  {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
