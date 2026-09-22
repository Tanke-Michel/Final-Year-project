"""
Tests for the exact llama.cpp quantizer simulation.

    python3 src/test_quant_sim.py

Needs PyTorch. The study's central claim — that training, evaluation and the
deployed model see the same weights — rests on src/quant_sim.py being a
faithful copy of ggml's arithmetic. These tests check the vectorised code
against a scalar, line-by-line translation of the C reference functions, and
check the training machinery around it.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def f32(x: float) -> float:
    """Round to IEEE float32, as C float arithmetic does."""
    return struct.unpack("f", struct.pack("f", x))[0]


def f16(x: float) -> float:
    """Round to IEEE float16 (round-to-nearest-even), as GGML_FP32_TO_FP16."""
    return struct.unpack("e", struct.pack("e", x))[0]


# --- scalar translations of ggml-quants.c -----------------------------------

def ref_q4_0_block(xs: list[float]) -> list[float]:
    """quantize_row_q4_0_ref + dequantize_row_q4_0, one block of 32."""
    amax, mx = 0.0, 0.0
    for v in xs:                                  # if (amax < fabsf(v)) { amax = fabsf(v); max = v; }
        if amax < abs(v):
            amax, mx = abs(v), v
    d = f32(mx / -8.0)
    inv = f32(1.0 / d) if d else 0.0
    d16 = f16(d)
    out = []
    for v in xs:
        t = f32(f32(v * inv) + 8.5)
        q = min(15, int(t))                       # (int8_t) truncates toward zero
        out.append(f32((q - 8) * d16))
    return out


def ref_q8_0_block(xs: list[float]) -> list[float]:
    """quantize_row_q8_0_ref + dequantize_row_q8_0, one block of 32."""
    amax = 0.0
    for v in xs:
        amax = max(amax, abs(v))
    d = f32(amax / 127.0)
    inv = f32(1.0 / d) if d else 0.0
    d16 = f16(d)
    out = []
    for v in xs:
        t = f32(v * inv)
        q = math.copysign(math.floor(abs(t) + 0.5), t)   # roundf: half away from zero
        out.append(f32(q * d16))
    return out


def main() -> int:
    try:
        import torch
    except ImportError:
        print("  [SKIP] PyTorch not installed")
        return 0

    from quant_sim import (FakeQuant, prepare_qat, qdq_q4_0, qdq_q8_0,
                           simulate_deployment, strip_qat)
    from deploy_specs import linear_format, llama_quantize_args

    torch.manual_seed(0)

    print("=" * 70)
    print("QUANTIZER vs SCALAR TRANSLATION OF ggml C REFERENCE")
    print("=" * 70)
    # Realistic weights, plus adversarial blocks: all zero, ties between +a and
    # -a, a single outlier, and values sitting exactly on rounding boundaries.
    rows = [torch.randn(32) * 0.02 for _ in range(300)]
    rows += [torch.zeros(32),
             torch.tensor([0.5, -0.5] * 16),
             torch.tensor([-0.5, 0.5] * 16),
             torch.cat([torch.full((31,), 0.001), torch.tensor([3.0])]),
             torch.cat([torch.tensor([-3.0]), torch.full((31,), 0.001)]),
             torch.linspace(-1, 1, 32),
             torch.arange(32, dtype=torch.float32) / 8.0 - 2.0]
    w = torch.stack(rows)

    for name, vec, ref in (("Q4_0", qdq_q4_0, ref_q4_0_block),
                           ("Q8_0", qdq_q8_0, ref_q8_0_block)):
        got = vec(w)
        mism = 0
        worst = 0.0
        for r in range(w.shape[0]):
            exp = ref(w[r].tolist())
            for a, b in zip(got[r].tolist(), exp):
                if a != b:
                    mism += 1
                    worst = max(worst, abs(a - b))
        check(f"{name} bit-identical to the C reference", mism == 0,
              f"{w.numel()} values, {mism} differ" + (f", worst {worst:.2e}" if mism else ""))

    # Q4_0 is not symmetric int4: the largest-magnitude value keeps its sign
    # and lands exactly on -8 * d, so it is reproduced exactly.
    # Independent second reference: llama.cpp's own Python quantizers, shipped
    # in the `gguf` package. Optional — pip install gguf to run it.
    try:
        import numpy as np
        from gguf import GGMLQuantizationType as GT
        from gguf import quants
        wn = w.numpy()
        for name, qt, vec in (("Q4_0", GT.Q4_0, qdq_q4_0), ("Q8_0", GT.Q8_0, qdq_q8_0)):
            ref_np = quants.dequantize(quants.quantize(wn, qt), qt)
            diff = int((ref_np != vec(w).numpy()).sum())
            check(f"{name} bit-identical to llama.cpp's gguf-py quantizer", diff == 0,
                  f"{wn.size} values, {diff} differ")
    except ImportError:
        print("  [SKIP] gguf not installed — second reference check (pip install gguf)")

    blk = torch.cat([torch.full((31,), 0.1), torch.tensor([0.8])])
    check("Q4_0 reproduces the block's extreme value exactly",
          float(qdq_q4_0(blk.view(1, 32))[0, -1]) == float(f16(0.8 / -8) * -8))

    check("Q4_0 uses at most 16 distinct levels per block",
          all(len(set(qdq_q4_0(w)[r].tolist())) <= 16 for r in range(w.shape[0])))

    check("Q4_0 is idempotent (re-quantizing changes nothing)",
          torch.equal(qdq_q4_0(qdq_q4_0(w)), qdq_q4_0(w)))

    err4 = (qdq_q4_0(w[:300]) - w[:300]).abs().mean().item()
    err8 = (qdq_q8_0(w[:300]) - w[:300]).abs().mean().item()
    check("Q8_0 error far below Q4_0 error", err8 < err4 / 8, f"{err8:.2e} vs {err4:.2e}")

    print()
    print("=" * 70)
    print("QUANTIZATION-AWARE TRAINING MACHINERY")
    print("=" * 70)

    # Straight-through estimator: forward sees quantized weights, gradient
    # reaches the latent weight unchanged.
    lat = (torch.randn(4, 64) * 0.02).requires_grad_()
    out = FakeQuant("q4_0")(lat)
    check("fake-quant forward equals the exact quantizer",
          torch.equal(out.detach(), qdq_q4_0(lat.detach())))
    out.sum().backward()
    check("straight-through gradient is identity",
          torch.equal(lat.grad, torch.ones_like(lat)))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(64, 64)
            self.o_proj = torch.nn.Linear(64, 64)

    class Tiny(torch.nn.Module):
        def __init__(self, n=3):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(100, 64)
            self.layers = torch.nn.ModuleList(Block() for _ in range(n))
            self.lm_head = torch.nn.Linear(64, 100, bias=False)

    m = Tiny()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    counts = prepare_qat(m, "mixed")
    check("mixed spec: first and last blocks at Q8_0, middle at Q4_0",
          counts == {"q8_0": 4, "q4_0": 2}, str(counts))
    check("output head is never fake-quantized",
          not torch.nn.utils.parametrize.is_parametrized(m.lm_head, "weight"))

    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(2, 64)
    loss = sum(layer.o_proj(layer.q_proj(x)).pow(2).mean() for layer in m.layers)
    loss.backward()
    opt.step()

    n = strip_qat(m)
    check("strip removes every fake quantizer", n == 6, f"{n} stripped")
    keys_ok = set(m.state_dict()) == set(before)
    check("stripped model has the original state-dict keys (loadable)", keys_ok)
    moved = not torch.equal(m.layers[1].q_proj.weight, before["layers.1.q_proj.weight"])
    check("training updated the latent weights", moved)
    check("stripped weights are latent, not pre-rounded",
          not torch.equal(m.layers[1].q_proj.weight, qdq_q4_0(m.layers[1].q_proj.weight)))

    print()
    print("=" * 70)
    print("DEPLOYMENT SIMULATION AND EXPORT SPEC")
    print("=" * 70)

    m2 = Tiny()
    m2.lm_head.weight = m2.embed_tokens.weight          # tied, as in Qwen2.5-0.5B
    c = simulate_deployment(m2, "q4_0")
    check("q4_0 spec: linears Q4_0, tied embedding/output rounded once at Q8_0",
          c == {"q4_0": 6, "q8_0": 1}, str(c))
    check("simulated weights are on the Q4_0 grid",
          torch.equal(m2.layers[0].q_proj.weight, qdq_q4_0(m2.layers[0].q_proj.weight)))

    m3 = Tiny()
    snap = m3.layers[0].q_proj.weight.clone()
    check("spec 'none' leaves the weights untouched",
          simulate_deployment(m3, "none") == {} and torch.equal(m3.layers[0].q_proj.weight, snap))

    check("spec names map to layer formats",
          [linear_format("mixed", f"model.layers.{i}.mlp.up_proj", 24) for i in (0, 5, 23)]
          == ["q8_0", "q4_0", "q8_0"])

    base, flags = llama_quantize_args("mixed", 24)
    check("export flags pin embeddings/output and override blocks 0 and 23",
          base == "Q4_0" and flags.count("--tensor-type") == 2
          and r"blk\.0\.=q8_0" in flags and r"blk\.23\.=q8_0" in flags
          and "--output-tensor-type" in flags, " ".join(flags))

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S): " + ", ".join(FAILS))
        return 1
    print("All quantizer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
