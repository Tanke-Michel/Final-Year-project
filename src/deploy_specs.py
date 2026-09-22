"""
Deployment specs — the single definition of how each arm is quantized.

Pure Python, no PyTorch, so the orchestrator and the GGUF exporter can use it
on machines without a GPU stack. src/quant_sim.py (training and evaluation)
and src/quantize.py (llama-quantize) both read from here, which is what keeps
the three in agreement.

  none   no quantization                                  upper bound
  q4_0   linear layers Q4_0; embeddings and output Q8_0   main target
  q8_0   everything Q8_0
  mixed  as q4_0, but the first and last transformer blocks at Q8_0
"""

from __future__ import annotations

import re

SPECS = ("none", "q4_0", "q8_0", "mixed")
EMBEDDING_FORMAT = "q8_0"        # embeddings and output head, in every quantized spec

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def deploy_spec_for_arm(arm: str) -> str:
    """The spec each training arm is evaluated and shipped in."""
    return {"qat8": "q8_0", "qat_mixed": "mixed"}.get(arm, "q4_0")


def layer_index(module_name: str) -> int | None:
    m = _LAYER_RE.search(module_name)
    return int(m.group(1)) if m else None


def protected_layers(n_layers: int) -> set[int]:
    """Blocks held at Q8_0 in the mixed spec: the first and the last, which
    are consistently the most sensitive to quantization."""
    return {0, n_layers - 1} if n_layers > 1 else {0}


def linear_format(spec: str, module_name: str, n_layers: int) -> str | None:
    """Format of a transformer linear layer under a spec (None = unquantized)."""
    if spec == "none":
        return None
    if spec in ("q4_0", "q8_0"):
        return spec
    if spec == "mixed":
        i = layer_index(module_name)
        return "q8_0" if i is not None and i in protected_layers(n_layers) else "q4_0"
    raise ValueError(f"unknown spec {spec!r}; expected one of {SPECS}")


def llama_quantize_args(spec: str, n_layers: int) -> tuple[str, list[str]]:
    """(base type, extra flags) for llama-quantize, making every choice
    explicit rather than relying on llama.cpp's defaults, which have changed
    between versions. --tensor-type takes a regex over GGUF tensor names,
    where transformer block i is 'blk.i.'."""
    if spec == "none":
        raise ValueError("spec 'none' is not exported")
    base = "Q8_0" if spec == "q8_0" else "Q4_0"
    flags = ["--token-embedding-type", EMBEDDING_FORMAT,
             "--output-tensor-type", EMBEDDING_FORMAT]
    if spec == "mixed":
        for i in sorted(protected_layers(n_layers)):
            flags += ["--tensor-type", rf"blk\.{i}\.=q8_0"]
    return base, flags
