"""
Exact simulation of llama.cpp's Q4_0 and Q8_0 quantization, in PyTorch.

ONE QUANTIZER, THREE USES
  1. Quantization-aware training   the forward pass sees weights rounded
                                   exactly as the phone will round them
  2. Evaluation                    answers are scored from the weights the
                                   phone will actually hold, not from fp16
  3. Deployment                    src/quantize.py passes the same spec to
                                   llama-quantize and verifies the result

Using one implementation for all three means training, evaluation and the
deployed model see bit-identical weights. The alternative - training against
one library's quantizer, evaluating unquantized, deploying another - measures
something other than what the study claims.

The arithmetic follows ggml's reference quantizers (quantize_row_q4_0_ref,
quantize_row_q8_0_ref and the matching dequantizers). src/test_quant_sim.py
checks the vectorised code against a direct scalar translation of the C.

DEPLOYMENT SPECS
  none   no quantization                        upper bound
  q4_0   linear layers Q4_0, embeddings Q8_0    main target
  q8_0   everything Q8_0
  mixed  Q4_0, except the first and last transformer blocks at Q8_0

Embeddings and the output head are held at Q8_0 in every quantized spec, and
src/quantize.py sets them explicitly with --token-embedding-type and
--output-tensor-type rather than relying on llama.cpp's defaults, which have
changed between versions.

KNOWN SIMPLIFICATION. On device, llama.cpp also rounds activations to 8 bits
in blocks of 32 inside each matrix multiply. That is not simulated here. Its
error is small beside the 4-bit weight error; say so in the methods.
"""

from __future__ import annotations

import torch
from torch import nn

QK = 32                          # block size shared by Q4_0 and Q8_0


# ------------------------------------------------------------------ quantizers

def _as_blocks(w: torch.Tensor) -> torch.Tensor:
    """Rows of 32 consecutive values along the input dimension. ggml stores a
    weight matrix with the input features contiguous, and quantizes in blocks
    along that axis — which for a PyTorch [out, in] weight is the last axis."""
    if w.shape[-1] % QK:
        raise ValueError(f"last dimension {w.shape[-1]} is not a multiple of {QK}")
    return w.detach().float().reshape(-1, QK)


@torch.no_grad()
def qdq_q4_0(w: torch.Tensor) -> torch.Tensor:
    """Quantize to Q4_0 and dequantize again.

    Q4_0 is not the textbook symmetric int4. The element with the largest
    magnitude keeps its sign and is mapped to -8, so the block scale is
    d = max / -8 and one side of the range is exactly representable. That is
    why a generic symmetric-int4 fake quantizer does not match what the phone
    runs.
    """
    x = _as_blocks(w)
    idx = x.abs().argmax(dim=1, keepdim=True)       # first maximum wins, as in C
    signed_max = x.gather(1, idx)
    d = signed_max / -8.0
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    # C: MIN(15, (int8_t)(x*id + 8.5f)). The value is always >= 0.5, so the
    # int8 cast's truncation is a floor here.
    q = torch.clamp(torch.trunc(x * inv + 8.5), 0, 15)
    d16 = d.half().float()                          # the scale is stored as fp16
    return ((q - 8.0) * d16).reshape(w.shape).to(w.dtype)


@torch.no_grad()
def qdq_q8_0(w: torch.Tensor) -> torch.Tensor:
    """Quantize to Q8_0 and dequantize again: d = amax / 127, round half away
    from zero, scale stored as fp16."""
    x = _as_blocks(w)
    amax = x.abs().amax(dim=1, keepdim=True)
    d = amax / 127.0
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    v = x * inv
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)  # C roundf: half away from zero
    d16 = d.half().float()
    return (q * d16).reshape(w.shape).to(w.dtype)


QDQ = {"q4_0": qdq_q4_0, "q8_0": qdq_q8_0}


# ------------------------------------------------------------ layer selection

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from deploy_specs import (SPECS, EMBEDDING_FORMAT, deploy_spec_for_arm,  # noqa: E402,F401
                          layer_index, linear_format, protected_layers)


def num_layers(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and cfg is not None and hasattr(cfg, "text_config"):
        n = getattr(cfg.text_config, "num_hidden_layers", None)
    if n is None:
        idx = [layer_index(name) for name, _ in model.named_modules()]
        idx = [i for i in idx if i is not None]
        n = max(idx) + 1 if idx else 0
    return int(n)


def _is_output_head(name: str) -> bool:
    return name.split(".")[-1] in ("lm_head", "output", "embed_out")


def quantizable_linears(model: nn.Module):
    """Transformer linear layers that llama.cpp quantizes at the main type:
    every nn.Linear except the output head, whose input width is a multiple of
    the block size. Yields (name, module)."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not _is_output_head(name) \
                and mod.in_features % QK == 0:
            yield name, mod


# ---------------------------------------------------- quantization-aware training

class FakeQuant(nn.Module):
    """Weight parametrization for quantization-aware training.

    Forward returns the exactly-quantized weight; backward passes the gradient
    straight through to the latent full-precision weight (the straight-through
    estimator). The optimizer therefore updates full-precision weights that
    learn to survive rounding, which is the whole idea of QAT.
    """

    def __init__(self, fmt: str):
        super().__init__()
        if fmt not in QDQ:
            raise ValueError(f"unknown format {fmt!r}")
        self.fmt = fmt

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w + (QDQ[self.fmt](w) - w).detach()


def prepare_qat(model: nn.Module, spec: str) -> dict[str, int]:
    """Attach fake quantization to every quantizable linear layer. Embeddings
    and the output head are left in full precision during training; they are
    deployed at Q8_0, which is near-lossless. Returns a count per format."""
    from torch.nn.utils import parametrize

    n_layers = num_layers(model)
    counts: dict[str, int] = {}
    for name, mod in list(quantizable_linears(model)):
        fmt = linear_format(spec, name, n_layers)
        if fmt is None:
            continue
        parametrize.register_parametrization(mod, "weight", FakeQuant(fmt))
        counts[fmt] = counts.get(fmt, 0) + 1
    return counts


def strip_qat(model: nn.Module) -> int:
    """Remove the fake-quant parametrizations, keeping the trained latent
    weights. The saved checkpoint is then an ordinary model: llama.cpp converts
    it, and quantizing it with the same spec reproduces exactly the weights the
    training forward pass used. Returns the number of layers stripped."""
    from torch.nn.utils import parametrize

    n = 0
    for _, mod in list(model.named_modules()):
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
            n += 1
    return n


# --------------------------------------------------------- deployment simulation

@torch.no_grad()
def simulate_deployment(model: nn.Module, spec: str) -> dict[str, int]:
    """Round a model's weights in place exactly as llama.cpp will under `spec`,
    so that evaluation scores the model the phone actually runs.

    Linear layers take the spec's format; embeddings and the output head take
    Q8_0. Tied weights are rounded once. Returns a count of tensors per format.
    """
    if spec == "none":
        return {}
    n_layers = num_layers(model)
    done: set[int] = set()
    counts: dict[str, int] = {}

    def apply(t: torch.Tensor, fmt: str) -> None:
        ptr = t.data_ptr()
        if ptr in done or t.shape[-1] % QK:
            return
        t.copy_(QDQ[fmt](t))
        done.add(ptr)
        counts[fmt] = counts.get(fmt, 0) + 1

    for name, mod in quantizable_linears(model):
        fmt = linear_format(spec, name, n_layers)
        if fmt:
            apply(mod.weight, fmt)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) or (isinstance(mod, nn.Linear) and _is_output_head(name)):
            apply(mod.weight, EMBEDDING_FORMAT)

    return counts
