import math
import torch
from torch import nn
from torch.nn import functional as F
from .lora import AttentionLoRALinear


class _FrozenDelta(nn.Module):
    def __init__(self, a, b, *, scale, gain):
        super().__init__()
        self.register_buffer("a", a.detach().contiguous())
        self.register_buffer("b", b.detach().contiguous())
        self.scale, self.gain = float(scale), float(gain)

    def forward(self, x):
        return F.linear(F.linear(x.to(self.a.dtype), self.a), self.b) * (self.scale * self.gain)


def _attach_converted(dit, converted, *, scale, gain=1.0, adapter_name="acceleration"):
    """Validate every tensor/target first; never merge or replace existing adapters."""
    if not converted or not math.isfinite(scale) or scale <= 0 or not math.isfinite(gain):
        raise ValueError("converted adapter and finite positive scale are required")
    from .lora import _validate_adapter_name

    _validate_adapter_name(adapter_name)
    suffixes = (".lora_A.weight", ".lora_B.weight")
    if any(not key.endswith(suffixes) for key in converted):
        raise ValueError("unexpected acceleration tensor key")
    prefixes = sorted(
        key.removesuffix(suffixes[0]) for key in converted if key.endswith(suffixes[0])
    )
    expected = {name + suffix for name in prefixes for suffix in suffixes}
    if expected != converted.keys():
        raise ValueError("unpaired acceleration A/B tensors")
    plans = []
    for name in prefixes:
        try:
            layer = dit.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"acceleration target missing: {name}") from error
        if not isinstance(layer, (nn.Linear, AttentionLoRALinear)):
            raise TypeError(f"acceleration target must be a direct linear: {name}")
        if isinstance(layer, AttentionLoRALinear) and adapter_name in layer.adapters:
            raise ValueError(f"duplicate acceleration adapter at {name}")
        a, b = converted[name + suffixes[0]], converted[name + suffixes[1]]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] < 1
            or a.shape[1] != layer.in_features
            or b.shape != (layer.out_features, a.shape[0])
        ):
            raise ValueError(f"invalid acceleration dimensions at {name}")
        if (
            not a.is_floating_point()
            or not b.is_floating_point()
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise ValueError(f"invalid acceleration values at {name}")
        plans.append((name, layer, a, b))
    adapters = []
    for name, layer, a, b in plans:
        if not isinstance(layer, AttentionLoRALinear):
            prefix, _, attr = name.rpartition(".")
            parent = dit.get_submodule(prefix) if prefix else dit
            layer = AttentionLoRALinear(layer)
            setattr(parent, attr, layer)
        adapter = _FrozenDelta(
            a.to(layer.weight.device, layer.weight.dtype),
            b.to(layer.weight.device, layer.weight.dtype),
            scale=scale,
            gain=gain,
        )
        layer.adapters[adapter_name] = adapter
        adapters.append(adapter)
    return adapters


def interleave_qkv_rows(tensor, head_dim=128):

    rows, rank = tensor.shape
    assert rows % (3 * head_dim) == 0
    return (
        tensor.reshape(3, rows // (3 * head_dim), head_dim, rank)
        .permute(1, 0, 2, 3)
        .reshape(rows, rank)
        .contiguous()
    )


def inspect_larry(path):
    """Check the acceleration checkpoint header without loading weight tensors."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        prefixes = {
            key.removesuffix(".lora_A.weight") for key in keys if key.endswith(".lora_A.weight")
        }
        expected = {
            name + suffix for name in prefixes for suffix in (".lora_A.weight", ".lora_B.weight")
        }
        if len(keys) != 518 or keys != expected:
            raise ValueError("Expected Larry v4 step600 EMA with 259 paired LoRA layers")
        for name in prefixes:
            a = handle.get_slice(name + ".lora_A.weight").get_shape()
            b = handle.get_slice(name + ".lora_B.weight").get_shape()
            rank = 16 if ".adaln_proj.linear" in name else 64
            if len(a) != 2 or len(b) != 2 or a[0] != rank or b[1] != rank:
                raise ValueError(f"Unexpected Larry tensor layout: {name}")
    return {"layers": len(prefixes), "tensors": len(keys)}


def attach_larry(model, path, gain=1.0):
    """Load Larry v4 with its native per-layer scale and H3 QKV ordering."""
    from safetensors.torch import load_file

    inspect_larry(path)
    state = load_file(str(path), device="cpu")
    if len(state) != 518:
        raise ValueError("Expected Larry v4 step600 EMA (518 tensors)")
    converted = {}
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            expected_rank = 16 if ".adaln_proj.linear." in key else 64
            if value.ndim != 2 or value.shape[0] != expected_rank:
                raise ValueError(f"Unexpected Larry rank: {key}")
        if key.endswith(".attn.qkv_proj.lora_B.weight"):
            value = interleave_qkv_rows(value)
        converted[key] = value
    return _attach_converted(model, converted, scale=1.0, gain=gain)
