from __future__ import annotations

import json
import math
import re
from typing import Iterable

import torch
from torch import nn

_FORMAT = "h3-trajectory-attention-lora-v1"

_TARGET = re.compile(r"^blocks\.(\d+)\.attn\.(qkv_proj|out_proj)$")


class _Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        qkv_mode: str,
        num_heads: int,
        head_dim: int,
        device,
        dtype,
    ):
        super().__init__()
        self.rank, self.alpha = rank, float(alpha)
        self.qkv_mode, self.num_heads, self.head_dim = qkv_mode, num_heads, head_dim
        self.gain = 1.0
        branches = 3 if qkv_mode == "separate" else 1
        branch_out = out_features // branches

        self.lora_A = nn.ModuleList(
            [
                nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
                for _ in range(branches)
            ]
        )
        self.lora_B = nn.ModuleList(
            [
                nn.Linear(rank, branch_out, bias=False, device=device, dtype=dtype)
                for _ in range(branches)
            ]
        )
        for a, b in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)

    @property
    def scale(self):
        return self.alpha / self.rank

    def forward(self, x, *, output_dtype=None):
        x = x.to(self.lora_A[0].weight.dtype)
        scale = self.scale * self.gain
        if self.qkv_mode == "separate" and output_dtype is not None:
            parts = []
            for a, b in zip(self.lora_A, self.lora_B):

                part = b(a(x)).mul_(scale).to(output_dtype)
                parts.append(part.reshape(*part.shape[:-1], self.num_heads, self.head_dim))
            return torch.stack(parts, dim=-2).flatten(-3)
        parts = [b(a(x)) for a, b in zip(self.lora_A, self.lora_B)]
        if self.qkv_mode == "separate":

            parts = [p.reshape(*p.shape[:-1], self.num_heads, self.head_dim) for p in parts]
            delta = torch.stack(parts, dim=-2).flatten(-3)
        else:
            delta = parts[0]

        delta.mul_(scale)
        return delta if output_dtype is None else delta.to(output_dtype)

    def import_pair(self, a, b):
        with torch.no_grad():
            if self.qkv_mode == "fused":
                self.lora_A[0].weight.copy_(a)
                self.lora_B[0].weight.copy_(b)
                return
            rows = torch.arange(b.shape[0], device=b.device).reshape(
                self.num_heads, 3, self.head_dim
            )
            for index, (la, lb) in enumerate(zip(self.lora_A, self.lora_B)):
                sl = slice(index * self.rank, (index + 1) * self.rank)
                la.weight.copy_(a[sl])
                lb.weight.copy_(b[rows[:, index].reshape(-1), sl])


class AttentionLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear):
        super().__init__()
        self.base_layer = base_layer.requires_grad_(False)
        self.adapters = nn.ModuleDict()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        result = self.base_layer(x)
        for adapter in self.adapters.values():

            if adapter.gain != 0:

                delta = (
                    adapter(x, output_dtype=result.dtype)
                    if isinstance(adapter, _Adapter)
                    else adapter(x).to(result.dtype)
                )
                result = result + delta
        return result


def _targets(model):
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
        raise ValueError("expected an unwrapped H3 DiT with a nonempty .blocks ModuleList")
    targets = []
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            raise ValueError(f"blocks.{index} has no attention")
        heads, dim = getattr(attn, "num_heads", None), getattr(attn, "head_dim", None)
        if not isinstance(heads, int) or not isinstance(dim, int) or heads < 1 or dim < 1:
            raise ValueError(f"blocks.{index}.attn requires valid num_heads and head_dim")
        for attr in ("qkv_proj", "out_proj"):
            layer = getattr(attn, attr, None)
            if not isinstance(layer, (nn.Linear, AttentionLoRALinear)):
                raise TypeError(
                    f"blocks.{index}.attn.{attr} must be a Linear, got {type(layer).__name__}"
                )
            if attr == "qkv_proj" and layer.out_features != heads * dim * 3:
                raise ValueError(
                    f"blocks.{index}.attn.qkv_proj shape does not match H3 interleaved QKV"
                )
            if attr == "out_proj" and layer.in_features != heads * dim:
                raise ValueError(
                    f"blocks.{index}.attn.out_proj shape does not match attention heads"
                )
            name = f"blocks.{index}.attn.{attr}"
            targets.append((name, attn, attr, layer, heads, dim))
    return targets


def _validate_adapter_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(
            "adapter_name must start with a letter and contain only letters, digits, underscore"
        )


def _block_indices(indices, num_blocks):
    if indices is None:
        return list(range(num_blocks))
    try:
        indices = list(indices)
    except TypeError as exc:
        raise ValueError("block_indices must be a nonempty iterable of block IDs") from exc
    if not indices or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < num_blocks
        for index in indices
    ):
        raise ValueError(f"block_indices must contain integer block IDs in [0, {num_blocks})")
    if len(set(indices)) != len(indices):
        raise ValueError("block_indices must not contain duplicates")
    return sorted(indices)


def _selected_targets(targets, indices):
    selected = set(indices)
    return [target for target in targets if int(_TARGET.fullmatch(target[0]).group(1)) in selected]


def inject_attention_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    adapter_name: str = "trajectory",
    qkv_mode: str = "fused",
    dtype: torch.dtype = torch.float32,
    block_indices: Iterable[int] | None = None,
) -> list[str]:
    """Freeze the model and add a named adapter to selected main-block attentions."""
    _validate_adapter_name(adapter_name)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if qkv_mode not in ("fused", "separate"):
        raise ValueError("qkv_mode must be 'fused' or 'separate'")
    if not dtype.is_floating_point:
        raise ValueError("adapter dtype must be floating point")
    targets = _targets(model)
    indices = _block_indices(block_indices, len(model.blocks))
    for name, _, _, layer, _, _ in targets:
        if isinstance(layer, AttentionLoRALinear) and adapter_name in layer.adapters:
            raise ValueError(f"adapter {adapter_name!r} already exists in {name}")
    model.requires_grad_(False)
    names = []
    for name, parent, attr, layer, heads, dim in _selected_targets(targets, indices):
        if not isinstance(layer, AttentionLoRALinear):
            layer = AttentionLoRALinear(layer)
            setattr(parent, attr, layer)
        adapter = _Adapter(
            layer.in_features,
            layer.out_features,
            rank=rank,
            alpha=alpha,
            qkv_mode=qkv_mode if attr == "qkv_proj" else "fused",
            num_heads=heads,
            head_dim=dim,
            device=layer.weight.device,
            dtype=dtype,
        )
        layer.adapters[adapter_name] = adapter.requires_grad_(False)
        names.append(name)
    return names


def _named_adapters(model, name):
    _validate_adapter_name(name)
    found = []
    targets = _targets(model)
    for target, _, _, layer, _, _ in targets:
        if isinstance(layer, AttentionLoRALinear) and name in layer.adapters:
            found.append((target, layer, layer.adapters[name]))
    if not found:
        raise ValueError(f"adapter {name!r} is not attached to any main-block attention")
    indices = {int(_TARGET.fullmatch(target).group(1)) for target, _, _ in found}
    if [target for target, _, _ in found] != [
        target[0] for target in _selected_targets(targets, indices)
    ]:
        raise ValueError(
            f"adapter {name!r} must cover both attention projections in each selected block"
        )
    return found


def set_adapter_gain(model: nn.Module, adapter_name: str, gain: float):
    if not math.isfinite(gain):
        raise ValueError("adapter gain must be finite")
    for _, _, adapter in _named_adapters(model, adapter_name):
        adapter.gain = float(gain)


def load_attention_lora(
    model: nn.Module,
    path,
    *,
    adapter_name: str = "trajectory",
    dtype: torch.dtype = torch.float32,
    expected_qkv_mode: str | None = None,
) -> dict:
    """Validate topology/tensors fully before attaching a native exported adapter."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        info = f.metadata() or {}
        if info.get("format") != _FORMAT:
            raise ValueError("not an H3 trajectory adapter with validated topology metadata")
        topology = json.loads(info["topology"])
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    targets = _targets(model)
    has_selection = "block_indices" in info
    if has_selection != ("num_main_blocks" in info):
        raise ValueError("checkpoint block selection requires num_main_blocks and block_indices")
    if has_selection:
        if info["num_main_blocks"] != str(len(model.blocks)):
            raise ValueError("checkpoint main-block count differs from this DiT")
        stored_indices = json.loads(info["block_indices"])
        if not isinstance(stored_indices, list):
            raise ValueError("checkpoint block_indices must be a list")
        indices = _block_indices(stored_indices, len(model.blocks))
    else:
        indices = _block_indices(None, len(model.blocks))
    targets = _selected_targets(targets, indices)
    expected_names = [item[0] for item in targets]
    if [entry.get("name") for entry in topology] != expected_names:
        raise ValueError("checkpoint attention layer topology differs from this DiT")
    if any(not _TARGET.fullmatch(entry["name"]) for entry in topology):
        raise ValueError("checkpoint contains forbidden LoRA targets")
    expected_keys = {
        entry["name"] + suffix
        for entry in topology
        for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha")
    }
    if tensors.keys() != expected_keys:
        raise ValueError("checkpoint has missing or unexpected tensors")
    ranks, alphas, qkv_modes = set(), set(), set()
    for entry, (name, _, attr, layer, heads, dim) in zip(topology, targets):
        if (entry["in_features"], entry["out_features"], entry["num_heads"], entry["head_dim"]) != (
            layer.in_features,
            layer.out_features,
            heads,
            dim,
        ):
            raise ValueError(f"checkpoint dimensions differ at {name}")
        rank, alpha, mode = entry["rank"], entry["alpha"], entry["qkv_mode"]
        if mode not in ("fused", "separate") or (attr == "out_proj" and mode != "fused"):
            raise ValueError(f"invalid adapter topology at {name}")
        if not isinstance(rank, int) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"invalid rank/alpha at {name}")
        ranks.add(rank)
        alphas.add(alpha)
        if attr == "qkv_proj":
            qkv_modes.add(mode)
        multiplier = 3 if mode == "separate" else 1
        a, b = tensors[name + ".lora_A.weight"], tensors[name + ".lora_B.weight"]
        if a.shape != (rank * multiplier, layer.in_features) or b.shape != (
            layer.out_features,
            rank * multiplier,
        ):
            raise ValueError(f"invalid LoRA tensor shape at {name}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"nonfinite LoRA tensor at {name}")
        exported_alpha = tensors[name + ".alpha"]
        if exported_alpha.numel() != 1 or not math.isclose(
            float(exported_alpha), alpha * multiplier, rel_tol=1e-6
        ):
            raise ValueError(f"invalid exported alpha at {name}")
        if mode == "separate":

            offdiag = b.clone().reshape(heads, 3, dim, rank * 3)
            for branch in range(3):
                offdiag[:, branch, :, branch * rank : (branch + 1) * rank] = 0
            if torch.count_nonzero(offdiag):
                raise ValueError(
                    f"separate-QKV checkpoint has nonzero cross-branch weights at {name}"
                )
    if len(ranks) != 1 or len(alphas) != 1 or len(qkv_modes) != 1:
        raise ValueError("checkpoint must use a uniform rank, alpha, and QKV mode")
    qkv_mode = next(iter(qkv_modes))
    if expected_qkv_mode is not None and qkv_mode != expected_qkv_mode:
        raise ValueError(f"expected {expected_qkv_mode!r} QKV, checkpoint has {qkv_mode!r}")
    inject_attention_lora(
        model,
        rank=next(iter(ranks)),
        alpha=next(iter(alphas)),
        adapter_name=adapter_name,
        qkv_mode=qkv_mode,
        dtype=dtype,
        block_indices=indices,
    )
    for name, _, adapter in _named_adapters(model, adapter_name):
        adapter.import_pair(tensors[name + ".lora_A.weight"], tensors[name + ".lora_B.weight"])
    return {
        **info,
        "topology": topology,
        "block_indices": indices,
        "num_main_blocks": len(model.blocks),
        "user_metadata": json.loads(info.get("user_metadata", "{}")),
    }
