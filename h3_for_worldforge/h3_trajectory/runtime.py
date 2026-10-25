from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_MODEL_ROOT = Path("weights/MiniMax-H3")

_COMPONENTS = {
    "dit": "transformer",
    "text_encoder": "text_encoder",
    "video_vae": "video_vae/source",
}


def _local_component_path(root: Path, component: str):
    folder = root / _COMPONENTS[component]
    if component in ("dit", "text_encoder"):
        index_path = folder / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        mapping = json.loads(index_path.read_text())["weight_map"]
        if not mapping:
            raise ValueError(f"empty shard index: {index_path}")
        names = sorted(set(mapping.values()))
        if any(Path(name).name != name for name in names):
            raise ValueError(f"nonlocal shard name in {index_path}")
        paths = [folder / name for name in names]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        return [str(path) for path in paths]
    path = folder / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def load_pipe(
    variant: str,
    components=("dit",),
    device="cuda",
    model_root=DEFAULT_MODEL_ROOT,
    *,
    torch_dtype=torch.bfloat16,
    load_processor: bool | None = None,
):
    """Load selected native BF16 components directly, without VRAM wrappers."""
    if variant != "Ref2VA":
        raise ValueError("This package uses MiniMax-H3 Ref2VA")
    if isinstance(components, str) or not components or len(set(components)) != len(components):
        raise ValueError("components must be a nonempty sequence with no duplicates")
    if any(component not in _COMPONENTS for component in components):
        raise ValueError(f"unknown component; supported: {sorted(_COMPONENTS)}")
    root = Path(model_root) / variant
    paths = [_local_component_path(root, component) for component in components]
    if load_processor is None:
        load_processor = "text_encoder" in components
    processor_path = root / "processor"
    if load_processor and not processor_path.is_dir():
        raise FileNotFoundError(processor_path)
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=str(device),
        model_configs=[ModelConfig(path=path, skip_download=True) for path in paths],
        processor_config=None,
        redirect_common_files=False,
    )
    if load_processor:
        from transformers import AutoProcessor

        pipe.processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True)
        pipe.tokenizer = pipe.processor.tokenizer
    for component in components:
        module = getattr(pipe, component, None)
        if module is None:
            raise RuntimeError(f"checkpoint loader did not populate {component}")
        if getattr(module, "vram_management_enabled", False):
            raise RuntimeError("runtime requires direct modules, not VRAM management wrappers")
        module.requires_grad_(False).eval()
    pipe.trajectory_variant = variant
    return pipe


def move_context(value, device, *, dtype=None):
    """Move nested cached tensors while retaining integer tags and float64 RoPE."""
    if torch.is_tensor(value):
        cast = (
            dtype
            if dtype is not None and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else value.dtype
        )
        return value.to(device=device, dtype=cast)
    if isinstance(value, dict):
        return {key: move_context(item, device, dtype=dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_context(item, device, dtype=dtype) for item in value)
    if isinstance(value, list):
        return [move_context(item, device, dtype=dtype) for item in value]
    return value


def _attention_modules(dit):
    if not isinstance(getattr(dit, "blocks", None), nn.ModuleList):
        raise ValueError("expected an unwrapped H3 DiT .blocks ModuleList")
    return [block.attn for block in dit.blocks]


def _biased_attention_forward(self, x, *, rope_freqs, cu_seqlens, max_seqlen=None):
    bias = self.key_log_bias
    if bias is None:
        return self._trajectory_original_attention_forward(
            x, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
    if bias.ndim != 1 or bias.shape[0] != x.shape[0]:
        raise ValueError(f"key_log_bias must be [{x.shape[0]}], got {list(bias.shape)}")
    from diffsynth.models.minimax_h3_dit import _apply_rope

    total = x.shape[0]
    qkv = self.qkv_proj(x).view(total, self.num_heads, 3, self.head_dim)
    q, k, v = self.q_norm(qkv[:, :, 0]), self.k_norm(qkv[:, :, 1]), qkv[:, :, 2]
    if rope_freqs is not None:
        q, k = _apply_rope(q, rope_freqs), _apply_rope(k, rope_freqs)
    bounds = cu_seqlens.tolist()
    if (
        not bounds
        or bounds[0] != 0
        or bounds[-1] != total
        or any(b > e for b, e in zip(bounds, bounds[1:]))
    ):
        raise ValueError("packed cu_seqlens must monotonically partition the full token sequence")
    out = torch.empty_like(q)
    bias = bias.to(device=q.device, dtype=q.dtype)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if start == stop:
            continue
        segment_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        segment_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        segment_v = v[start:stop].transpose(0, 1).unsqueeze(0)

        key_bias = bias[start:stop].view(1, 1, 1, stop - start)

        if q.is_cuda:
            from torch.nn.attention import sdpa_kernel, SDPBackend

            backend = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        else:
            backend = nullcontext()
        with backend:
            result = F.scaled_dot_product_attention(
                segment_q,
                segment_k,
                segment_v,
                attn_mask=key_bias,
                dropout_p=0.0,
                scale=self.softmax_scale,
                is_causal=False,
            )
        out[start:stop] = result.squeeze(0).transpose(0, 1)
    return self.out_proj(out.reshape(total, self.num_heads * self.head_dim))


def install_ref_attention_bias(dit):
    """Install once on main DiT self attention, excluding token_refiner entirely."""
    modules = _attention_modules(dit)
    for module in modules:
        for attr in (
            "qkv_proj",
            "out_proj",
            "q_norm",
            "k_norm",
            "num_heads",
            "head_dim",
            "softmax_scale",
        ):
            if not hasattr(module, attr):
                raise TypeError(f"expected native H3 attention with {attr}")
    for module in modules:
        if not hasattr(module, "_trajectory_original_attention_forward"):
            module._trajectory_original_attention_forward = module.forward
            module.key_log_bias = None
            module.forward = MethodType(_biased_attention_forward, module)
    return dit


@contextmanager
def reference_attention_bias(dit, bias):
    """Temporarily apply the reference key prior, preserving nested contexts."""
    if bias is not None:
        if not torch.is_tensor(bias) or bias.ndim != 1 or not bias.is_floating_point():
            raise ValueError("key_log_bias must be a one-dimensional floating tensor or None")
        if torch.isnan(bias).any() or torch.isposinf(bias).any() or (bias > 0).any():
            raise ValueError(
                "key_log_bias must be log visibility <= 0 (negative infinity is allowed)"
            )
    install_ref_attention_bias(dit)
    modules = _attention_modules(dit)
    previous = [module.key_log_bias for module in modules]
    try:
        for module in modules:
            module.key_log_bias = bias
        yield
    finally:
        for module, value in zip(modules, previous):
            module.key_log_bias = value


_CONDITION_KEYS = {
    "packed",
    "prompt_embeds",
    "keyframe_cond_anchor",
    "ref_visual_anchor",
    "ref_audio_anchor",
    "input_latents_video",
    "denoise_mask_video",
    "input_latents_audio",
    "denoise_mask_audio",
    "imgvid_cond_noise_aug",
    "audio_cond_noise_aug",
}


def velocity(pipe, context, video, audio, sigma_video, sigma_audio):
    """Return wrapper velocities u=(z_sigma-z0)/sigma, NOT raw DiT velocities."""
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3

    if getattr(pipe, "dit", None) is None:
        raise ValueError("velocity requires a loaded DiT")
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError("video latent must have shape [1,24,T,H,W]")
    if audio.ndim != 3 or audio.shape[0] != 2 or audio.shape[1] != 32:
        raise ValueError("native audio latent must have shape [2,32,T_audio]")
    if video.device != audio.device or video.dtype != audio.dtype:
        raise ValueError("video and audio latent device/dtype must match")
    for key in ("packed", "prompt_embeds"):
        if key not in context:
            raise KeyError(f"context requires {key}")
    timesteps = []
    for sigma in (sigma_video, sigma_audio):
        value = torch.as_tensor(sigma, dtype=torch.float32, device=video.device)
        if (
            value.numel() != 1
            or not torch.isfinite(value).all()
            or not ((value >= 0) & (value <= 1)).all()
        ):
            raise ValueError("sigma must be scalar, finite, and in [0,1]")
        timesteps.append(value.reshape(1) * 1000)
    inputs = {key: value for key, value in context.items() if key in _CONDITION_KEYS}
    return model_fn_minimax_h3(
        dit=pipe.dit,
        video_latents=video,
        audio_latents=audio,
        timestep_video=timesteps[0],
        timestep_audio=timesteps[1],
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **inputs,
    )
