from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass(frozen=True)
class VisibilityConfig:
    mode: Literal["hard", "soft"]
    temporal_aggregation: Literal["mean", "min"] = "mean"
    quantization: int | None = 8
    minimum_weight: float = 0.0
    minimum_tokens_per_frame: int = 16

    def __post_init__(self):
        if self.mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        if self.temporal_aggregation not in ("mean", "min"):
            raise ValueError("temporal_aggregation must be 'mean' or 'min'")
        if self.quantization is not None and (
            self.quantization < 1 or int(self.quantization) != self.quantization
        ):
            raise ValueError("quantization must be a positive integer or None")
        if not 0 <= self.minimum_weight <= 1:
            raise ValueError("minimum_weight must be in [0, 1]")
        if self.minimum_tokens_per_frame < 0:
            raise ValueError("minimum_tokens_per_frame must be nonnegative")


def h3_frame_groups(num_frames: int, *, process_image: bool = False) -> tuple[tuple[int, ...], ...]:
    """Frame groups matching H3 VAE clip resets, padding and tail removal."""
    if process_image:
        if num_frames != 1:
            raise ValueError("process_image requires exactly one frame")
        return ((0,),)
    if num_frames < 5 or (num_frames - 5) % 17:
        raise ValueError(f"H3 video length must be 17*n+5, got {num_frames}")
    groups = []
    for start in range(0, num_frames, 17):
        groups.append((start,))
        for offset in (1, 5, 9, 13):
            groups.append(tuple(min(start + offset + j, num_frames - 1) for j in range(4)))
    return tuple(groups[:-3])


def visibility_to_token_weights(
    visibility: torch.Tensor | np.ndarray,
    config: VisibilityConfig,
    *,
    process_image: bool = False,
    spatial_factor: int = 32,
) -> torch.Tensor:
    """Convert valid-pixel confidence [F,H,W] to weights [Lt,H/s,W/s]."""
    vis = torch.as_tensor(visibility).float()
    if vis.ndim != 3:
        raise ValueError("visibility must have shape [frames, height, width]")
    frames, height, width = vis.shape
    if spatial_factor not in (16, 32):
        raise ValueError("spatial_factor must be 16 (VAE cells) or 32 (DiT tokens)")
    if height % 32 or width % 32 or min(height, width) < 32:
        raise ValueError("H3 reference height and width must be positive multiples of 32")
    if not bool(torch.isfinite(vis).all()) or bool(((vis < 0) | (vis > 1)).any()):
        raise ValueError("visibility must be finite confidence values in [0, 1]")
    pooled = []
    for group in h3_frame_groups(frames, process_image=process_image):
        frame_vis = vis[list(group)]
        temporal = frame_vis.mean(0) if config.temporal_aggregation == "mean" else frame_vis.amin(0)
        weight = temporal.reshape(
            height // spatial_factor, spatial_factor, width // spatial_factor, spatial_factor
        ).mean(dim=(1, 3))
        if config.quantization is not None:
            weight = torch.round(weight * config.quantization) / config.quantization
        weight = torch.where(weight >= config.minimum_weight, weight, 0)
        if int(torch.count_nonzero(weight)) < config.minimum_tokens_per_frame:
            weight = torch.zeros_like(weight)
        pooled.append(weight)
    return torch.stack(pooled)


def fill_reference_holes(rgb: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Nearest visible RGB fill, then two 5x5 smoothing passes inside holes."""
    rgb = np.asarray(rgb)
    visibility = np.asarray(visibility)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be a uint8 [height,width,3] array")
    if visibility.shape != rgb.shape[:2]:
        raise ValueError("visibility shape does not match RGB")
    if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
        raise ValueError("visibility must be finite confidence values in [0, 1]")
    valid = visibility > 0
    if valid.all():
        return rgb.copy()
    if not valid.any():
        return np.zeros_like(rgb)
    from scipy.ndimage import distance_transform_edt, uniform_filter

    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    out = np.where(valid[..., None], rgb, rgb[tuple(nearest)])
    for _ in range(2):
        smooth = uniform_filter(out.astype(np.float32), size=(5, 5, 1), mode="mirror")
        out = np.where(valid[..., None], rgb, np.rint(smooth).astype(np.uint8))
    return out.astype(np.uint8)


@dataclass
class FilteredReference:
    packed: dict
    ref_visual_anchor: torch.Tensor

    reference_row_indices: torch.Tensor

    original_sequence_indices: torch.Tensor
    reference_positions: torch.Tensor
    target_video_positions: torch.Tensor

    key_log_bias: torch.Tensor


def filter_reference_tokens(
    packed: dict,
    ref_visual_anchor: torch.Tensor,
    token_weights: torch.Tensor,
    config: VisibilityConfig,
    *,
    sequence_alignment: int = 64,
) -> FilteredReference:
    """Filter reference visual tokens while preserving original RoPE values."""
    if sequence_alignment < 1:
        raise ValueError("sequence_alignment must be positive")
    if ref_visual_anchor.ndim != 2:
        raise ValueError("ref_visual_anchor must be [reference_rows, channels]")
    device = packed["img_pos"].device
    weights = torch.as_tensor(token_weights, device=device, dtype=torch.float32).reshape(-1)
    n_ref = ref_visual_anchor.shape[0]
    if weights.numel() != n_ref:
        raise ValueError(f"weights have {weights.numel()} rows but anchor has {n_ref}")
    if not bool(torch.isfinite(weights).all()) or bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("token weights must be finite and in [0, 1]")
    old_seq_len = int(packed["seq_len"])
    cu = packed["cu_seqlens"]
    if cu.numel() != 3 or int(cu[0]) != 0 or int(cu[-1]) != old_seq_len:
        raise ValueError("expected one H3 sample plus its padding sequence")
    used = int(cu[1])
    image_pos = packed["img_pos"].long()
    if n_ref > image_pos.numel():
        raise ValueError("reference anchor is longer than the visual sequence")
    if packed["img_position_ids"].shape != (1, old_seq_len, 3):
        raise ValueError("invalid H3 positional metadata shape")

    modalities = torch.cat([packed[name].long() for name in ("text_pos", "img_pos", "audio_pos")])
    if modalities.numel() != used or not torch.equal(
        torch.sort(modalities).values, torch.arange(used, device=device)
    ):
        raise ValueError("text, image, and audio positions must partition the real sequence")
    reference_pos = image_pos[:n_ref]
    keep_ref = (weights > 0) & (weights >= config.minimum_weight)
    reference_indices = torch.nonzero(keep_ref, as_tuple=True)[0]
    keep_sequence = torch.ones(used, dtype=torch.bool, device=device)
    keep_sequence[reference_pos[~keep_ref]] = False
    old_indices = torch.nonzero(keep_sequence, as_tuple=True)[0]
    new_used = old_indices.numel()
    new_seq_len = ((new_used + sequence_alignment - 1) // sequence_alignment) * sequence_alignment
    old_to_new = torch.full((old_seq_len,), -1, dtype=torch.long, device=device)
    old_to_new[old_indices] = torch.arange(new_used, device=device)
    new_packed = dict(packed)
    for name in ("img_pos", "audio_pos", "text_pos"):
        remapped = old_to_new[packed[name].long()]
        new_packed[name] = remapped[remapped >= 0].to(packed[name].dtype)
    positions = packed["img_position_ids"]
    new_positions = positions.new_zeros((1, new_seq_len, 3))
    new_positions[:, :new_used] = positions.index_select(1, old_indices.to(positions.device))
    new_packed["img_position_ids"] = new_positions
    tags = packed["token_tags"]
    new_tags = tags.new_full((new_seq_len,), -1)
    new_tags[:new_used] = tags.index_select(0, old_indices.to(tags.device))
    new_packed["token_tags"] = new_tags
    new_packed["seq_len"] = new_seq_len
    new_packed["cu_seqlens"] = cu.new_tensor([0, new_used, new_seq_len])
    original_indices = old_indices.new_full((new_seq_len,), -1)
    original_indices[:new_used] = old_indices
    new_ref_positions = old_to_new[reference_pos[keep_ref]]
    key_log_bias = weights.new_zeros((new_seq_len,))
    if config.mode == "soft":
        key_log_bias[new_ref_positions] = torch.log(weights[keep_ref])
    return FilteredReference(
        packed=new_packed,
        ref_visual_anchor=ref_visual_anchor.index_select(
            0, reference_indices.to(ref_visual_anchor.device)
        ),
        reference_row_indices=reference_indices,
        original_sequence_indices=original_indices,
        reference_positions=new_ref_positions,
        target_video_positions=old_to_new[image_pos[n_ref:]],
        key_log_bias=key_log_bias,
    )
