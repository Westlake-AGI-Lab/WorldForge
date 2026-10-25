from __future__ import annotations

import torch

from .masking import VisibilityConfig, filter_reference_tokens


def image_only_after_warp_cutoff(cache):
    if cache.get("ref_qwen_mode") != "image_only":
        raise ValueError(
            "Warp cutoff requires image-only Qwen embeddings; video-conditioned Qwen would retain warp information"
        )
    context = cache["context"]
    if (
        context.get("ref_audio_anchor") is not None
        or context.get("keyframe_cond_anchor") is not None
    ):
        raise ValueError("Warp cutoff supports one Ref2VA image plus a silent warped video")
    info = cache["reference_mask"]
    anchor = context["ref_visual_anchor"]
    original = info["original_reference_rows"].to(device=anchor.device, dtype=torch.long)
    image_rows = info["image_rows"]
    if type(image_rows) is not int or image_rows <= 0 or original.shape != (anchor.shape[0],):
        raise ValueError("Invalid reference image row metadata")
    if not torch.equal(original[:image_rows], torch.arange(image_rows, device=original.device)):
        raise ValueError("All original image rows must precede warped-video rows")
    if bool((original[image_rows:] < image_rows).any()):
        raise ValueError("Unexpected image rows after warped-video rows")
    keep = original < image_rows
    filtered = filter_reference_tokens(
        context["packed"], anchor, keep.float(), VisibilityConfig(mode="hard")
    )
    result = dict(context, packed=filtered.packed, ref_visual_anchor=filtered.ref_visual_anchor)
    bias = context.get("key_log_bias")
    if bias is not None:

        remap = filtered.original_sequence_indices.to(bias.device)
        retained = remap >= 0
        new_bias = bias.new_zeros(remap.shape)
        new_bias[retained] = bias[remap[retained]]
        result["key_log_bias"] = new_bias
    audit = dict(
        image_reference_rows=image_rows,
        warp_reference_rows_before=anchor.shape[0] - image_rows,
        warp_reference_rows_after=0,
        seq_len_before=int(context["packed"]["seq_len"]),
        seq_len_after=int(filtered.packed["seq_len"]),
        retained_positions_preserved=True,
        first_image_anchor_unchanged=True,
        qwen_embeddings_unchanged=True,
        qwen_mode="image_only",
        target_audio_tokens_retained=True,
    )
    return result, audit
