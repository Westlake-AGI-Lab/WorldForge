"""Encode an input image, warped RGB frames and visibility masks for H3."""

import gc
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from .geometry import SpatialControlConfig, augmented_reference, reference_shape, spatial_gate
from .masking import (
    VisibilityConfig,
    fill_reference_holes,
    filter_reference_tokens,
    visibility_to_token_weights,
)
from .runtime import load_pipe, move_context


def image_files(directory):
    files = [p for p in Path(directory).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return sorted(
        files, key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)]
    )


def case_preprocessing(config, overrides):
    overrides = overrides or {}
    if not isinstance(overrides, dict) or set(overrides) - {"frames", "image_framing"}:
        raise ValueError("Case preprocessing supports frames and image_framing only")
    return {**config, **overrides}


def load_cases(path):
    path = Path(path).resolve()
    records = json.loads(path.read_text())
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError("Input manifest must contain one case or a nonempty list")
    seen = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or not {"id", "image", "warp", "mask", "prompt"} <= record.keys()
        ):
            raise ValueError("Each case needs id, image, warp, mask and prompt")
        name = record["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name in seen
        ):
            raise ValueError("Case IDs must be unique simple filenames")
        seen.add(name)
        for key in ("image", "warp", "mask"):
            record[key] = str((path.parent / record[key]).resolve())
        record.setdefault("seed", 42)
        if type(record["seed"]) is not int or record["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
    return records


def read_inputs(record, config):
    images, masks = image_files(record["warp"]), image_files(record["mask"])
    if not images or len(images) != len(masks):
        raise ValueError("Warp and mask directories need the same nonzero frame count")
    frames = config["frames"]
    if type(frames) is not int or frames < 5 or (frames - 5) % 17:
        raise ValueError("H3 requires 17*n+5 frames, for example 56 or 73")
    width, height = config["width"], config["height"]
    if any(type(x) is not int or x < 32 or x % 32 for x in (width, height)):
        raise ValueError("Width and height must be positive multiples of 32")
    policy = config["frame_policy"]
    if policy == "pad":
        if len(images) > frames:
            raise ValueError(
                "Input exceeds target length; increase frames or explicitly select resample"
            )
        indices = list(range(len(images))) + [len(images) - 1] * (frames - len(images))
        export_frames = len(images)
    elif policy == "resample":
        indices = np.rint(np.linspace(0, len(images) - 1, frames)).astype(int).tolist()
        export_frames = frames
    else:
        raise ValueError("frame_policy must be pad or resample")
    with Image.open(record["image"]) as source:
        image = source.convert("RGB")
    framing = config.get("image_framing")
    if framing:
        aspect = framing["aspect_width"] / framing["aspect_height"]
        iw, ih = image.size
        cw, ch = (round(ih * aspect), ih) if iw / ih > aspect else (iw, round(iw / aspect))
        left, top = (iw - cw) // 2, (ih - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        cw, ch = (round(height * aspect), height) if width / height > aspect else (width, round(width / aspect))
        left, top = (width - cw) // 2, (height - ch) // 2
        content = np.asarray(image.resize((cw, ch), Image.Resampling.LANCZOS))
        target_image = Image.fromarray(np.pad(content, ((top, height-ch-top), (left, width-cw-left), (0, 0)), mode="edge"))
    else:
        target_image = image.resize((width, height), Image.Resampling.LANCZOS)
    with Image.open(images[0]) as source:
        source_size = source.size
    reference_config = config["reference"]
    image_size = reference_shape(*image.size, short_edge=reference_config["image_short_edge"])
    text_image_size = reference_shape(*image.size, short_edge=reference_config.get("text_image_short_edge", reference_config["image_short_edge"]))
    reference_size = reference_shape(
        *source_size,
        short_edge=reference_config["video_short_edge"],
        max_pixels=reference_config["video_max_pixels"],
    )

    def read_pair(size):
        warp, visibility = [], []
        for index in indices:
            with Image.open(images[index]) as source:
                if source.size != source_size:
                    raise ValueError("Warp frame dimensions vary")
                warp.append(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
            with Image.open(masks[index]) as source:
                if source.size != source_size:
                    raise ValueError("Mask dimensions differ from warp dimensions")
                valid = np.asarray(source.convert("L").resize(size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0
                visibility.append(torch.from_numpy(valid.copy()))
        return warp, torch.stack(visibility)

    warp, visibility = read_pair((width, height))
    ref_warp, ref_visibility = (
        (warp, visibility) if reference_size == (width, height) else read_pair(reference_size)
    )
    return dict(
        image=target_image,
        ref_image=image.resize(image_size, Image.Resampling.LANCZOS),
        text_image=image.resize(text_image_size, Image.Resampling.LANCZOS),
        warp=warp,
        visibility=visibility,
        ref_warp=ref_warp,
        ref_visibility=ref_visibility,
        frame_indices=indices,
        export_frames=export_frames,
    )


def filled_frames(frames, masks):
    return [
        Image.fromarray(fill_reference_holes(np.asarray(frame), mask.numpy()))
        for frame, mask in zip(frames, masks)
    ]


@torch.no_grad()
def encode_geometry(pipe, pair, config):
    def encode_video(frames):
        pixels = pipe.preprocess_video(
            frames, torch_dtype=torch.float32, min_value=0, device=pipe.device
        )
        return pipe.video_vae.encode_video(
            pixels, dtype=pipe.torch_dtype, tiled=True, tile_size=256, tile_overlap=64
        ).to(device="cpu", dtype=pipe.torch_dtype)

    def encode_image(image):
        pixels = pipe.preprocess_image(image, torch_dtype=torch.float32, min_value=0).to(
            pipe.device
        )
        return pipe.video_vae.encode_video(
            pixels,
            dtype=pipe.torch_dtype,
            process_image=True,
            tiled=True,
            tile_size=256,
            tile_overlap=64,
        ).to(device="cpu", dtype=pipe.torch_dtype)

    warp = encode_video(filled_frames(pair["warp"], pair["visibility"]))
    expected = (
        1,
        24,
        ((config["frames"] - 5) // 17) * 5 + 2,
        config["height"] // 16,
        config["width"] // 16,
    )
    if tuple(warp.shape) != expected:
        raise ValueError(f"Unexpected VAE layout: {tuple(warp.shape)} != {expected}")
    reference_warp = (
        warp
        if pair["ref_warp"] is pair["warp"]
        else encode_video(filled_frames(pair["ref_warp"], pair["ref_visibility"]))
    )
    visibility = visibility_to_token_weights(
        pair["visibility"],
        VisibilityConfig("hard", quantization=None, minimum_tokens_per_frame=0),
        spatial_factor=16,
    )
    visible_mask = spatial_gate(
        visibility[None, None], SpatialControlConfig(**config["latent_mask"])
    )
    return dict(
        warp_latents=warp,
        visible_mask=visible_mask,
        first_latent=encode_image(pair["image"]),
        reference_image=encode_image(pair["ref_image"]),
        reference_warp=reference_warp,
        reference_weights=visibility_to_token_weights(
            pair["ref_visibility"], VisibilityConfig(**config["visibility"])
        ),
    )


@torch.no_grad()
def encode_context(pipe, pair, geometry, config, record):
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_PromptEmbedder,
        MiniMaxH3Unit_PackedSequenceBuilder,
    )

    reference, warp = geometry["reference_image"], geometry["reference_warp"]
    audio_shape = (2, 32, round(config["frames"] / 24 * 40))
    target = torch.zeros_like(geometry["warp_latents"])
    audio = torch.zeros(audio_shape, dtype=target.dtype)
    blocks = [
        dict(
            kind="image",
            latent_t=1,
            latent_h=reference.shape[-2],
            latent_w=reference.shape[-1],
            ref_audio_t=0,
            prepared_image=pair["ref_image"],
        ),
        dict(
            kind="video",
            latent_t=warp.shape[2],
            latent_h=warp.shape[-2],
            latent_w=warp.shape[-1],
            ref_audio_t=0,
            prepared_frames=filled_frames(pair["ref_warp"], pair["ref_visibility"]),
        ),
    ]
    prompt = MiniMaxH3Unit_PromptEmbedder().process(
        pipe,
        record["prompt"],
        ref_blocks=[dict(blocks[0], prepared_image=pair["text_image"])],
        height=config["height"],
        width=config["width"],
    )
    anchor = torch.cat(
        [
            augmented_reference(
                latent,
                target_latent_t=target.shape[2],
                visual_blocks=2,
                seed=config["condition_seed_offset"] + record["seed"],
                factor=config["condition_noise_aug"],
            )
            for latent in (reference, warp)
        ]
    )
    packed = MiniMaxH3Unit_PackedSequenceBuilder().process(
        pipe,
        prompt["prompt_embeds"],
        target,
        audio,
        text_token_tags=prompt["text_token_tags"],
        ref_blocks=blocks,
    )["packed"]
    image_rows = reference.shape[-2] * reference.shape[-1] // 4
    weights = torch.cat([torch.ones(image_rows), geometry["reference_weights"].flatten()])
    filtered = filter_reference_tokens(
        packed, anchor, weights, VisibilityConfig(**config["visibility"])
    )
    context = dict(
        imgvid_cond_noise_aug=config["condition_noise_aug"],
        audio_cond_noise_aug=1.0,
        prompt_embeds=prompt["prompt_embeds"],
        packed=filtered.packed,
        ref_visual_anchor=filtered.ref_visual_anchor,
        ref_audio_anchor=None,
        key_log_bias=filtered.key_log_bias,
    )
    return dict(
        context=move_context(context, "cpu"),
        audio_shape=audio_shape,
        ref_qwen_mode="image_only",
        reference_mask=dict(
            image_rows=image_rows, original_reference_rows=filtered.reference_row_indices.cpu()
        ),
    )


def prepare_cases(records, config, model_root, cache_dir, device):
    """Load the VAE and text encoder once each for the entire input list."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if (cache_dir / (record["id"] + ".pt")).exists():
            raise FileExistsError(
                f"Cache already exists: {record['id']}; use a new cache directory"
            )
    geometries = []
    pipe = load_pipe("Ref2VA", components=("video_vae",), device=device, model_root=model_root)
    for record in records:
        case_config = case_preprocessing(config, record.get("preprocessing"))
        pair = read_inputs(record, case_config)
        geometries.append(encode_geometry(pipe, pair, case_config))
        print(f"Encoded geometry: {record['id']}", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    pipe = load_pipe("Ref2VA", components=("text_encoder",), device=device, model_root=model_root)
    for record, geometry in zip(records, geometries):
        case_config = case_preprocessing(config, record.get("preprocessing"))
        pair = read_inputs(record, case_config)
        cache = {k: geometry[k] for k in ("warp_latents", "visible_mask", "first_latent")}
        cache.update(encode_context(pipe, pair, geometry, case_config, record))
        cache.update(
            schema=1,
            id=record["id"],
            seed=record["seed"],
            frames=case_config["frames"],
            width=case_config["width"],
            height=case_config["height"],
            fps=24,
            export_frames=pair["export_frames"],
            frame_indices=pair["frame_indices"],
            preprocessing=case_config,
            preprocessing_overrides=record.get("preprocessing", {}),
        )
        path = cache_dir / (record["id"] + ".pt")
        temporary = path.with_suffix(".tmp")
        torch.save(cache, temporary)
        temporary.replace(path)
        print(f"Prepared: {path}", flush=True)
