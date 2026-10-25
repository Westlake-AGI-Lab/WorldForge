"""Prepare H3 conditions or generate videos from prepared conditions."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time


def read_config(path):
    settings = json.loads(Path(path).read_text())
    if settings.get("acceleration") not in (None, "larry_v4"):
        raise ValueError("Supported acceleration: null or larry_v4")
    return settings


def inspect_checkpoint(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        info = handle.metadata() or {}
    if info.get("format") != "h3-trajectory-attention-lora-v1":
        raise ValueError("Expected a native H3 trajectory LoRA checkpoint")
    topology = json.loads(info["topology"])
    metadata = json.loads(info.get("user_metadata", "{}"))
    if metadata.get("variant") != "Ref2VA":
        raise ValueError("This inference package requires a Ref2VA trajectory LoRA")
    ranks = {entry["rank"] for entry in topology}
    alphas = {entry["alpha"] for entry in topology}
    if len(ranks) != 1 or len(alphas) != 1:
        raise ValueError("Expected a uniform trajectory rank and alpha")
    blocks = sorted({int(entry["name"].split(".")[1]) for entry in topology})
    return dict(rank=ranks.pop(), stored_alpha=alphas.pop(), blocks=blocks)


def resolve_strength(info, alpha=None, gain=None):
    if alpha is not None and gain is not None:
        raise ValueError("Choose alpha or gain, not both")
    if alpha is not None:
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        gain = alpha / info["stored_alpha"]
    gain = 1.0 if gain is None else gain
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("gain must be finite and nonnegative")
    return gain, gain * info["stored_alpha"]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_cache(path):
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "id",
        "seed",
        "frames",
        "height",
        "width",
        "fps",
        "warp_latents",
        "visible_mask",
        "first_latent",
        "audio_shape",
        "context",
        "reference_mask",
        "ref_qwen_mode",
    }
    if not isinstance(cache, dict) or required - cache.keys() or cache.get("schema") != 1:
        raise ValueError(f"Unsupported prepared cache: {path}")
    if Path(str(cache["id"])).name != cache["id"] or cache["id"] in (".", ".."):
        raise ValueError("Cache ID must be a simple filename")
    if type(cache["frames"]) is not int or cache["frames"] < 5 or (cache["frames"] - 5) % 17:
        raise ValueError("Cache frame count must be 17*n+5")
    if type(cache["seed"]) is not int or cache["seed"] < 0:
        raise ValueError("Cache seed must be a nonnegative integer")
    warp, mask, first = (cache[k] for k in ("warp_latents", "visible_mask", "first_latent"))
    expected = (
        1,
        24,
        ((cache["frames"] - 5) // 17) * 5 + 2,
        cache["height"] // 16,
        cache["width"] // 16,
    )
    if tuple(warp.shape) != expected or tuple(mask.shape) != (1, 1, *warp.shape[2:]):
        raise ValueError("Cache has inconsistent latent and mask shapes")
    if tuple(first.shape) != (1, 24, 1, *warp.shape[-2:]):
        raise ValueError("First-image latent shape differs from the target")
    if not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
        raise ValueError("Mask must contain finite values in [0,1]")
    if not torch.isfinite(warp).all() or not torch.isfinite(first).all():
        raise ValueError("Cache latents must be finite")
    return cache


def write_video(frames, path, count, fps):
    import imageio.v2 as imageio
    import torch

    if frames.ndim != 5 or frames.shape[:2] != (1, 3) or not torch.isfinite(frames).all():
        raise ValueError("Expected finite decoded RGB [1,3,T,H,W]")
    if not 1 <= count <= frames.shape[2]:
        raise ValueError("Invalid export frame count")
    array = (
        (frames[0, :, :count].float().clamp(0, 1).permute(1, 2, 3, 0) * 255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    temporary = path.with_name(path.stem + ".tmp.mp4")
    try:
        with imageio.get_writer(
            str(temporary),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        ) as writer:
            for frame in array:
                writer.append_data(frame)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(args, settings):
    import torch
    from .lora import load_attention_lora, set_adapter_gain
    from .placement import place_model, synchronize
    from .preprocess import case_preprocessing
    from .runtime import load_pipe, reference_attention_bias, velocity
    from .sampling import SamplingConfig, sample

    config = SamplingConfig(**settings["sampling"])
    info = inspect_checkpoint(args.lora)
    gain, effective_alpha = resolve_strength(info, args.alpha, args.gain)
    devices = args.devices.split(",")
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices):
        raise ValueError("Use one or two distinct CUDA devices")
    if any(torch.device(d).type != "cuda" or torch.device(d).index is None for d in devices):
        raise ValueError("Use explicit CUDA device names")
    accelerated = settings["acceleration"] == "larry_v4"
    if accelerated and (not args.acceleration_lora or config.steps != 8):
        raise ValueError("The Larry preset requires its checkpoint and an 8-step schedule")
    if accelerated:
        from .acceleration import inspect_larry

        inspect_larry(args.acceleration_lora)
    if not accelerated and args.acceleration_lora:
        raise ValueError("Use an accelerated preset with --acceleration-lora")
    if not math.isfinite(args.acceleration_gain) or args.acceleration_gain < 0:
        raise ValueError("acceleration-gain must be finite and nonnegative")
    paths = (
        sorted(Path(args.cache).glob("*.pt")) if Path(args.cache).is_dir() else [Path(args.cache)]
    )
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError(args.cache)
    output = Path(args.output)
    ids = set()
    for path in paths:
        cache = load_cache(path)
        expected_preprocessing = case_preprocessing(
            settings["preprocessing"], cache.get("preprocessing_overrides")
        )
        if cache.get("preprocessing") != expected_preprocessing:
            raise ValueError("Cache preprocessing differs from the config; prepare a new cache")
        if cache["id"] in ids:
            raise ValueError("Duplicate cache IDs")
        ids.add(cache["id"])
        if any((output / (cache["id"] + suffix)).exists() for suffix in (".mp4", ".json")):
            raise FileExistsError(
                f"Output already exists for {cache['id']}; use a new output directory"
            )
    del cache
    summary = dict(
        sampling=asdict(config),
        trajectory=info,
        trajectory_gain=gain,
        effective_alpha=effective_alpha,
        acceleration=settings["acceleration"],
        acceleration_gain=args.acceleration_gain,
        devices=devices,
        cases=len(paths),
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for H3 generation")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(trajectory_sha256=sha256(args.lora), **summary)
    pipe = load_pipe("Ref2VA", components=("dit",), device="cpu", model_root=args.model_root)
    load_attention_lora(pipe.dit, args.lora, expected_qkv_mode="separate")
    if accelerated:
        from .acceleration import attach_larry

        attach_larry(pipe.dit, args.acceleration_lora, args.acceleration_gain)
        identity["acceleration_sha256"] = sha256(args.acceleration_lora)
    pipe.dit.eval()
    pipe.device = torch.device(devices[0])
    decoder = None
    with torch.no_grad():
        for path in paths:
            cache = load_cache(path)
            place_model(pipe.dit, devices, mlp_chunk_tokens=2048 if cache["frames"] > 56 else 0)

            def predict(context, video, audio, sv, sa):
                with reference_attention_bias(pipe.dit, context.get("key_log_bias")):
                    return velocity(pipe, context, video, audio, sv, sa)

            synchronize(devices)
            start = time.perf_counter()
            latent = sample(
                cache,
                predict,
                config,
                device=devices[0],
                gain=gain,
                seed=args.seed,
                set_gain=lambda value: set_adapter_gain(pipe.dit, "trajectory", value),
                progress=lambda step, total: print(f"{cache['id']}: {step}/{total}", flush=True),
            )
            synchronize(devices)
            sampling_seconds = time.perf_counter() - start
            pipe.dit.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            if decoder is None:
                decoder = load_pipe(
                    "Ref2VA",
                    components=("video_vae",),
                    device=devices[0],
                    model_root=args.model_root,
                )
            else:
                decoder.video_vae.to(devices[0])
            synchronize(devices)
            start = time.perf_counter()
            frames = decoder.video_vae.decode_video(
                latent, dtype=torch.bfloat16, tiled=True, tile_size=256, tile_overlap=64
            ).cpu()
            synchronize(devices)
            decode_seconds = time.perf_counter() - start
            destination = output / (cache["id"] + ".mp4")
            count = cache.get("export_frames", cache["frames"])
            write_video(frames, destination, count, cache["fps"])
            metadata = dict(
                identity,
                id=cache["id"],
                seed=cache["seed"] if args.seed is None else args.seed,
                preprocessing=cache.get("preprocessing"),
                frames=cache["frames"],
                export_frames=count,
                width=cache["width"],
                height=cache["height"],
                sampling_seconds=sampling_seconds,
                decode_seconds=decode_seconds,
                timing_scope="sampling only; decoding separate; loading excluded",
            )
            destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            decoder.video_vae.to("cpu")
            del latent, frames, cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Saved: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Encode image / warp / mask inputs; no GT is required")
    prepare.add_argument("--input", required=True, help="Case JSON or JSON list")
    default_config = str(Path(__file__).resolve().parents[1] / "configs" / "8step.json")
    prepare.add_argument("--config", default=default_config, help="Sampling config (default: 8-step)")
    prepare.add_argument("--model-root", required=True)
    prepare.add_argument("--cache-dir", required=True)
    prepare.add_argument("--device", default="cuda:0")
    generate_parser = sub.add_parser(
        "generate", help="Generate from one cache or a directory of caches"
    )
    generate_parser.add_argument("--config", default=default_config, help="Sampling config (default: 8-step)")
    generate_parser.add_argument("--model-root", required=True)
    generate_parser.add_argument("--cache", required=True)
    generate_parser.add_argument("--lora", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--devices", default="cuda:0")
    strength = generate_parser.add_mutually_exclusive_group()
    strength.add_argument("--alpha", type=float, help="Trajectory LoRA alpha (default: preset value; 8-step 128, 50-step 96)")
    strength.add_argument("--gain", type=float)
    generate_parser.add_argument("--acceleration-lora")
    generate_parser.add_argument("--acceleration-gain", type=float, default=1.0)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--dry-run", action="store_true")
    inspect_parser = sub.add_parser(
        "inspect", help="Read LoRA topology without loading the base model"
    )
    inspect_parser.add_argument("checkpoint")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))
        return
    settings = read_config(args.config)
    if args.command == "prepare":
        from .preprocess import load_cases, prepare_cases

        prepare_cases(
            load_cases(args.input),
            settings["preprocessing"],
            args.model_root,
            args.cache_dir,
            args.device,
        )
    else:
        if args.seed is not None and args.seed < 0:
            parser.error("seed must be nonnegative")
        if args.alpha is None and args.gain is None:
            args.alpha = float(settings.get("trajectory_alpha", 96.0))
        generate(args, settings)
