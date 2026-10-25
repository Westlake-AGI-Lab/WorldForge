"""Ref2VA sampling with visibility-guided latent fusion."""

from dataclasses import dataclass
import math

import torch

from .reference import image_only_after_warp_cutoff


@dataclass(frozen=True)
class SamplingConfig:
    steps: int
    video_shift: float
    guide_steps: int
    reference_steps: int
    lora_steps: int
    audio_shift: float = 3.0

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        for name in ("guide_steps", "reference_steps", "lora_steps"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= self.steps:
                raise ValueError(f"{name} must be between 0 and steps")
        for name in ("video_shift", "audio_shift"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def sigmas(steps, shift):
    grid = torch.linspace(1, 0, steps + 1, dtype=torch.float32)[:-1]
    return (shift * grid / (1 + (shift - 1) * grid)).tolist()


def noise_like(value, generator):
    return torch.randn(value.shape, generator=generator, dtype=torch.float32).to(
        device=value.device, dtype=value.dtype
    )


def fuse_latents(video, warp, mask, sigma, previous_velocity, noise):
    work = video.float()
    reference = (1 - sigma) * warp.float() + sigma * noise.float()
    mixed = work * (1 - mask) + reference * mask
    if previous_velocity is not None:
        inferred_noise = work + (1 - sigma) * previous_velocity.float()
        mixed = mixed + sigma * (noise.float() - inferred_noise) * (1 - mask)
    return mixed.to(video.dtype)


@torch.no_grad()
def sample(cache, predict, config, *, device, gain=1.0, seed=None, set_gain=None, progress=None):
    """Use one model call per step; all step cutoffs are exclusive."""
    from .runtime import move_context

    warp = cache["warp_latents"].to(device)
    mask = cache["visible_mask"].to(device)
    target = cache["first_latent"].to(device, warp.dtype)
    seed = cache["seed"] if seed is None else seed
    rng = torch.Generator("cpu").manual_seed(seed)
    video = noise_like(warp, rng)
    audio = noise_like(torch.empty(cache["audio_shape"], dtype=warp.dtype), rng).to(device)
    context = move_context(cache["context"], device)
    tail_context = None
    if config.reference_steps < config.steps:
        tail_context = move_context(image_only_after_warp_cutoff(cache)[0], device)
    inputs = torch.zeros_like(video)
    inputs[:, :, :1] = target
    denoise_mask = torch.ones((1, 1, *video.shape[2:]), device=device, dtype=video.dtype)
    denoise_mask[:, :, :1] = 0
    video_sigmas = sigmas(config.steps, config.video_shift) + [0.0]
    audio_sigmas = sigmas(config.steps, config.audio_shift) + [0.0]
    previous = None
    for step in range(config.steps):
        sv, sa = video_sigmas[step], audio_sigmas[step]
        # Keep RNG advancement identical across guided and unguided steps.
        noise = noise_like(video, rng)
        if step < config.guide_steps:
            video = fuse_latents(video, warp, mask, sv, previous, noise)
        video[:, :, :1] = target
        active_context = context if step < config.reference_steps else tail_context
        active_context = dict(
            active_context, input_latents_video=inputs, denoise_mask_video=denoise_mask
        )
        if set_gain is not None:
            set_gain(gain if step < config.lora_steps else 0.0)
        velocity_video, velocity_audio = predict(active_context, video, audio, sv, sa)
        if velocity_video.shape != video.shape or velocity_audio.shape != audio.shape:
            raise ValueError("Model velocities do not match latent shapes")
        if not torch.isfinite(velocity_video).all() or not torch.isfinite(velocity_audio).all():
            raise FloatingPointError(f"Nonfinite model output at step {step}")
        velocity_video = velocity_video.clone()
        velocity_video[:, :, :1] = 0
        video = (video.float() + (video_sigmas[step + 1] - sv) * velocity_video.float()).to(
            video.dtype
        )
        audio = (audio.float() + (audio_sigmas[step + 1] - sa) * velocity_audio.float()).to(
            audio.dtype
        )
        previous = velocity_video.detach()
        if progress is not None:
            progress(step + 1, config.steps)
    return video
