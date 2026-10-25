from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F


def reference_shape(
    width: int, height: int, *, short_edge: int, max_pixels: int | None = None
) -> tuple[int, int]:
    """Return (width,height) using the native scale and 32px rounding rules."""
    if min(width, height) <= 0:
        raise ValueError("Reference dimensions must be positive")
    scale = float(short_edge) / min(width, height)
    if max_pixels is not None:
        scale = min(scale, float(np.sqrt(max_pixels / (width * height))))
    return tuple(max(32, int(round(value * scale / 32)) * 32) for value in (width, height))


def _patchify(latent: torch.Tensor) -> torch.Tensor:

    batch, channel, frames, height, width = latent.shape
    packed = latent.reshape(batch, channel, frames, 1, height // 2, 2, width // 2, 2)
    return torch.einsum("nctrhpwq->nthwcrpq", packed).reshape(-1, channel * 4).contiguous()


def augmented_reference(
    latent: torch.Tensor, *, target_latent_t: int, visual_blocks: int, seed: int, factor: float
) -> torch.Tensor:
    """Native .999 condition augmentation, including full noise allocation."""
    clean = _patchify(latent)
    if factor == 1:
        return clean
    shape = (1, 24, target_latent_t + visual_blocks, latent.shape[-2], latent.shape[-1])
    noise = torch.randn(
        shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=latent.dtype
    )
    noise_rows = _patchify(noise[:, :, : latent.shape[2]])
    coefficient = torch.tensor(factor, dtype=latent.dtype)
    return (coefficient * clean + (1 - coefficient) * noise_rows).contiguous()


@dataclass(frozen=True)
class SpatialControlConfig:
    spatial_erode_radius: int = 1
    spatial_blur_radius: int = 1
    temporal_blur_radius: int = 1

    def __post_init__(self):
        for key in ("spatial_erode_radius", "spatial_blur_radius", "temporal_blur_radius"):
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value <= 4:
                raise ValueError(f"{key} must be an integer in [0,4]")


def spatial_gate(visibility, config):
    """Conservative spatial support; soft spatial/time edges; exact zero in holes."""
    if visibility.ndim != 5 or visibility.shape[:2] != (1, 1):
        raise ValueError("Expected visibility [1,1,T,H,W]")
    v = visibility.float()
    if not torch.isfinite(v).all() or ((v < 0) | (v > 1)).any():
        raise ValueError("Visibility must be finite in [0,1]")
    r = config.spatial_erode_radius
    safe = (
        -F.max_pool3d(
            -F.pad(v, (r, r, r, r, 0, 0), mode="replicate"), (1, 2 * r + 1, 2 * r + 1), stride=1
        )
        if r
        else v
    )
    s, t = config.spatial_blur_radius, config.temporal_blur_radius
    if s or t:
        safe = F.avg_pool3d(
            F.pad(safe, (s, s, s, s, t, t), mode="replicate"),
            (2 * t + 1, 2 * s + 1, 2 * s + 1),
            stride=1,
        )
    return torch.minimum(v, safe.clamp(0, 1))
