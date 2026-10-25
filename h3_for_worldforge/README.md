# WorldForge V2 — H3 Trajectory LoRA

Generate camera-controlled videos with **MiniMax-H3 Ref2VA**, a trajectory LoRA,
an input image, warped RGB frames, and visibility masks. The default uses
**8-step sampling**.

## Requirements

Use Python 3.10, PyTorch 2.6 / CUDA 12.4, and an 80 GB GPU. Run these commands
from `h3_for_worldforge/` in a separate environment:

```bash
conda create -n worldforge-v2 python=3.10 -y
conda activate worldforge-v2
python -m pip install --upgrade pip wheel 'setuptools>=77,<81'
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Download [MiniMax-H3 Ref2VA](https://modelscope.cn/models/MiniMax/MiniMax-H3)
and keep its component directories under `weights/MiniMax-H3/Ref2VA/`.
Place the [trajectory LoRA](https://huggingface.co/Chenxis27/WorldForge-V2/blob/main/h3-wfv2.safetensors) at
`weights/h3-wfv2.safetensors`. See [WEIGHTS.md](WEIGHTS.md) for the folder layout.

```bash
hf download Chenxis27/WorldForge-V2 \
  h3-wfv2.safetensors --local-dir weights
```

The 8-step configuration uses
[Larry's v4 step600 EMA acceleration LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600_ema.safetensors):

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors --local-dir weights
```

## Quick Start

Three DL3DV examples are included in `examples/`: `91d19c97__arc_right`,
`8069d6f7__smooth`, and the longer `9d821bd4__orbit_return_push__90`. Each contains
`ref.png`, warped RGB frames in `frames/`, and visibility masks in `mask/`.
[examples/cases.json](examples/cases.json) lists their inputs and preserves the
frame count and framing used for each example. Prepare them once:

```bash
python infer.py prepare --input examples/cases.json \
  --model-root weights/MiniMax-H3 --cache-dir cache/examples
```

**8-step sampling (default):**

```bash
python infer.py generate --cache cache/examples \
  --model-root weights/MiniMax-H3 --lora weights/h3-wfv2.safetensors \
  --acceleration-lora weights/minimax_h3_turbo_v4_step600_ema.safetensors \
  --output outputs/8step
```

**50-step sampling:** reuse the same cache without the acceleration LoRA.

```bash
python infer.py generate --cache cache/examples --config configs/50step.json \
  --model-root weights/MiniMax-H3 --lora weights/h3-wfv2.safetensors \
  --output outputs/50step
```

The presets include the recommended settings. Use `--alpha` to adjust trajectory
LoRA strength if needed. Choose a new output directory for each run. Each result
is an MP4 with a JSON file recording its settings and timings.

Commands use `cuda:0` by default. To split the transformer across two GPUs,
pass `--devices cuda:0,cuda:1` to `generate`. Preparation uses one GPU.

## Use Your Own Data

Use the [WorldForge V1 geometry pipeline](../README.md#quick-start) to estimate
depth, define a camera path, and render warped RGB frames and visibility masks.
The input image and warped frames should use the same crop and field of view.
Masks use **white for visible pixels and black for holes**; gray values preserve
soft visibility. Frame and mask directories must contain matching sizes and
frame counts, ordered by filename.

Create a JSON file, for example `my_data/cases.json`:

```json
[
  {
    "id": "my_scene",
    "image": "ref.png",
    "warp": "frames",
    "mask": "mask",
    "prompt": "Follow the camera motion in the reference video and preserve the scene.",
    "seed": 42
  }
]
```

Paths are relative to the JSON file. Run `prepare` with
`--input my_data/cases.json --cache-dir cache/my_data`, then use that cache with
either generation command above. No ground-truth video is required.

The presets use an 832×480 output canvas and 56 internal frames. Shorter inputs
are padded for H3 and trimmed back to their original length when saved. For
longer inputs, add `"preprocessing": {"frames": 90}` to that case in the input
JSON, using a sufficient `17*n+5` frame count. The prepared cache keeps these
case-specific settings, so the generation commands stay the same.
The default image preprocessing takes a centered 16:9 view; for already aligned
inputs with another aspect ratio, add `"image_framing": null` to the case
preprocessing object, as shown in the 90-frame example.

## Credits

See [THIRD_PARTY.md](THIRD_PARTY.md) for model and runtime sources. The inference
code follows the repository's [Apache-2.0 license](../LICENSE); external models
retain their upstream licenses.
