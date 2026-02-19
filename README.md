<img src="teaser/worldforge.png" alt="WorldForge Logo" width="45" align="left" style="margin-right: 10px;"> 

# WorldForge
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2509.15130-b31b1b)](https://arxiv.org/abs/2509.15130)&nbsp;
[![project page](https://img.shields.io/badge/Project%20page-WorldForge-blue)](https://worldforge-agi.github.io)&nbsp;
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2509.15130)&nbsp;

</div>

**[WorldForge: Unlocking Emergent 3D/4D Generation in Video Diffusion via Training-Free Guidance](https://arxiv.org/abs/2509.15130)**

<div align="center">
  <img src="teaser/Page_office_30fps.gif" alt="WorldForge Teaser" width="100%">
</div>

[Chenxi Song](https://chenxi-song.github.io)<sup>1</sup>, [Yanming Yang](https://2hiTee.github.io)<sup>1</sup>, [Tong Zhao](https://tongzhao1030.github.io)<sup>1</sup>, [Ruibo Li](https://scholar.google.com/citations?hl=zh-CN&user=qtGY5T4AAAAJ&view_op=list_works&sortby=pubdate)<sup>2</sup>, [Chi Zhang](https://icoz69.github.io)<sup>1*</sup>

<sup>1</sup>AGI Lab, Westlake University  
<sup>2</sup>The College of Computing and Data Science, Nanyang Technological University  
<sup>*</sup>Corresponding Author

## Update
- [2026.02] 🔥 **Code released!** VGGT 3D warping, DepthCrafter 4D warping, Wan2.1 & LongCat-Video inference are now available.
- [2025.09] [arXiv](https://arxiv.org/abs/2509.15130) preprint is available.
- [2025.09] [Project page](https://worldforge-agi.github.io) is online.

<details>
<summary><b>Introduction</b></summary>

**WorldForge** is a **training-free** framework that unlocks the world-modeling potential of video diffusion models for controllable **3D/4D generation**. By leveraging latent world priors in pretrained video diffusion models, WorldForge achieves precise trajectory control and photorealistic content generation — all **without any additional training**.

![image](teaser/pipeline3.png)

The core idea is a *warping-and-repainting* pipeline with three inference-time mechanisms (IRR, FLF, DSG) that jointly inject trajectory-aligned guidance while preserving visual fidelity. For details, please refer to our [paper](https://arxiv.org/abs/2509.15130) and [project page](https://worldforge-agi.github.io).

</details>

<details>
<summary><b>What's Released</b></summary>

| Module | Description |
| --- | --- |
| `vggt/` | VGGT-based 3D scene warping from a single image |
| `DepthCrafter/` | DepthCrafter-based 4D warping for dynamic video scenes |
| `wan_for_worldforge/` | WorldForge inference with [Wan-2.1](https://github.com/Wan-Video/Wan2.1) (480p & 720p) |
| `longcat_for_worldforge/` | WorldForge inference with [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) (480p, distilled & upscaling) |

**Tips:**
- **LongCat-Video** supports a distilled mode (16 steps) with 480p→720p upscaling, offering faster generation. It works well on realistic scenes; for stylized or non-photorealistic content, Wan2.1 720p may yield better results.
- **Wan2.1 720p** (50 steps) generally delivers higher visual quality, at the cost of longer inference time.

</details>

## Installation

### 1. Create environment

```bash
conda create -n worldforge python=3.11 -y
conda activate worldforge
```

### 2. Install PyTorch

```bash
# Choose your CUDA version from https://pytorch.org
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install GPU acceleration libraries

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124

# flash-attn (use pre-built wheel, source build may have ABI issues)
# Torch 2.6 + CUDA 12 + Python 3.11:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-deps
# Other combos: https://github.com/Dao-AILab/flash-attention/releases (pick cxx11abiFALSE)
```

### 5. Install pytorch3d (for DepthCrafter warping)

```bash
conda install -c nvidia cuda-toolkit -y
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; c=torch.cuda.get_device_capability(); print(f'{c[0]}.{c[1]}')")
export MAX_JOBS=$(( $(nproc) / 2 ))
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 6. Install LightGlue (for VGGT warping)

```bash
pip install --no-build-isolation git+https://github.com/cvg/LightGlue.git
```

### Model Weights

- **Wan-2.1**: Download model weights from the official [Wan-2.1 repository](https://github.com/Wan-Video/Wan2.1). You will need `Wan2.1-I2V-14B-480P` and/or `Wan2.1-I2V-14B-720P`.
- **LongCat-Video**: Download model weights from the official [LongCat-Video repository](https://github.com/meituan-longcat/LongCat-Video). Both standard and distilled checkpoints are supported.

Set the model weight paths in the corresponding `run_test_case.sh` scripts before running.

## Quick Start

VGGT 3D scene warping — generate warped frames from single/few images:
```bash
bash vggt/run_test_case.sh
```

DepthCrafter 4D dynamic warping — warp a video sequence with depth-guided trajectory:
```bash
bash DepthCrafter/run_test_case.sh
```

Wan2.1 WorldForge inference — high-quality video generation (edit `MODELS_DIR` in script first):
```bash
bash wan_for_worldforge/run_test_case.sh
```

LongCat-Video WorldForge inference — fast generation with optional distillation & upscaling (edit `CHECKPOINT_DIR` first):
```bash
bash longcat_for_worldforge/run_test_case.sh
```

<details>
<summary>💡 LongCat Interactive Notebook</summary>

We also provide a Jupyter Notebook (`longcat_for_worldforge/longcat_interactive.ipynb`) for continuous inference — models are loaded once and kept in VRAM, so you can run multiple experiments without reloading weights each time. See the parameter details in the sections below.

</details>

## Running on Your Own Data

The overall pipeline is: **warping → prompt → video inference**. First generate warped frames and masks with a 3D vision model, then feed them into a video diffusion model for repainting.

### Step 1: Warping

#### VGGT — 3D scene warping (single / few images)

```bash
cd vggt
python run_warp.py \
    --image_path <your_image_folder> \
    --output_path <output_folder> \
    --camera <camera_index> \
    --direction <direction> \
    --degree <angle> \
    --frame_single <num_frames> \
    --look_at_depth <depth>
```

<details>
<summary>Key parameters</summary>

| Parameter | Description |
| --- | --- |
| `--camera` | Which input view to use as the source (index starting from 0) |
| `--direction` | Camera motion direction: `left`, `right`, `up`, `down`, `forward`, `backward` |
| `--degree` | Rotation angle in degrees (or movement ratio for forward/backward) |
| `--frame_single` | Number of output frames to generate |
| `--look_at_depth` | Focus depth coefficient (1.0 = scene mean depth; smaller = closer focus) |

</details>

#### DepthCrafter — 4D dynamic warping (video input)

```bash
cd DepthCrafter
python warp_depthcrafter.py \
    --video_path <your_video_or_image_folder> \
    --output_path <output_folder> \
    --direction <direction> \
    --degree <angle> \
    --look_at_depth <depth> \
    --enable_edge_filter
```

<details>
<summary>Key parameters</summary>

| Parameter | Description |
| --- | --- |
| `--direction` | Camera motion direction: `left`, `right`, `up`, `down` |
| `--degree` | Warping angle in degrees |
| `--look_at_depth` | Focus depth multiplier (similar to VGGT) |
| `--zoom` | Optional zoom mode: `zoom_in`, `zoom_out`, `none` |
| `--stable` | Stable mode: complete camera motion in first N frames, then hold |
| `--enable_edge_filter` | Reduce artifacts at depth boundaries (recommended) |

</details>

The warping output (frames + masks) will be saved to the output folder. Pass this folder as `--video-ref` to the video model in the next step.

### Step 2: Prepare Prompts

Add a text prompt for your scene to `prompts.py`. We recommend using an LLM (e.g., GPT, Gemini) to generate prompts — see existing examples in `prompts.py` for the expected style. For **static scenes**, emphasize stillness (*"completely frozen"*, *"utterly motionless"*) and avoid words implying movement.

### Step 3: Video Inference

**Our method offers flexible parameter combinations.** The `run_test_case.sh` scripts provide commonly used parameter grids. We recommend tuning parameters for your specific scene to find the best combination.

Key tunable parameters (see `run_test_case.sh` for both Wan and LongCat):

| Parameter | Description |
| --- | --- |
| `--resolution` | Output resolution: `480p` or `720p` |
| `--omega` | DSG guidance strength |
| `--guide-steps` | Number of initial denoising steps with guided fusion |
| `--resample-steps` | Resampling iterations per denoising step |
| `--resample-round` | Total steps for resample guidance (typically `guide-steps` + 0~1) |
| `--transition-distance` | Mask edge softening distance in pixels (0 = hard edge) |
| `--guidance-scale` | CFG scale (ignored when using distilled model) |
| `--max-replace` | *(LongCat only)* Max FLF replacement channels per step |
| `--use_distill` | *(LongCat only)* Enable 16-step distilled mode for faster inference |
| `--enable-upscale` | *(LongCat only)* Upscale 480p output to 720p |

> Wan2.1 and LongCat-Video have different model priors, so optimal parameter ranges may differ. We suggest starting from the defaults in `run_test_case.sh` and adjusting based on your results.

>**Quick tuning tips:** Start by adjusting `--guide-steps` (IRR iterations) — for Wan's 50-step sampling, 10–25 typically works well; for LongCat's 50 steps, try 20–30. `--resample-round` adds extra IRR passes without warping injection for quality refinement (keep it 0 or 1; higher values lose trajectory). `--omega` controls DSG strength — 4 or 6 is recommended; too high causes artifacts. `--resample-steps` can be fixed at 2. `--transition-distance` softens mask edges — 15, 20, or 25 are good choices. `--guidance-scale` is usually fixed at 4. `--max-replace` (LongCat only) works best at 2 or 3 due to LongCat's different channel characteristics. These are empirical guidelines — some scenes may benefit from values outside these ranges.

### Exploring Further

We encourage the community to try newer 3D vision models for the warping stage (e.g., [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3), [PI3](https://github.com/yyfz/Pi3), etc.), and to port WorldForge to other video diffusion models (e.g., [LTX-2](https://github.com/Lightricks/LTX-2)) for even better results.

## TODO
- [x] Paper released on arXiv
- [x] Project page available
- [x] Code released: VGGT warping, DepthCrafter warping, Wan2.1 inference, LongCat-Video inference
- [ ] **Mega-SAM** dynamic video warping
- [ ] **UniDepth** single-view warping
- [ ] **Lang-SAM** video editing framework
- [ ] **Multi-GPU** parallel inference acceleration

We welcome contributions, issues, and discussions! If you have ported WorldForge to other video diffusion models, feel free to share.

> P.S. Stay tuned for our upcoming new work — a faster, simpler, and more unified world model architecture. Coming soon! 🚀🚀🚀

## Acknowledgments

We thank the research community for their valuable contributions to video diffusion models and 3D/4D generation. Special thanks to the following open-source projects that inspired and supported our work:

- [**Wan-2.1**](https://github.com/Wan-Video/Wan2.1) - Large-scale video generation model
- [**LongCat-Video**](https://github.com/meituan-longcat/LongCat-Video) - Efficient long video generation with distillation
- [**SVD (Stable Video Diffusion)**](https://github.com/Stability-AI/generative-models) - Video diffusion model by Stability AI
- [**VGGT**](https://vgg-t.github.io/) - Visual Geometry Grounded Transformer
- [**ReCamMaster**](https://github.com/KwaiVGI/ReCamMaster) - Trajectory-controlled video generation
- [**TrajectoryCrafter**](https://trajectorycrafter.github.io/) - Trajectory-based video synthesis
- [**NVS-Solver**](https://github.com/ZHU-Zhiyu/NVS_Solver) - Novel view synthesis solution
- [**ViewExtrapolator**](https://kunhao-liu.github.io/ViewExtrapolator) - View extrapolation for 3D scenes
- [**DepthCrafter**](https://github.com/Tencent/DepthCrafter) - Video sequence depth estimation
- [**Mega-SAM**](https://github.com/mega-sam/mega-sam) - Video depth and pose estimation

## Citation

```bibtex
@misc{song2025worldforgeunlockingemergent3d4d,
  title={WorldForge: Unlocking Emergent 3D/4D Generation in Video Diffusion Model via Training-Free Guidance}, 
  author={Chenxi Song and Yanming Yang and Tong Zhao and Ruibo Li and Chi Zhang},
  year={2025},
  url={https://arxiv.org/abs/2509.15130}, 
}
```

## Contact

For questions and discussions, please feel free to open an [issue](https://github.com/worldforge-agi/worldforge/issues) or contact:
- Chenxi Song: songchenxi@westlake.edu.cn
- Chi Zhang: chizhang@westlake.edu.cn

---

<div align="center">
<strong>🌟 Star us on GitHub if you find WorldForge useful! 🌟</strong>
</div>
