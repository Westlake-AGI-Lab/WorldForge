"""
Standalone 480p → 720p video upscaler using LongCat refinement LoRA.

For integrated upscaling after generation, use --enable-upscale in
run_longcat_worldforge_single.py instead.

Usage (multi-GPU via torchrun):
    torchrun run_upscale.py \
        --input_video input_480p.mp4 \
        --output_video output_720p.mp4 \
        --checkpoint_dir /path/to/checkpoint \
        --scene horse \
        --first_frame /path/to/first_frame.png \
        --t_thresh 0.5
"""

import os
import argparse
import datetime
import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
import torchvision.io as tvio

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel
from prompts import get_prompt


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_video(video_path):
    """Load video and return as list of PIL Images."""
    video, audio, info = tvio.read_video(video_path, pts_unit='sec')
    frames = []
    for i in range(video.shape[0]):
        frame_np = video[i].numpy()
        frames.append(PIL.Image.fromarray(frame_np))
    return frames


def upscale_video(args):
    # Performance optimization
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    if hasattr(torch._dynamo.config, 'cache_size_limit'):
        torch._dynamo.config.cache_size_limit = 128
    
    # Input configuration
    input_video_path = args.input_video
    output_video_path = args.output_video
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    first_frame_path = args.first_frame
    seed = args.seed
    t_thresh = args.t_thresh
   
    # Distributed setup
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    
    # Initialize context parallel
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)
    
    # Get prompt
    if args.scene:
        prompt = get_prompt(args.scene)
        print(f"[Rank {local_rank}] Using prompt for scene: {args.scene}")
    else:
        prompt = "Realistic style. In a bullet time effect video with a 3D photography style."
        print(f"[Rank {local_rank}] Using default prompt")
    
    # Load models (suppress harmless LOAD REPORT from transformers 5.x about T5 weight-tying)
    print(f"[Rank {local_rank}] Loading models...")
    import transformers as _tf; _prev_verbosity = _tf.logging.get_verbosity(); _tf.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler")
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    _tf.logging.set_verbosity(_prev_verbosity)
    
    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)
    
    # Load refinement LoRA
    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()
    
    # Load input video
    print(f"[Rank {local_rank}] Loading video: {input_video_path}")
    video_frames = load_video(input_video_path)
    print(f"[Rank {local_rank}] Loaded {len(video_frames)} frames")
    
    # Load high-resolution first frame if provided
    first_frame_image = None
    num_cond_frames = 0
    if first_frame_path:
        print(f"[Rank {local_rank}] Loading high-res first frame: {first_frame_path}")
        first_frame_image = load_image(first_frame_path)
        num_cond_frames = 1
    
    # Setup generator
    effective_seed = seed + global_rank
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(effective_seed)
    
    # Upscale
    print(f"[Rank {local_rank}] Starting upscaling (480p -> 720p, t_thresh={t_thresh})...")
    output_refine = pipe.generate_refine(
        image=first_frame_image,
        prompt=prompt,
        stage1_video=video_frames,
        num_cond_frames=num_cond_frames,
        num_inference_steps=50,
        generator=generator,
        spatial_refine_only=True,
        t_thresh=t_thresh,
    )[0]
    
    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()
    
    # Save output
    if local_rank == 0:
        print(f"Saving output to: {output_video_path}")
        output_frames = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        output_tensor = torch.from_numpy(np.array(output_frames))
        write_video(output_video_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "10"})
        print("Done!")
    
    torch_gc()


def _parse_args():
    parser = argparse.ArgumentParser(description="Upscale 480p video to 720p using refinement LoRA")
    parser.add_argument("--input_video", type=str, required=True, help="Input 480p video path")
    parser.add_argument("--output_video", type=str, required=True, help="Output 720p video path")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--scene", type=str, default=None, help="Scene name for prompt (from prompts.py)")
    parser.add_argument("--first_frame", type=str, default=None, 
                       help="High-resolution (720p) first frame image for conditioning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--t_thresh", type=float, default=0.5, 
                       help="Denoising threshold 0.0-1.0 (default: 0.5)")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--enable_compile", action='store_true', help="Enable torch.compile")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    upscale_video(args)
