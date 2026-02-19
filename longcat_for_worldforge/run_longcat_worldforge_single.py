"""
WorldForge: Single-GPU Inference for LongCat Video I2V with Trajectory Control

This script performs image-to-video generation with optional trajectory guidance
via resample-based guidance and optical flow channel filtering (FLF).

Usage:
    python run_longcat_worldforge_single.py \
        --checkpoint_dir /path/to/checkpoint \
        --video-ref /path/to/warped_frames \
        --scene girl \
        --num-frames 93 \
        --guided \
        --output output/result.mp4
"""

import os
import argparse
import datetime
import PIL.Image
import numpy as np
import glob
import torch
import torch.distributed as dist
from scipy.ndimage import distance_transform_edt
import torchvision.io as tvio

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from prompts import get_prompt, list_available_scenes
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


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


def read_frames_from_directory(directory):
    """Read video frames and masks from directory."""
    print(f"Reading frames from: {directory}")
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    all_files = []
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        all_files.extend(glob.glob(pattern))
    
    all_files = sorted(all_files)
    
    if not all_files:
        raise ValueError(f"No images found in {directory}")
    
    frame_files = []
    mask_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename.startswith('mask_'):
            mask_files.append(file_path)
        else:
            frame_files.append(file_path)
    
    frames = [PIL.Image.open(f).convert('RGB') for f in frame_files]
    masks = [PIL.Image.open(f).convert('L') for f in mask_files]
    
    if not masks and frames:
        print("No mask images found, creating zero masks")
        masks = [PIL.Image.new('L', frames[0].size, 0) for _ in frames]
    
    if len(masks) != len(frames):
        print(f"Warning: mask count ({len(masks)}) != frame count ({len(frames)})")
        while len(masks) < len(frames):
            masks.append(masks[-1] if masks else PIL.Image.new('L', frames[0].size, 0))
        if len(masks) > len(frames):
            masks = masks[:len(frames)]
    
    first_frame = frames[0] if frames else None
    first_frame_path = frame_files[0] if frame_files else None
    print(f"Loaded {len(frames)} frames and {len(masks)} masks")
    return frames, masks, first_frame, first_frame_path


def soften_mask(mask_array, transition_distance=15, decay_type='sine'):
    """Soften mask boundaries with smooth transitions."""
    softened_mask = mask_array.copy().astype(np.float32)
    
    for frame_idx in range(mask_array.shape[0]):
        current_mask = mask_array[frame_idx].astype(bool)
        
        if np.all(current_mask) or np.all(~current_mask):
            continue
        
        softened_frame = mask_array[frame_idx].copy().astype(np.float32)
        
        distance_from_ones = distance_transform_edt(current_mask)
        distance_from_zeros = distance_transform_edt(~current_mask)
        
        ones_transition = current_mask & (distance_from_ones <= transition_distance)
        
        def smooth_transition(t, decay_type):
            t = np.clip(t, 0.0, 1.0)
            if decay_type == 'linear':
                return t
            elif decay_type == 'exponential':
                return 1.0 - np.exp(-3.0 * t)
            elif decay_type == 'sine':
                return np.sin(np.pi / 2 * t)
            elif decay_type == 'cosine':
                return 1.0 - np.cos(np.pi / 2 * t)
            else:
                raise ValueError(f"Unsupported decay type: {decay_type}")
        
        if np.any(ones_transition):
            distances = distance_from_ones[ones_transition]
            t_values = distances / transition_distance
            transition_values = smooth_transition(t_values, decay_type)
            softened_frame[ones_transition] = transition_values
        
        softened_mask[frame_idx] = softened_frame
    
    return softened_mask


def generate(args):
    # Performance optimization
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    if hasattr(torch._dynamo.config, 'cache_size_limit'):
        torch._dynamo.config.cache_size_limit = 128
    
    # Load configuration
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile
    use_distill = args.use_distill
    
    # Single GPU setup
    local_rank = 0
    global_rank = 0
    num_processes = 1
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    else:
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    
    # Initialize single-process distributed environment (required for context_parallel)
    if not dist.is_initialized():
        import socket
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        free_port = find_free_port()
        master_port = str(free_port)
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{master_port}",
            rank=0,
            world_size=1,
            timeout=datetime.timedelta(seconds=3600)
        )
    
    # Initialize context parallel
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)
    
    # Load models (suppress harmless LOAD REPORT from transformers 5.x about T5 weight-tying)
    print("Loading models...")
    import transformers as _tf; _prev_verbosity = _tf.logging.get_verbosity(); _tf.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler")
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    _tf.logging.set_verbosity(_prev_verbosity)
    
    # Load LoRA for distillation if needed
    if use_distill:
        cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
        dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
        dit.enable_loras(['cfg_step_lora'])
    
    if enable_compile:
        print("Compiling DiT model...")
        dit = torch.compile(dit)
    
    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)
    print("Models loaded successfully.")
    
    # Load video reference and masks
    frames, masks, first_frame, first_frame_path = read_frames_from_directory(args.video_ref)
    
    # Load input image
    if args.image is not None:
        print(f"Loading input image: {args.image}")
        image = load_image(args.image)
    else:
        print("Using first frame as input image")
        image = first_frame
        if image is None:
            raise ValueError("No first frame available, please specify --image")
    
    # Determine target size based on resolution
    scale_factor_spatial = pipe.vae_scale_factor_spatial * 2
    if pipe.dit.cp_split_hw is not None:
        scale_factor_spatial *= max(pipe.dit.cp_split_hw)
    height, width = pipe.get_condition_shape(image, args.resolution, 
                                             scale_factor_spatial=scale_factor_spatial)
    
    # Process video frames
    video_ref = None
    if frames:
        try:
            resized_frames = [frame.resize((width, height)) for frame in frames]
            video_frames = torch.stack([
                torch.tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0
                for frame in resized_frames
            ])
            video_ref = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
            print(f"Video reference: {video_ref.shape}")
        except Exception as e:
            print(f"Error processing video frames: {e}")
            video_ref = None
    
    # Process masks
    mask = None
    if masks:
        try:
            resized_masks = [mask.resize((width, height)) for mask in masks]
            mask_array = np.stack([np.array(mask) / 255.0 for mask in resized_masks])
            
            if args.soften_mask:
                mask_array = soften_mask(mask_array, args.transition_distance, args.decay_type)
            
            mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
            print(f"Mask: {mask.shape}")
            
            if mask.shape[1] != 1:
                raise ValueError(f"Mask should be single-channel, got {mask.shape[1]}")
        except Exception as e:
            print(f"Error processing masks: {e}")
            mask = None
    
    # Setup prompts
    if args.prompt:
        prompt = args.prompt
    elif args.scene:
        prompt = get_prompt(args.scene)
        print(f"Using scene: {args.scene}")
    else:
        prompt = "A high-quality video with smooth motion and clear details"
    
    # Negative prompts
    negative_prompt_static = (
        "Blink, twinkle, waggle, speak, wind, windy, leaves shaking, leaves tremble, "
        "sighboard, background dynamics, dynamic imagery, gray sky, hazy sky, overcast, "
        "gloomy sky, dim, murky, smoggy, shake, object motion blur, streaking objects, "
        "object jitter, camera shake, time flow, illogical composition, bright tones, "
        "overexposed, blurred details, subtitles, text, logo, overall gray, worst quality, "
        "low quality, JPEG compression residue, ugly, incomplete, sudden scene shift, "
        "incoherent scene jump, extra fingers, poorly drawn hands, poorly drawn faces, "
        "deformed, disfigured, misshapen limbs, fused fingers, any movement, character motion, "
        "slight object movement, object swaying, character micro-movements, subtle object rotation, "
        "object vibration, messy background, three legs, many people in the background, walking, "
        "scene changes, visual detail movement, object disintegration, object breakage."
    )
    
    negative_prompt_dynamic = (
        "Streaking objects, mosaic, grainy, pixelated, noise, flickering, cropped, glitch, "
        "fragmented, broken, artifacts, chromatic aberration, micro camera shake, grid, tiling, "
        "blurry, camera shake, sudden scene shift, incoherent scene jump, sudden object appearance, "
        "blinking, object jitter, camera shake, illogical composition, bright tones, overexposed, "
        "blurred details, subtitles, overall gray, solid color, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "messy background, three legs, many people in the background, walking backwards"
    )
    
    static_bool = (args.static == 'True')
    
    if args.negative_prompt:
        negative_prompt = args.negative_prompt
    elif static_bool:
        negative_prompt = negative_prompt_static
    else:
        negative_prompt = negative_prompt_dynamic
    
    # Setup generator
    global_seed = args.seed
    seed = global_seed + global_rank
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    
    # Display configuration
    print("\n" + "="*60)
    print("Generation Configuration:")
    print(f"  Resolution: {args.resolution} | Frames: {args.num_frames} | Steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale if not use_distill else 1.0} | Distill: {use_distill}")
    print(f"  Static: {static_bool} | Seed: {seed}")
    if args.guided:
        print(f"  Guided: ON (resample={args.resample_steps}, guide_steps={args.guide_steps}, "
              f"rounds={args.resample_round}, omega={args.omega}/{args.omega_resample})")
    if args.use_pca_channel_selection:
        print(f"  FLF (Flow Latent Filtering): ON" + 
              (f" (max_replace={args.max_replace})" if args.max_replace else ""))
    if args.soften_mask:
        print(f"  Mask softening: ON (distance={args.transition_distance}, decay={args.decay_type})")
    print("="*60 + "\n")
    
    # Prepare output directory
    temp_output_path = args.output
    if os.path.isdir(temp_output_path) or not os.path.splitext(temp_output_path)[1]:
        os.makedirs(temp_output_path, exist_ok=True)
    else:
        output_dir = os.path.dirname(temp_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Generate video
    print("Starting video generation...")
    
    output = pipe.generate_i2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=args.resolution,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        use_distill=use_distill,
        guidance_scale=args.guidance_scale if not use_distill else 1.0,
        num_videos_per_prompt=1,
        generator=generator,
        # Resample guidance parameters
        video_ref=video_ref,
        mask=mask,
        guided=args.guided,
        resample_steps=args.resample_steps,
        guide_steps=args.guide_steps,
        resample_round=args.resample_round,
        omega=args.omega,
        omega_resample=args.omega_resample,
        use_pca_channel_selection=args.use_pca_channel_selection,
        static=static_bool,
        max_replace_threshold=args.max_replace,
    )[0]
    
    # Save video
    output_path = args.output
    if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
        filename = f"{args.scene}.mp4" if args.scene else "output.mp4"
        output_path = os.path.join(output_path, filename)
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving video to: {output_path}")
    output_pil = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    output_pil = [PIL.Image.fromarray(img) for img in output_pil]
    output_tensor = torch.from_numpy(np.array(output_pil))
    write_video(output_path, output_tensor, fps=args.fps, video_codec="libx264", options={"crf": "10"})
    
    # Save PNG sequence if requested
    if args.save_png:
        dir_path = os.path.dirname(output_path)
        basename = os.path.basename(output_path)
        name_without_ext = os.path.splitext(basename)[0]
        png_folder_name = f"imgs_{name_without_ext}"
        png_output_dir = os.path.join(dir_path, png_folder_name) if dir_path else png_folder_name
        
        os.makedirs(png_output_dir, exist_ok=True)
        print(f"Saving PNG sequence to: {png_output_dir}")
        
        for i, frame in enumerate(output_pil):
            if isinstance(frame, PIL.Image.Image):
                frame_pil = frame
            elif isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().numpy()
                if frame_np.dtype in [np.float32, np.float64]:
                    frame_np = (frame_np * 255).astype(np.uint8)
                if len(frame_np.shape) == 3 and frame_np.shape[0] in [1, 3, 4]:
                    frame_np = np.transpose(frame_np, (1, 2, 0))
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                frame_pil = PIL.Image.fromarray(frame_np)
            elif isinstance(frame, np.ndarray):
                frame_np = frame.copy()
                if frame_np.dtype in [np.float32, np.float64]:
                    frame_np = (frame_np * 255).astype(np.uint8)
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                frame_pil = PIL.Image.fromarray(frame_np)
            else:
                continue
            
            frame_pil.save(os.path.join(png_output_dir, f"{i:05d}.png"))
        
        print(f"Saved {len(output_pil)} frames to {png_output_dir}")
    
    print("Generation complete!")
    
    # Upscale to 720p if requested
    if args.enable_upscale:
        print("\nStarting upscaling to 720p...")
        
        output_base, output_ext = os.path.splitext(output_path)
        output_720p_path = f"{output_base}_720p{output_ext}"
        
        # Load refinement LoRA
        refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
        pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
        pipe.dit.enable_loras(['refinement_lora'])
        pipe.dit.enable_bsa()
        
        # Load the generated 480p video
        video_frames = load_video(output_path)
        print(f"Loaded {len(video_frames)} frames for upscaling")
        
        # Load and resize first frame to 720p
        first_frame_image = None
        num_cond_frames = 0
        if first_frame_path:
            first_frame_image = load_image(first_frame_path)
            
            scale_factor_spatial_720p = pipe.vae_scale_factor_spatial * 2
            if pipe.dit.cp_split_hw is not None:
                scale_factor_spatial_720p *= max(pipe.dit.cp_split_hw)
            
            height_720p, width_720p = pipe.get_condition_shape(
                first_frame_image, '720p', scale_factor_spatial=scale_factor_spatial_720p
            )
            
            first_frame_image = first_frame_image.resize((width_720p, height_720p))
            num_cond_frames = 1
        
        # Setup generator (must be CPU for torch.randn compatibility)
        generator_upscale = torch.Generator(device='cpu')
        generator_upscale.manual_seed(seed)
        
        # Upscale
        output_refine = pipe.generate_refine(
            image=first_frame_image,
            prompt=prompt,
            stage1_video=video_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=50,
            generator=generator_upscale,
            spatial_refine_only=True,
            t_thresh=args.t_thresh,
        )[0]
        
        pipe.dit.disable_all_loras()
        pipe.dit.disable_bsa()
        
        # Save 720p video
        print(f"Saving 720p video to: {output_720p_path}")
        output_720p_frames = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        output_720p_tensor = torch.from_numpy(np.array(output_720p_frames))
        write_video(output_720p_path, output_720p_tensor, fps=args.fps, video_codec="libx264", options={"crf": "10"})
        
        print("Upscaling complete!")
        torch_gc()


def _parse_args():
    parser = argparse.ArgumentParser(description='WorldForge: LongCat Video I2V with Trajectory Control')
    
    # Model configuration
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Model checkpoint directory')
    parser.add_argument('--context_parallel_size', type=int, default=1, help='Context parallel size')
    parser.add_argument('--enable_compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--use_distill', action='store_true', help='Use 16-step distillation mode')
    
    # Input configuration
    parser.add_argument('--video-ref', type=str, required=True, help='Video frames directory for guidance')
    parser.add_argument('--image', type=str, default=None, help='Input image path')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt (overrides --scene)')
    parser.add_argument('--scene', type=str, default=None, help='Scene name from prompts.py')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Negative prompt (overrides --static)')
    
    # Generation configuration
    parser.add_argument('--resolution', type=str, default='480p', choices=['480p', '720p'], help='Output resolution')
    parser.add_argument('--num-frames', type=int, default=93, help='Number of frames to generate')
    parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=4.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fps', type=int, default=15, help='Output video FPS')
    
    # Resample guidance parameters
    parser.add_argument('--guided', action='store_true', help='Enable guided generation')
    parser.add_argument('--resample-steps', type=int, default=3, help='Resample steps per timestep')
    parser.add_argument('--guide-steps', type=int, default=20, help='Number of guide steps')
    parser.add_argument('--resample-round', type=int, default=20, help='Resample rounds')
    parser.add_argument('--omega', type=float, default=1.8, help='Auto-guidance omega')
    parser.add_argument('--omega_resample', type=float, default=1.0, help='Resample omega')
    
    # Mask processing
    parser.add_argument('--soften-mask', action='store_true', help='Enable mask softening')
    parser.add_argument('--transition-distance', type=int, default=15, help='Mask transition distance')
    parser.add_argument('--decay-type', type=str, default='sine', 
                       choices=['linear', 'exponential', 'sine', 'cosine'], help='Mask decay type')
    
    # Flow Latent Filtering (FLF)
    parser.add_argument('--use-pca-channel-selection', action='store_true', 
                       help='Enable optical flow-based channel selection (FLF)')
    parser.add_argument('--static', type=str, choices=['True', 'False'], default='False',
                       help='Static scene mode')
    parser.add_argument('--max-replace', type=int, default=None,
                       help='Max channels to replace (default: 3 for distill, 2 for standard)')
    
    # Output configuration
    parser.add_argument('--output', type=str, default='output_i2v.mp4', help='Output video path')
    parser.add_argument('--save-png', action='store_true', help='Save PNG sequence')
    
    # Upscaling configuration
    parser.add_argument('--enable-upscale', action='store_true', help='Enable upscaling to 720p after generation')
    parser.add_argument('--t-thresh', type=float, default=0.6, 
                       help='Denoising threshold for upscaling (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
