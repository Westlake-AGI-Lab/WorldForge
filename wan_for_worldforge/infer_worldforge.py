import torch
import numpy as np
import argparse
import os
import glob
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image
from scipy.ndimage import distance_transform_edt

# Import custom pipeline, scheduler and prompts from utils folder
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from pipeline_wan_i2v_clean import WanImageToVideoPipeline
from scheduling_unipc_multistep_clean import UniPCMultistepScheduler
from prompts import get_prompt, list_available_scenes

parser = argparse.ArgumentParser(description='Run Wan2.1 model inference with optical flow guided generation')
parser.add_argument('--model', type=str, choices=['480p', '720p'], default='720p',
                    help='Model version: 480p or 720p (default: 720p)')
parser.add_argument('--models-dir', type=str, required=True,
                    help='Base directory for model weights')
parser.add_argument('--output', type=str, default='output.mp4',
                    help='Output video filename (default: output.mp4)')
parser.add_argument('--image', type=str, default=None,
                    help='Input image path or URL. If not provided, use first frame of video')
parser.add_argument('--video-ref', type=str, required=True,
                    help='Video frames directory path for guiding generation process. Images with mask_ prefix are used as masks')
parser.add_argument('--guided', action='store_true',
                    help='Enable guided generation')
parser.add_argument('--resample-steps', type=int, default=3,
                    help='Resampling steps (default: 3)')
parser.add_argument('--guide-steps', type=int, default=20,
                    help='Guidance steps (default: 20)')
parser.add_argument('--omega', type=float, default=1.8,
                    help='Auto-guidance parameter (default: 1.8)')
parser.add_argument('--omega_resample', type=float, default=1.0,
                    help='Resampling auto-guidance parameter (default: 1.0)')
parser.add_argument('--num-frames', type=int, default=25,
                    help='Number of frames to generate (default: 25)')
parser.add_argument('--num-inference-steps', type=int, default=50,
                    help='Number of inference steps (default: 50)')
parser.add_argument('--guidance-scale', type=float, default=5.0,
                    help='Classifier-free guidance weight (default: 5.0)')
parser.add_argument('--resample-round', type=int, default=20,
                    help='Resampling rounds (default: 20)')
parser.add_argument('--static', type=str, choices=['True', 'False'], default='False',
                    help='Use static negative prompt (True/False, default: False)')
parser.add_argument('--scene', type=str, default='horn',
                    help='Scene type (horn/man/fern/girl/hike/train, default: horn)')
parser.add_argument('--use-pca-channel-selection', action='store_true',
                    help='Enable PCA-based intelligent channel selection with optical flow (FLF)')
parser.add_argument('--soften-mask', action='store_true',
                    help='Enable mask softening for smooth transitions at 1-0 boundaries')
parser.add_argument('--transition-distance', type=int, default=15,
                    help='Transition region distance in pixels (default: 15)')
parser.add_argument('--decay-type', type=str, choices=['linear', 'exponential', 'sine', 'cosine'], default='sine',
                    help='Decay type: linear, exponential, sine, or cosine (default: sine)')
parser.add_argument('--save-png', action='store_true',
                    help='Save output frames as PNG files to avoid video compression artifacts')
args = parser.parse_args()


def read_frames_from_directory(directory):
    """Read all image frames and masks from directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    all_files = []
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        all_files.extend(glob.glob(pattern))
    
    all_files = sorted(all_files)
    
    if not all_files:
        raise ValueError(f"No image files found in directory {directory}")
    
    frame_files = []
    mask_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename.startswith('mask_'):
            mask_files.append(file_path)
        else:
            frame_files.append(file_path)
    
    frames = [Image.open(f).convert('RGB') for f in frame_files]
    masks = [Image.open(f).convert('L') for f in mask_files]
    
    if not masks and frames:
        masks = [Image.new('L', frames[0].size, 0) for _ in frames]
    
    if len(masks) != len(frames):
        while len(masks) < len(frames):
            masks.append(masks[-1] if masks else Image.new('L', frames[0].size, 0))
        if len(masks) > len(frames):
            masks = masks[:len(frames)]
    
    first_frame = frames[0] if frames else None
    return frames, masks, first_frame


def soften_mask(mask_array, transition_distance=15, decay_type='sine'):
    """
    Soften binary mask by creating smooth transitions at 1-0 boundaries.
    
    Args:
        mask_array: numpy array of shape [F, H, W], values 0 or 1
        transition_distance: transition region distance in pixels (default: 15)
        decay_type: decay type, supports 'linear', 'exponential', 'sine', 'cosine'
    
    Returns:
        numpy array: softened mask with values between 0-1
    """
    softened_mask = mask_array.copy().astype(np.float32)
    
    for frame_idx in range(mask_array.shape[0]):
        current_mask = mask_array[frame_idx].astype(bool)
        
        if np.all(current_mask) or np.all(~current_mask):
            continue
        
        softened_frame = mask_array[frame_idx].copy().astype(np.float32)
        distance_from_ones = distance_transform_edt(current_mask)
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


# Read video frames and masks
print("=" * 80)
print("Loading reference video frames and masks...")
frames, masks, first_frame = read_frames_from_directory(args.video_ref)
print(f"Loaded {len(frames)} frames and {len(masks)} masks")

# Determine model path
if args.model == '480p':
    model_folder_name = "wan2.1_480p_model_local"
    max_area = 480 * 832
else:
    model_folder_name = "wan2.1_720p_model_local"
    max_area = 720 * 1280

model_path = os.path.join(args.models_dir, model_folder_name)

if not os.path.exists(model_path):
    raise ValueError(f"Model path does not exist: {model_path}")

print(f"Loading model from: {model_path}")

# Load model components (suppress harmless LOAD REPORT from transformers 5.x)
import transformers as _tf; _prev_verbosity = _tf.logging.get_verbosity(); _tf.logging.set_verbosity_error()
image_encoder_path = os.path.join(model_path, "image_encoder")
vae_path = os.path.join(model_path, "vae")

image_encoder = CLIPVisionModel.from_pretrained(
    image_encoder_path, 
    torch_dtype=torch.float32, 
    local_files_only=True
)

vae = AutoencoderKLWan.from_pretrained(
    vae_path, 
    torch_dtype=torch.float32, 
    local_files_only=True
)

pipe = WanImageToVideoPipeline.from_pretrained(
    model_path, 
    vae=vae, 
    image_encoder=image_encoder, 
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
_tf.logging.set_verbosity(_prev_verbosity)

# Replace with clean version of scheduler
scheduler_config = pipe.scheduler.config
pipe.scheduler = UniPCMultistepScheduler.from_config(scheduler_config)

pipe.to("cuda")
# pipe.set_progress_bar_config(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

print("Model loaded successfully")

# Load input image
if args.image is not None:
    image = load_image(args.image)
else:
    image = first_frame
    if image is None:
        raise ValueError("Cannot get first frame as input image, please specify --image parameter")

# Calculate appropriate dimensions
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

# Process video frames
video_ref = None
if frames:
    try:
        resized_frames = [frame.resize((width, height)) for frame in frames]
        video_frames = torch.stack([
            torch.tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0
            for frame in resized_frames
        ])
        video_ref = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    except Exception as e:
        print(f"Error processing video frames: {e}")

# Process masks
mask = None
if masks:
    try:
        resized_masks = [mask.resize((width, height)) for mask in masks]
        mask_array = np.stack([np.array(mask) / 255.0 for mask in resized_masks])
        
        if args.soften_mask:
            mask_array = soften_mask(mask_array, args.transition_distance, args.decay_type)
        
        mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
        
        if mask.shape[1] != 1:
            raise ValueError(f"Mask should be single-channel, but has {mask.shape[1]} channels")
            
    except Exception as e:
        print(f"Error processing masks: {e}")
        mask = None

# Set prompt
prompt = get_prompt(args.scene)

print("=" * 80)
print("INFERENCE CONFIGURATION:")
print(f"  Scene:                     {args.scene}")
print(f"  Model:                     {args.model}")
print(f"  Num Frames:                {args.num_frames}")
print(f"  Inference Steps:           {args.num_inference_steps}")
print(f"  Guided Generation:         {args.guided}")
if args.guided:
    print(f"  Guide Steps:               {args.guide_steps}")
    print(f"  Resample Rounds:           {args.resample_round}")
    print(f"  FLF Channel Selection:     {args.use_pca_channel_selection}")
print(f"  Soften Mask:               {args.soften_mask}")
print(f"  Static Mode:               {args.static}")
print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
print("=" * 80)

# Define negative prompts
negative_prompt_static = "Blink, twinkle, waggle, speak, wind, windy, leaves shaking, leaves tremble, sighboard, background dynamics, dynamic imagery, gray sky, hazy sky, overcast, gloomy sky, dim, murky, smoggy, shake, object motion blur, streaking objects, object jitter, camera shake, time flow, illogical composition, bright tones, overexposed, blurred details, subtitles, text, logo, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, sudden scene shift, incoherent scene jump, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, any movement, character motion, slight object movement, object swaying, character micro-movements, subtle object rotation, object vibration, messy background, three legs, many people in the background, walking, scene changes, visual detail movement, object disintegration, object breakage."
negative_prompt_dynamic = "Streaking objects, mosaic, grainy, pixelated, noise, flickering, cropped, glitch, fragmented, broken, artifacts, chromatic aberration, micro camera shake, grid, tiling, blurry, camera shake, sudden scene shift, incoherent scene jump, sudden object appearance, blinking, object jitter, camera shake, illogical composition, bright tones, overexposed, blurred details, subtitles, overall gray, solid color, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, messy background, three legs, many people in the background, walking backwards"

if args.static == 'True':
    negative_prompt = negative_prompt_static
    static_bool = True
else:
    negative_prompt = negative_prompt_dynamic
    static_bool = False

generator = torch.manual_seed(42)

print("\nStarting video generation...")

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height, width=width,
    num_frames=args.num_frames,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    generator=generator,
    video_ref=video_ref,
    mask=mask,
    guided=args.guided,
    resample_steps=args.resample_steps,
    guide_steps=args.guide_steps,
    omega=args.omega,
    omega_resample=args.omega_resample,
    resample_round=args.resample_round,
    use_pca_channel_selection=args.use_pca_channel_selection,
    static=static_bool,
).frames[0]

# Save output video
output_path = args.output
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

export_to_video(output, output_path, fps=16)
print("=" * 80)
print(f"Video generation completed!")
print(f"Output saved to: {output_path}")

# Save frames as PNG if requested
if args.save_png:
    png_dir = os.path.splitext(output_path)[0] + "_frames"
    os.makedirs(png_dir, exist_ok=True)
    
    for i, frame in enumerate(output):
        if isinstance(frame, Image.Image):
            frame_img = frame
        else:
            frame_arr = np.array(frame)
            if frame_arr.dtype in [np.float32, np.float64]:
                frame_arr = (frame_arr * 255).clip(0, 255).astype(np.uint8)
            frame_img = Image.fromarray(frame_arr)
        
        png_path = os.path.join(png_dir, f"frame_{i:04d}.png")
        frame_img.save(png_path, format='PNG')
    
    print(f"PNG frames saved to: {png_dir}/ ({len(output)} frames)")

print("=" * 80)
