import gc
import os
import sys
import numpy as np
import torch
import cv2
import glob
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import resize
from diffusers.training_utils import set_seed
from diffusers.utils import export_to_video

# Add DepthCrafter module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DepthCrafter'))

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import read_video_frames

from utils import (
    get_look_up_camera_seq, get_look_right_camera_seq, 
    apply_zoom_to_camera_seq, get_stable_look_up_camera_seq, 
    get_stable_look_right_camera_seq, apply_stable_zoom_to_camera_seq,
    project_points_to_image_pytorch, project_points_to_image_pytorch_with_edge_filter
)


def run_depth_estimation(video_path, save_folder, unet_path, pre_train_path, 
                        num_inference_steps, guidance_scale, window_size, 
                        overlap, max_res, process_length, target_fps, seed, cpu_offload):
    """
    Run DepthCrafter depth estimation on input video/image sequence
    """
    print("=" * 60)
    print("Stage 1: Depth Estimation")
    print("=" * 60)
    
    # Initialize DepthCrafter pipeline
    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
        unet_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    pipe = DepthCrafterPipeline.from_pretrained(
        pre_train_path,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    
    # CPU offload configuration
    if cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif cpu_offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    
    # Enable memory optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_attention_slicing()
    
    set_seed(seed)
    
    # Read input frames
    frames, target_fps = read_video_frames(
        video_path,
        process_length,
        target_fps,
        max_res,
        "open",
    )
    
    # Run depth inference
    print(f"Processing {len(frames)} frames...")
    with torch.inference_mode():
        res = pipe(
            frames,
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            window_size=window_size,
            overlap=overlap,
            track_time=False,
        ).frames[0]
    
    # Convert to single channel depth map and normalize
    res = res.sum(-1) / res.shape[-1]
    res = (res - res.min()) / (res.max() - res.min())
    
    # Determine input type and save path
    is_image_sequence = os.path.isdir(video_path)
    
    if is_image_sequence:
        save_path = os.path.join(save_folder, os.path.basename(video_path.rstrip('/')))
    else:
        save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(video_path))[0])
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save depth as npz
    depth_npz_path = os.path.join(save_path, "depth.npz")
    np.savez_compressed(depth_npz_path, depth=res)
    print(f"Depth map saved to: {depth_npz_path}")
    
    # Save depth as RGB visualization
    depth_rgb_folder = os.path.join(save_path, "depth_RGB")
    os.makedirs(depth_rgb_folder, exist_ok=True)
    
    print(f"Saving RGB depth visualizations to: {depth_rgb_folder}")
    for i, depth_frame in enumerate(res):
        # Normalize depth to 0-255 range
        depth_normalized = (depth_frame * 255).astype(np.uint8)
        # Apply colormap (turbo/jet-like colormap for better visualization)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        # Convert BGR to RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        # Save as PNG
        depth_rgb_path = os.path.join(depth_rgb_folder, f"depth_{i:04d}.png")
        cv2.imwrite(depth_rgb_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
    
    print(f"Saved {len(res)} RGB depth frames")
    
    # Clean up
    del pipe, unet
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path, depth_npz_path, frames


def run_warping(video_path, depth_folder, output_path, direction, degree, look_at_depth,
                zoom, rate, stable, stable_frame, enable_edge_filter, edge_threshold, edge_dilation,
                depth_jump_threshold, neighbor_check_radius):
    """
    Apply 3D warping to video using depth estimation results
    """
    print("\n" + "=" * 60)
    print("Stage 2: 3D Warping")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(f'{output_path}/imgs', exist_ok=True)
    
    # Load depth map
    depth_files = glob.glob(f'{depth_folder}/*.npz')
    if not depth_files:
        raise FileNotFoundError(f"No depth file found in {depth_folder}")
    
    depth_path = depth_files[0]
    print(f"Loading depth from: {depth_path}")
    depth = np.load(depth_path)['depth']
    depth = torch.from_numpy(depth).to(device)
    
    # Load input frames
    if os.path.isdir(video_path):
        # Image sequence input
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(video_path, ext)))
            image_files.extend(glob.glob(os.path.join(video_path, ext.upper())))
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {video_path}")
        
        image_files.sort()
        frames = []
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                pil_image = Image.open(img_path)
                frame = np.array(pil_image)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frames = np.array(frames)
    else:
        # Video file input
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
    
    frames = torch.from_numpy(frames).to(device) / 255.0
    frames = frames.permute(0, 3, 1, 2)
    frames = resize(frames, (depth.shape[-2], depth.shape[-1]))
    
    # Camera intrinsics
    H, W = depth.shape[-2], depth.shape[-1]
    K = torch.tensor([[525.0, 0, 0.5 * W],
                      [0, 525.0, 0.5 * H],
                      [0, 0, 1]]).to(device)
    
    extrinsics = np.eye(4)
    look_at_depth_value = np.median((1.0 / (depth[0] + 0.1)).cpu().numpy()) * look_at_depth
    frame_num = frames.shape[0]
    
    # Generate camera sequence
    if stable:
        print(f"Stable warping mode: {direction} {degree}° over first {stable_frame} frames")
        if zoom != 'none':
            print(f"  + {zoom} with rate {rate}")
        
        if direction == 'up':
            cams = get_stable_look_up_camera_seq(extrinsics, degree, frame_num, look_at_depth_value, stable_frame)
        elif direction == 'down':
            cams = get_stable_look_up_camera_seq(extrinsics, -degree, frame_num, look_at_depth_value, stable_frame)
        elif direction == 'right':
            cams = get_stable_look_right_camera_seq(extrinsics, degree, frame_num, look_at_depth_value, stable_frame)
        elif direction == 'left':
            cams = get_stable_look_right_camera_seq(extrinsics, -degree, frame_num, look_at_depth_value, stable_frame)
        
        if zoom != 'none':
            cams = apply_stable_zoom_to_camera_seq(cams, zoom, rate, look_at_depth_value, stable_frame)
    else:
        print(f"Standard warping mode: {direction} {degree}°")
        if zoom != 'none':
            print(f"  + {zoom} with rate {rate}")
        
        if direction == 'up':
            cams = get_look_up_camera_seq(extrinsics, degree, frame_num, look_at_depth_value)
        elif direction == 'down':
            cams = get_look_up_camera_seq(extrinsics, -degree, frame_num, look_at_depth_value)
        elif direction == 'right':
            cams = get_look_right_camera_seq(extrinsics, degree, frame_num, look_at_depth_value)
        elif direction == 'left':
            cams = get_look_right_camera_seq(extrinsics, -degree, frame_num, look_at_depth_value)
        
        if zoom != 'none':
            cams = apply_zoom_to_camera_seq(cams, zoom, rate, look_at_depth_value)
    
    if enable_edge_filter:
        print(f"Edge filtering enabled: threshold={edge_threshold}, dilation={edge_dilation}")
    
    # Render warped frames
    rendered_frames = []
    masks = []
    
    print(f"Warping {frame_num} frames...")
    for idx, cam in tqdm(enumerate(cams), total=frame_num):
        rgb = frames[idx].permute(1, 2, 0).reshape(-1, 3)
        depth_frame = 1.0 / (depth[idx] + 0.1)
        
        # Generate point cloud
        ii, jj = torch.from_numpy(np.indices((H, W))).to(device)
        X_cam = (jj - K[0, 2]) * depth_frame / K[0, 0]
        Y_cam = (ii - K[1, 2]) * depth_frame / K[1, 1]
        Z_cam = depth_frame
        point_cloud = torch.stack((X_cam, Y_cam, Z_cam), axis=-1).reshape(-1, 3)
        
        # Project with optional edge filtering (skip for first frame)
        if enable_edge_filter and idx > 0:
            rendered_image, mask = project_points_to_image_pytorch_with_edge_filter(
                point_cloud, rgb, torch.from_numpy(cam).to(device),
                K, torch.tensor([H, W], device=device),
                depth_2d=depth_frame,
                enable_edge_filter=True,
                edge_threshold=edge_threshold,
                edge_dilation=edge_dilation,
                depth_jump_threshold=depth_jump_threshold,
                neighbor_check_radius=neighbor_check_radius,
                morph=True,
                verbose=False
            )
        else:
            rendered_image, mask = project_points_to_image_pytorch(
                point_cloud, rgb, torch.from_numpy(cam).to(device),
                K, torch.tensor([H, W], device=device),
                morph=True
            )
        
        rendered_frames.append(rendered_image)
        masks.append(mask)
        
        cv2.imwrite(f'{output_path}/imgs/rendered_image_{idx:02d}.png',
                    cv2.cvtColor(rendered_image * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'{output_path}/imgs/mask_{idx:02d}.png', mask * 255)
    
    # Export videos
    export_to_video(rendered_frames, f'{output_path}/video.mp4', fps=6)
    export_to_video(masks, f'{output_path}/mask.mp4', fps=6)
    
    print(f"\nWarping completed!")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='DepthCrafter + 3D Warping Pipeline')
    
    # Input/Output paths
    parser.add_argument('--video_path', type=str, required=True,
                       help='Input video file or image sequence directory')
    parser.add_argument('--output_path', type=str, default='output',
                       help='Output folder path')
    
    # Depth estimation parameters
    parser.add_argument('--unet_path', type=str, default='tencent/DepthCrafter',
                       help='DepthCrafter UNet model path')
    parser.add_argument('--pre_train_path', type=str, 
                       default='stabilityai/stable-video-diffusion-img2vid-xt',
                       help='Pre-trained model path')
    parser.add_argument('--num_inference_steps', type=int, default=5,
                       help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                       help='Guidance scale for inference')
    parser.add_argument('--window_size', type=int, default=110,
                       help='Sliding window size')
    parser.add_argument('--overlap', type=int, default=25,
                       help='Overlap between windows')
    parser.add_argument('--max_res', type=int, default=1024,
                       help='Maximum resolution')
    parser.add_argument('--process_length', type=int, default=-1,
                       help='Process length (-1 for full video)')
    parser.add_argument('--target_fps', type=int, default=-1,
                       help='Target FPS (-1 for original)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cpu_offload', type=str, default='model',
                       choices=['model', 'sequential', 'none'],
                       help='CPU offload mode')
    
    # Warping parameters
    parser.add_argument('--direction', type=str, choices=['up', 'down', 'left', 'right'],
                       default='left', help='Warping direction')
    parser.add_argument('--degree', type=float, default=15,
                       help='Warping angle in degrees')
    parser.add_argument('--look_at_depth', type=float, default=1.0,
                       help='Look at depth multiplier')
    
    # Zoom mode
    parser.add_argument('--zoom', type=str, choices=['zoom_in', 'zoom_out', 'none'],
                       default='none', help='Zoom mode')
    parser.add_argument('--rate', type=float, default=0.8,
                       help='Zoom rate (0.0-1.0)')
    
    # Stable warping mode
    parser.add_argument('--stable', action='store_true',
                       help='Enable stable warping (complete transformation in first N frames)')
    parser.add_argument('--stable_frame', type=int, default=17,
                       help='Number of frames to complete stable transformation (default: 17)')
    
    # Edge filtering
    parser.add_argument('--enable_edge_filter', action='store_true',
                       help='Enable depth edge filtering to reduce warping artifacts')
    parser.add_argument('--edge_threshold', type=float, default=0.1,
                       help='Edge detection threshold (0.0-1.0)')
    parser.add_argument('--edge_dilation', type=int, default=3,
                       help='Edge region dilation in pixels')
    parser.add_argument('--depth_jump_threshold', type=float, default=0.3,
                       help='Depth discontinuity threshold')
    parser.add_argument('--neighbor_check_radius', type=int, default=2,
                       help='Neighbor depth check radius')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.zoom != 'none' and not (0.0 < args.rate <= 1.0):
        raise ValueError(f"Rate must be between 0.0 and 1.0, got {args.rate}")
    
    # Determine depth folder path
    is_image_sequence = os.path.isdir(args.video_path)
    if is_image_sequence:
        video_name = os.path.basename(args.video_path.rstrip('/'))
    else:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    depth_folder = os.path.join(args.output_path, 'depth_estimation', video_name)
    depth_npz_path = os.path.join(depth_folder, 'depth.npz')
    
    # Stage 1: Depth Estimation (skip if depth.npz already exists)
    if os.path.exists(depth_npz_path):
        print("=" * 60)
        print("Stage 1: Depth Estimation - SKIPPED")
        print("=" * 60)
        print(f"Found existing depth file: {depth_npz_path}")
        print("Skipping depth estimation, using existing depth data...\n")
    else:
        depth_folder, depth_npz_path, _ = run_depth_estimation(
            video_path=args.video_path,
            save_folder=os.path.join(args.output_path, 'depth_estimation'),
            unet_path=args.unet_path,
            pre_train_path=args.pre_train_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            window_size=args.window_size,
            overlap=args.overlap,
            max_res=args.max_res,
            process_length=args.process_length,
            target_fps=args.target_fps,
            seed=args.seed,
            cpu_offload=args.cpu_offload
        )
    
    # Stage 2: Warping
    warping_output_path = os.path.join(args.output_path, 'warped')
    run_warping(
        video_path=args.video_path,
        depth_folder=depth_folder,
        output_path=warping_output_path,
        direction=args.direction,
        degree=args.degree,
        look_at_depth=args.look_at_depth,
        zoom=args.zoom,
        rate=args.rate,
        stable=args.stable,
        stable_frame=args.stable_frame,
        enable_edge_filter=args.enable_edge_filter,
        edge_threshold=args.edge_threshold,
        edge_dilation=args.edge_dilation,
        depth_jump_threshold=args.depth_jump_threshold,
        neighbor_check_radius=args.neighbor_check_radius
    )
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

