"""
Single-View Warping with VGGT

"""
import torch
import os
import numpy as np
import glob
import cv2
import argparse
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from modules.utils_warp import get_original_image_size, warp_single_img, create_default_crack_params, print_progress_bar


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single-View Warping with VGGT')
    
    # Input/Output
    parser.add_argument('--image_path', type=str, required=True, 
                       help='Path to input image directory')
    parser.add_argument('--output_path', type=str, default='output_warp', 
                       help='Path to output directory')
    
    # Camera selection
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index for single-view warping')
    
    # Warping parameters
    parser.add_argument('--direction', type=str, default='right', 
                       choices=['up', 'down', 'left', 'right', 'forward', 'backward'],
                       help='Camera motion direction: up/down/left/right (orbit around focus point), '
                            'forward/backward (dolly)')
    parser.add_argument('--degree', type=float, default=15.0,
                       help='Rotation angle in degrees (for orbit/pan), or movement percentage (for forward/backward)')
    parser.add_argument('--frame_single', type=int, default=24,
                       help='Number of frames to generate')
    parser.add_argument('--look_at_depth', type=float, default=1.0,
                       help='Focus depth coefficient (1.0=scene mean depth, <1 closer, >1 farther)')
    
    # Depth confidence
    parser.add_argument('--conf_single', type=float, default=1.0,
                       help='Depth confidence percentile threshold (1.0=use all, 0.8=keep top 80%%)')
    
    # Crack filling parameters
    parser.add_argument('--crack_depth_threshold', type=float, default=0.1,
                       help='Depth difference threshold for depth-guided crack filling')
    parser.add_argument('--crack_max_size', type=int, default=6,
                       help='Maximum crack size to fill (pixels)')
    parser.add_argument('--crack_min_neighbors', type=int, default=2,
                       help='Minimum valid neighbors required for crack filling')
    
    # Depth segmentation
    parser.add_argument('--depth_segments', type=int, default=8,
                       help='Number of depth segments for layered crack filling')
    
    # Outlier detection
    parser.add_argument('--outlier_min_neighbors', type=int, default=10,
                       help='Minimum neighbors to keep a pixel (fewer = outlier)')
    parser.add_argument('--outlier_neighbor_radius', type=int, default=3,
                       help='Search radius for counting neighbors')
    
    return parser.parse_args()


def validate_args(args):
    """Validate and process arguments"""
    args.conf_single = max(0.0, min(1.0, args.conf_single))
    
    # Fixed internal defaults
    args.fill_cracks = True
    args.disable_depth_aware_fill = False
    args.skip_outlier_detection = False
    args.use_fast_outlier_detection = True
    
    return args


def load_images_from_path(image_path):
    """Load image files from specified path"""
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.png']
    image_names = []
    
    if os.path.exists(image_path) and os.path.isdir(image_path):
        for ext in image_extensions:
            image_names.extend(glob.glob(os.path.join(image_path, f"*{ext}")))
        
        if image_names:
            image_names.sort()
            print(f"Found {len(image_names)} images")
        else:
            raise ValueError(f"No images found in {image_path}")
    else:
        raise ValueError(f"Directory not found: {image_path}")
    
    return image_names


def save_warped_results(warped_images, warped_masks, camera_interpolations, 
                       output_folder, camera_idx, args):
    """Save warped images and masks"""
    output_img_folder = os.path.join(output_folder, "warped_images")
    os.makedirs(output_img_folder, exist_ok=True)
    
    # Save camera info
    info_file = os.path.join(output_folder, "camera_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Single-View Warping Info:\n")
        f.write(f"Camera Index: {camera_idx}\n")
        f.write(f"Direction: {args.direction}\n")
        f.write(f"Angle: {args.degree} degrees\n")
        f.write(f"Frames: {args.frame_single}\n")
        f.write(f"Confidence Threshold: {args.conf_single}\n")
        f.write(f"Crack Filling: {'Enabled' if args.fill_cracks else 'Disabled'}\n")
        f.write("\nFrame List:\n")
        for i, info in enumerate(camera_interpolations):
            f.write(f"{i}: {info['type']}")
            if info['type'] == 'single_view_warped':
                f.write(f" - {info['direction']} {info['angle']:.2f}°")
            f.write("\n")
    
    # Save images and masks
    for i, (img, mask, info) in enumerate(zip(warped_images, warped_masks, camera_interpolations)):
        # Generate filename
        if info["type"] == "original":
            direction = info.get('direction', 'original')
            img_filename = f"warp_cam{camera_idx}_{direction}_00.00_deg.png"
            mask_filename = f"mask_cam{camera_idx}_{direction}_00.00_deg.png"
        else:
            img_filename = f"warp_cam{camera_idx}_{info['direction']}_{info['angle']:05.2f}_deg.png"
            mask_filename = f"mask_cam{camera_idx}_{info['direction']}_{info['angle']:05.2f}_deg.png"
        
        img_path = os.path.join(output_img_folder, img_filename)
        mask_path = os.path.join(output_img_folder, mask_filename)
        
        # Save image
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
            
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Save mask
        if isinstance(mask, np.ndarray):
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = (mask.astype(np.float32) * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask)
    
    print(f"Saved {len(warped_images)} warped views to: {output_img_folder}")
    return output_img_folder


def create_video(warped_images, warped_masks, output_folder):
    """Create video from warped images"""
    if len(warped_images) < 3:
        return
    
    try:
        video_path = os.path.join(output_folder, "warping_video.mp4")
        
        h, w = warped_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15
        
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        for img, mask in zip(warped_images, warped_masks):
            # Convert image to BGR
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
            
            video_writer.write(img)
        
        video_writer.release()
        print(f"Videos saved: {video_path}")
    except Exception as e:
        print(f"Video creation failed: {e}")


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    args = validate_args(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    
    image_names = load_images_from_path(args.image_path)
    
    if args.camera >= len(image_names):
        print(f"Warning: camera index {args.camera} out of range, using 0")
        args.camera = 0
    
    images = load_and_preprocess_images(image_names).to(device)
    
    print("Running model inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        
        # Get camera parameters
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        
        # Get depth map
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    
    # Extract camera parameters
    camera_idx = args.camera
    extrinsic_squeezed = extrinsic.squeeze(0) if extrinsic.dim() > 3 else extrinsic
    intrinsic_squeezed = intrinsic.squeeze(0) if intrinsic.dim() > 3 else intrinsic
    
    camera_extrinsic = extrinsic_squeezed[camera_idx].cpu().numpy()
    camera_intrinsic = intrinsic_squeezed[camera_idx].cpu().numpy()
    
    # Load original image
    original_image_path = image_names[camera_idx]
    orig_width, orig_height = get_original_image_size(original_image_path)
    
    try:
        pil_image = Image.open(original_image_path)
        original_image = np.array(pil_image)
        
        if original_image.shape[-1] == 4:
            original_image = original_image[..., :3]
        
        if original_image.dtype == np.uint8:
            original_image = original_image.astype(np.float32) / 255.0
        
        original_image_tensor = torch.from_numpy(original_image.transpose(2, 0, 1)).float()
    except Exception as e:
        print(f"Failed to load original image: {e}")
        original_image_tensor = images[0, camera_idx].clone()
    
    # Get depth for selected camera
    model_height, model_width = images.shape[-2], images.shape[-1]
    camera_depth = depth_map.squeeze(0)[camera_idx].cpu().numpy()
    camera_depth_conf = None
    if depth_conf is not None:
        camera_depth_conf = depth_conf.squeeze(0)[camera_idx].cpu().numpy()
    
    if camera_depth.ndim > 2:
        camera_depth = camera_depth.squeeze()
    
    # Resize depth to original image size
    resized_depth = cv2.resize(camera_depth, (orig_width, orig_height), 
                               interpolation=cv2.INTER_LINEAR)
    
    resized_depth_conf = None
    if camera_depth_conf is not None:
        if camera_depth_conf.ndim > 2:
            camera_depth_conf = camera_depth_conf.squeeze()
        resized_depth_conf = cv2.resize(camera_depth_conf, (orig_width, orig_height),
                                       interpolation=cv2.INTER_LINEAR)
    
    # Adjust intrinsics for original image size
    width_scale = orig_width / model_width
    height_scale = orig_height / model_height
    
    adjusted_intrinsic = camera_intrinsic.copy()
    adjusted_intrinsic[0, 0] *= width_scale  # fx
    adjusted_intrinsic[0, 2] *= width_scale  # cx
    adjusted_intrinsic[1, 1] *= height_scale  # fy
    adjusted_intrinsic[1, 2] *= height_scale  # cy
    
    # Prepare crack filling parameters
    crack_params = create_default_crack_params(args)
    
    # Perform single-view warping
    print(f"Warping camera {camera_idx} in {args.direction} direction...")
    warped_images, warped_masks, camera_interpolations = warp_single_img(
        extrinsic=camera_extrinsic,
        intrinsic=adjusted_intrinsic,
        image=original_image_tensor,
        depth_map=resized_depth,
        depth_conf=resized_depth_conf,
        direction=args.direction,
        degree=args.degree,
        conf_threshold=args.conf_single,
        frame_num=args.frame_single,
        use_backward=False,
        fill_cracks=args.fill_cracks,
        crack_params=crack_params,
        args=args
    )
    
    # Save results
    save_warped_results(warped_images, warped_masks, camera_interpolations,
                       args.output_path, camera_idx, args)
    
    # Create video
    create_video(warped_images, warped_masks, args.output_path)
    
    print(f"\nComplete! Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()

