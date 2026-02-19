"""
Single-View Warping Utilities for VGGT

Core functionality for generating novel views from single images using depth-guided warping.
"""

import numpy as np
import torch
import cv2
from PIL import Image
import scipy.ndimage as ndimage
import sys


# ============================================================================
# Progress Bar Utilities
# ============================================================================

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Simple progress bar for terminal output
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()


# ============================================================================
# Image Utilities
# ============================================================================

def get_original_image_size(image_path):
    """Get original image dimensions"""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return 518, 392


def ensure_cpu(tensor):
    """Ensure tensor is on CPU"""
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        return tensor.cpu()
    return tensor


# ============================================================================
# Camera Generation Functions
# ============================================================================

def get_look_up_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate up/down camera sequence"""
    camera_list = []
    deg_list = np.linspace(0, max_degree, frame_num)
    
    for deg in deg_list:
        rad = np.deg2rad(deg)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]
        ])
        
        new_cam = extrinsic.copy()
        R, t = new_cam[:3, :3], new_cam[:3, 3]
        cam_pos = -R.T @ t
        look_at_pos = cam_pos + R.T @ np.array([0, 0, look_at_depth])
        
        cam_to_look = look_at_pos - cam_pos
        rotated_cam_to_look = rot_x @ cam_to_look
        new_cam_pos = look_at_pos - rotated_cam_to_look
        
        z_axis = (look_at_pos - new_cam_pos)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        y_axis_init = R.T @ np.array([0, 1, 0])
        y_axis = y_axis_init - np.dot(y_axis_init, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        new_R = np.column_stack([x_axis, y_axis, z_axis]).T
        new_t = -new_R @ new_cam_pos
        
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


def get_look_right_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate left/right camera sequence"""
    camera_list = []
    deg_list = np.linspace(0, max_degree, frame_num)
    
    for deg in deg_list:
        rad = np.deg2rad(deg)
        rot_y = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
        
        new_cam = extrinsic.copy()
        R, t = new_cam[:3, :3], new_cam[:3, 3]
        cam_pos = -R.T @ t
        look_at_pos = cam_pos + R.T @ np.array([0, 0, look_at_depth])
        
        cam_to_look = look_at_pos - cam_pos
        rotated_cam_to_look = rot_y @ cam_to_look
        new_cam_pos = look_at_pos - rotated_cam_to_look
        
        z_axis = (look_at_pos - new_cam_pos)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        y_axis_init = R.T @ np.array([0, 1, 0])
        y_axis = y_axis_init - np.dot(y_axis_init, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        new_R = np.column_stack([x_axis, y_axis, z_axis]).T
        new_t = -new_R @ new_cam_pos
        
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


def get_look_forward_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate forward camera sequence"""
    camera_list = []
    progress_list = np.linspace(0, max_degree / 100.0, frame_num)
    
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    cam_pos = -R.T @ t
    scene_center = cam_pos + R.T @ np.array([0, 0, look_at_depth])
    cam_to_center = scene_center - cam_pos
    radius = np.linalg.norm(cam_to_center)
    forward_direction = cam_to_center / radius
    
    for progress in progress_list:
        forward_distance = radius * progress
        new_cam_pos = cam_pos + forward_direction * forward_distance
        
        new_cam_to_center = scene_center - new_cam_pos
        new_cam_to_center_norm = np.linalg.norm(new_cam_to_center)
        
        if new_cam_to_center_norm > 1e-6:
            z_axis = new_cam_to_center / new_cam_to_center_norm
            y_axis_init = R.T @ np.array([0, 1, 0])
            y_axis = y_axis_init - np.dot(y_axis_init, z_axis) * z_axis
            y_axis_norm = np.linalg.norm(y_axis)
            
            if y_axis_norm > 1e-6:
                y_axis = y_axis / y_axis_norm
            else:
                y_axis = np.array([0, 1, 0]) if abs(z_axis[1]) < 0.9 else np.array([1, 0, 0])
                y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
                y_axis = y_axis / np.linalg.norm(y_axis)
            
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            new_R = np.column_stack([x_axis, y_axis, z_axis]).T
            new_t = -new_R @ new_cam_pos
        else:
            new_R = R.copy()
            new_t = -new_R @ new_cam_pos
        
        new_cam = extrinsic.copy()
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


def get_look_backward_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate backward camera sequence"""
    camera_list = []
    progress_list = np.linspace(0, max_degree / 100.0, frame_num)
    
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    cam_pos = -R.T @ t
    scene_center = cam_pos + R.T @ np.array([0, 0, look_at_depth])
    cam_to_center = scene_center - cam_pos
    radius = np.linalg.norm(cam_to_center)
    backward_direction = -cam_to_center / radius
    
    for progress in progress_list:
        backward_distance = radius * progress
        new_cam_pos = cam_pos + backward_direction * backward_distance
        
        new_cam_to_center = scene_center - new_cam_pos
        new_cam_to_center_norm = np.linalg.norm(new_cam_to_center)
        
        if new_cam_to_center_norm > 1e-6:
            z_axis = new_cam_to_center / new_cam_to_center_norm
            y_axis_init = R.T @ np.array([0, 1, 0])
            y_axis = y_axis_init - np.dot(y_axis_init, z_axis) * z_axis
            y_axis_norm = np.linalg.norm(y_axis)
            
            if y_axis_norm > 1e-6:
                y_axis = y_axis / y_axis_norm
            else:
                y_axis = np.array([0, 1, 0]) if abs(z_axis[1]) < 0.9 else np.array([1, 0, 0])
                y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
                y_axis = y_axis / np.linalg.norm(y_axis)
            
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            new_R = np.column_stack([x_axis, y_axis, z_axis]).T
            new_t = -new_R @ new_cam_pos
        else:
            new_R = R.copy()
            new_t = -new_R @ new_cam_pos
        
        new_cam = extrinsic.copy()
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


def get_right_pan_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate right pan sequence (rotation only, no translation)"""
    camera_list = []
    deg_list = np.linspace(0, max_degree, frame_num)
    
    # Get original camera position (stays fixed)
    original_R = extrinsic[:3, :3]
    original_t = extrinsic[:3, 3]
    original_cam_pos = -original_R.T @ original_t
    
    for deg in deg_list:
        rad = np.deg2rad(deg)
        
        # Pure rotation around Y-axis (yaw rotation)
        rot_y = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
        
        # Apply rotation to original camera orientation
        new_R = original_R @ rot_y
        
        # Keep original camera position
        new_t = -new_R @ original_cam_pos
        
        new_cam = extrinsic.copy()
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


def get_left_pan_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate left pan sequence (rotation only, no translation)"""
    camera_list = []
    deg_list = np.linspace(0, max_degree, frame_num)
    
    # Get original camera position (stays fixed)
    original_R = extrinsic[:3, :3]
    original_t = extrinsic[:3, 3]
    original_cam_pos = -original_R.T @ original_t
    
    for deg in deg_list:
        rad = np.deg2rad(-deg)  # Negative for left
        
        # Pure rotation around Y-axis (yaw rotation)
        rot_y = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
        
        # Apply rotation to original camera orientation
        new_R = original_R @ rot_y
        
        # Keep original camera position
        new_t = -new_R @ original_cam_pos
        
        new_cam = extrinsic.copy()
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


def get_up_pan_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate up pan sequence (rotation only, no translation)"""
    camera_list = []
    deg_list = np.linspace(0, max_degree, frame_num)
    
    # Get original camera position (stays fixed)
    original_R = extrinsic[:3, :3]
    original_t = extrinsic[:3, 3]
    original_cam_pos = -original_R.T @ original_t
    
    for deg in deg_list:
        rad = np.deg2rad(deg)
        
        # Pure rotation around X-axis (pitch rotation)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]
        ])
        
        # Apply rotation to original camera orientation
        new_R = original_R @ rot_x
        
        # Keep original camera position
        new_t = -new_R @ original_cam_pos
        
        new_cam = extrinsic.copy()
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


def get_down_pan_camera_seq(extrinsic, max_degree, frame_num, look_at_depth):
    """Generate down pan sequence (rotation only, no translation)"""
    camera_list = []
    deg_list = np.linspace(0, max_degree, frame_num)
    
    # Get original camera position (stays fixed)
    original_R = extrinsic[:3, :3]
    original_t = extrinsic[:3, 3]
    original_cam_pos = -original_R.T @ original_t
    
    for deg in deg_list:
        rad = np.deg2rad(-deg)  # Negative for down
        
        # Pure rotation around X-axis (pitch rotation)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]
        ])
        
        # Apply rotation to original camera orientation
        new_R = original_R @ rot_x
        
        # Keep original camera position
        new_t = -new_R @ original_cam_pos
        
        new_cam = extrinsic.copy()
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_t
        camera_list.append(new_cam)
    
    return camera_list


# ============================================================================
# Crack Filling & Outlier Removal
# ============================================================================

def fill_small_cracks(warped_image, warped_mask, original_depth, depth_conf=None,
                      depth_threshold=0.1, max_crack_size=5, min_valid_neighbors=3):
    """
    Fill small cracks in warped images.
    
    Uses morphological closing + neighbor averaging, with optional depth-guided
    fill for small connected holes when depth confidence is available.
    """
    filled_image = warped_image.copy()
    filled_mask = warped_mask.copy()
    
    holes = (warped_mask == 0)
    if not np.any(holes):
        return filled_image, filled_mask
    
    H, W = warped_mask.shape
    
    # Step 1: Morphological closing + neighbor average fill
    kernel = np.ones((3, 3), np.uint8)
    closed_mask = cv2.morphologyEx(filled_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    newly_filled = (closed_mask > filled_mask) & (filled_mask == 0)
    
    morph_count = 0
    if np.any(newly_filled):
        kernel_neighbor = np.ones((3, 3), dtype=np.float32)
        kernel_neighbor[1, 1] = 0
        neighbor_counts = cv2.filter2D(filled_mask.astype(np.float32), -1, kernel_neighbor)
        valid_fill_mask = newly_filled & (neighbor_counts >= min_valid_neighbors)
        
        if np.any(valid_fill_mask):
            safe_counts = np.maximum(neighbor_counts, 1e-6)
            mask_bool = filled_mask > 0
            if warped_image.ndim == 3:
                for c in range(warped_image.shape[2]):
                    channel_img = warped_image[:, :, c].astype(np.float32)
                    masked_channel = np.where(mask_bool, channel_img, 0.0)
                    neighbor_sum = cv2.filter2D(masked_channel, -1, kernel_neighbor)
                    filled_image[valid_fill_mask, c] = (neighbor_sum / safe_counts)[valid_fill_mask]
            else:
                masked_img = np.where(mask_bool, warped_image.astype(np.float32), 0.0)
                neighbor_sum = cv2.filter2D(masked_img, -1, kernel_neighbor)
                filled_image[valid_fill_mask] = (neighbor_sum / safe_counts)[valid_fill_mask]
            
            filled_mask[valid_fill_mask] = 1
            morph_count = int(np.sum(valid_fill_mask))
    
    # Step 2: Depth-guided fill for remaining small connected holes
    if depth_conf is not None and morph_count < np.sum(holes) * 0.5:
        current_holes = (filled_mask == 0)
        labeled_holes, num_holes = ndimage.label(current_holes)
        
        for hole_id in range(1, num_holes + 1):
            hole_mask = (labeled_holes == hole_id)
            hole_size = np.sum(hole_mask)
            
            if hole_size <= max_crack_size and hole_size <= 4:
                hole_coords = np.where(hole_mask)
                for k in range(len(hole_coords[0])):
                    y, x = hole_coords[0][k], hole_coords[1][k]
                    y_min, y_max = max(0, y-1), min(H, y+2)
                    x_min, x_max = max(0, x-1), min(W, x+2)
                    
                    neighbor_mask_local = filled_mask[y_min:y_max, x_min:x_max]
                    valid_neighbors = neighbor_mask_local > 0
                    
                    if np.sum(valid_neighbors) >= min_valid_neighbors:
                        neighbor_image = filled_image[y_min:y_max, x_min:x_max]
                        neighbor_depth = original_depth[y_min:y_max, x_min:x_max]
                        center_depth = original_depth[y, x]
                        valid_depths_local = neighbor_depth[valid_neighbors]
                        depth_diff = np.abs(valid_depths_local - center_depth)
                        depth_valid = depth_diff <= depth_threshold
                        
                        if np.sum(depth_valid) >= min_valid_neighbors:
                            valid_colors = neighbor_image[valid_neighbors][depth_valid]
                            filled_image[y, x] = np.mean(valid_colors, axis=0)
                            filled_mask[y, x] = 1
    
    return filled_image, filled_mask


def remove_outliers(warped_image, warped_mask, warped_depth, min_neighbors=4, neighbor_radius=1):
    """
    Remove outlier pixels with insufficient neighbors.
    
    Auto-selects between OpenCV (fast, for large data) and scipy (precise, for small data).
    """
    cleaned_image = warped_image.copy()
    cleaned_mask = warped_mask.copy()
    cleaned_depth = warped_depth.copy()
    
    valid_count = np.sum(warped_mask > 0)
    if valid_count == 0 or valid_count < min_neighbors * 2:
        return cleaned_image, cleaned_mask, cleaned_depth
    
    kernel_size = 2 * neighbor_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel[neighbor_radius, neighbor_radius] = 0
    
    mask_float = (warped_mask > 0).astype(np.float32)
    if valid_count > 5000:
        neighbor_counts = cv2.filter2D(mask_float, -1, kernel)
    else:
        neighbor_counts = ndimage.convolve(mask_float, kernel, mode='constant', cval=0)
    
    valid_mask = warped_mask > 0
    outlier_mask = valid_mask & (neighbor_counts < min_neighbors)
    
    if np.sum(outlier_mask) > 0:
        cleaned_mask[outlier_mask] = 0
        cleaned_image[outlier_mask] = 0
        cleaned_depth[outlier_mask] = np.nan
    
    return cleaned_image, cleaned_mask, cleaned_depth


# ============================================================================
# Depth-Aware Crack Filling
# ============================================================================

def segment_depth_map(depth_map, depth_mask, num_segments=5):
    """Segment depth map into layers"""
    valid_mask = depth_mask > 0
    valid_depths = depth_map[valid_mask]
    
    if len(valid_depths) == 0:
        return [], [], valid_depths
    
    min_depth = np.nanmin(valid_depths)
    max_depth = np.nanmax(valid_depths)
    
    if min_depth == max_depth:
        return [valid_mask], [(min_depth, max_depth)], valid_depths
    
    depth_ranges = []
    segment_indices_list = []
    boundaries = np.linspace(min_depth, max_depth, num_segments + 1)
    
    for i in range(num_segments):
        lower, upper = boundaries[i], boundaries[i + 1]
        depth_ranges.append((lower, upper))
        
        if i == num_segments - 1:
            segment_mask = (depth_map >= lower) & (depth_map <= upper) & valid_mask
        else:
            segment_mask = (depth_map >= lower) & (depth_map < upper) & valid_mask
        
        segment_indices_list.append(segment_mask)
    
    return segment_indices_list, depth_ranges, valid_depths


def vectorized_depth_estimation(warped_depth, newly_filled_mask, kernel_size=3):
    """Vectorized depth estimation for filled pixels"""
    if not np.any(newly_filled_mask):
        return warped_depth.copy()
    
    valid_depth_mask = ~np.isnan(warped_depth)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size//2, kernel_size//2] = 0
    
    depth_for_conv = np.where(valid_depth_mask, warped_depth, 0.0)
    valid_for_conv = valid_depth_mask.astype(np.float32)
    
    depth_sum = cv2.filter2D(depth_for_conv, -1, kernel, borderType=cv2.BORDER_REFLECT)
    valid_count = cv2.filter2D(valid_for_conv, -1, kernel, borderType=cv2.BORDER_REFLECT)
    
    valid_count = np.maximum(valid_count, 1e-6)
    neighbor_avg_depth = depth_sum / valid_count
    
    result_depth = warped_depth.copy()
    result_depth[newly_filled_mask] = neighbor_avg_depth[newly_filled_mask]
    
    return result_depth


def fill_segment_cracks(warped_image, warped_depth, segment_mask, crack_params):
    """Fill cracks in a single depth segment with outlier removal and crack filling."""
    if np.sum(segment_mask) == 0:
        return warped_image.copy(), segment_mask.copy(), warped_depth.copy()
    
    # Step 1: Outlier detection
    skip_outlier = crack_params.get('skip_outlier_detection', False)
    use_fast_outlier = crack_params.get('use_fast_outlier_detection', True)
    
    if skip_outlier:
        cleaned_image = warped_image
        cleaned_mask = segment_mask
        cleaned_depth = warped_depth
    elif use_fast_outlier:
        min_neighbors = crack_params.get('min_neighbors', 4)
        neighbor_radius = crack_params.get('neighbor_radius', 1)
        
        # Kernel does NOT zero the center (consistent with original)
        kernel_size = 2 * neighbor_radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(segment_mask.astype(np.float32), -1, kernel.astype(np.float32))
        outlier_mask = (segment_mask > 0) & (neighbor_count < min_neighbors)
        
        cleaned_mask = segment_mask.copy()
        cleaned_mask[outlier_mask] = 0
        
        # No need to copy image/depth here - fill_small_cracks makes its own copies
        cleaned_image = warped_image
        cleaned_depth = warped_depth
    else:
        cleaned_image, cleaned_mask, cleaned_depth = remove_outliers(
            warped_image=warped_image,
            warped_mask=segment_mask,
            warped_depth=warped_depth,
            min_neighbors=crack_params.get('min_neighbors', 4),
            neighbor_radius=crack_params.get('neighbor_radius', 1)
        )
    
    # Step 2: Check if crack filling is needed (only holes within this segment)
    holes_mask = (cleaned_mask == 0) & (segment_mask > 0)
    if not np.any(holes_mask):
        return cleaned_image, cleaned_mask, cleaned_depth
    
    # Step 3: Fill cracks
    filled_image, filled_mask = fill_small_cracks(
        warped_image=cleaned_image,
        warped_mask=cleaned_mask,
        original_depth=cleaned_depth,
        depth_threshold=crack_params.get('depth_threshold', 0.1),
        max_crack_size=crack_params.get('max_crack_size', 5),
        min_valid_neighbors=crack_params.get('min_valid_neighbors', 3)
    )
    
    # Step 4: Estimate depth for newly filled pixels
    newly_filled = (filled_mask > 0) & (cleaned_mask == 0)
    if np.any(newly_filled):
        filled_depth = vectorized_depth_estimation(cleaned_depth, newly_filled)
    else:
        filled_depth = cleaned_depth
    
    return filled_image, filled_mask, filled_depth


def merge_depth_segments(segment_indices_list, filled_results, image_shape):
    """Merge depth segments with proper z-buffer (far to near)"""
    if not filled_results:
        return None, None, None
    
    H, W, C = image_shape
    
    merged_image = np.zeros((H, W, C), dtype=np.float32)
    merged_mask = np.zeros((H, W), dtype=np.uint8)
    merged_depth = np.full((H, W), np.nan, dtype=np.float32)
    
    # Calculate average depth for each segment and sort far to near
    segment_priorities = []
    for i, (filled_image, filled_mask, filled_depth) in enumerate(filled_results):
        if filled_image is not None and np.any(filled_mask > 0):
            valid_depths = filled_depth[~np.isnan(filled_depth) & (filled_mask > 0)]
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
            else:
                avg_depth = float('inf')
            segment_priorities.append((avg_depth, i, filled_image, filled_mask, filled_depth))
    
    # Sort by depth descending (far to near) - far segments first, near segments overwrite
    segment_priorities.sort(key=lambda x: x[0], reverse=True)
    
    # Merge from far to near
    for avg_depth, seg_idx, filled_image, filled_mask, filled_depth in segment_priorities:
        valid_pixels = (filled_mask > 0) & (~np.isnan(filled_depth))
        
        if np.any(valid_pixels):
            merged_image[valid_pixels] = filled_image[valid_pixels]
            merged_mask[valid_pixels] = filled_mask[valid_pixels]
            merged_depth[valid_pixels] = filled_depth[valid_pixels]
    
    return merged_image, merged_mask, merged_depth


def depth_aware_crack_filling(warped_image, warped_mask, warped_depth, crack_params, num_segments=5):
    """Depth-aware crack filling with layered processing."""
    segment_indices_list, depth_ranges, global_valid_depths = segment_depth_map(
        warped_depth, warped_mask, num_segments
    )
    
    # Fallback to simple fill if segmentation fails
    if not segment_indices_list:
        return fill_small_cracks(
            warped_image, warped_mask, warped_depth,
            depth_threshold=crack_params.get('depth_threshold', 0.1),
            max_crack_size=crack_params.get('max_crack_size', 5),
            min_valid_neighbors=crack_params.get('min_valid_neighbors', 3)
        ) + (warped_depth,)
    
    # Process each depth segment
    filled_results = []
    for segment_mask in segment_indices_list:
        if np.sum(segment_mask) == 0:
            filled_results.append((None, None, None))
            continue
        
        filled_image, filled_mask, filled_depth = fill_segment_cracks(
            warped_image, warped_depth, segment_mask, crack_params
        )
        filled_results.append((filled_image, filled_mask, filled_depth))
    
    # Merge segments
    merged_image, merged_mask, merged_depth = merge_depth_segments(
        segment_indices_list, filled_results, warped_image.shape
    )
    
    if merged_image is None:
        return fill_small_cracks(
            warped_image, warped_mask, warped_depth,
            depth_threshold=crack_params.get('depth_threshold', 0.1),
            max_crack_size=crack_params.get('max_crack_size', 5),
            min_valid_neighbors=crack_params.get('min_valid_neighbors', 3)
        ) + (warped_depth,)
    
    return merged_image, merged_mask, merged_depth


def create_default_crack_params(args=None):
    """Create default crack filling parameters from command-line args."""
    return {
        'depth_threshold': getattr(args, 'crack_depth_threshold', 0.1) if args is not None else 0.1,
        'max_crack_size': getattr(args, 'crack_max_size', 5) if args is not None else 5,
        'min_valid_neighbors': getattr(args, 'crack_min_neighbors', 3) if args is not None else 3,
        'min_neighbors': getattr(args, 'outlier_min_neighbors', 4) if args is not None else 4,
        'neighbor_radius': getattr(args, 'outlier_neighbor_radius', 1) if args is not None else 1,
        'skip_outlier_detection': getattr(args, 'skip_outlier_detection', False) if args is not None else False,
        'use_fast_outlier_detection': getattr(args, 'use_fast_outlier_detection', True) if args is not None else True,
    }


# ============================================================================
# Main Warping Function
# ============================================================================

def warp_single_img(extrinsic, intrinsic, image, depth_map, depth_conf=None, direction="right",
                    degree=15.0, conf_threshold=0.5, frame_num=24, use_backward=False,
                    fill_cracks=True, crack_params=None, args=None):
    """
    Single-view warping with depth-guided novel view synthesis
    
    Args:
        extrinsic: Camera extrinsic matrix [3, 4] or [4, 4]
        intrinsic: Camera intrinsic matrix [3, 3]
        image: Input image [C, H, W]
        depth_map: Depth map [H, W]
        depth_conf: Depth confidence map [H, W], optional
        direction: Motion direction (up/down/left/right/forward/backward/up_pan/down_pan/left_pan/right_pan)
        degree: Motion parameter (angle in degrees or percentage)
        conf_threshold: Depth confidence threshold for filtering
        frame_num: Number of frames to generate
        use_backward: Use backward warping (deprecated, always False)
        fill_cracks: Enable crack filling
        crack_params: Crack filling parameters
        args: Additional arguments
    
    Returns:
        warped_images: List of warped images
        warped_masks: List of validity masks
        camera_interpolations: List of camera info dictionaries
    """
    extrinsic = ensure_cpu(extrinsic)
    intrinsic = ensure_cpu(intrinsic)
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = image[0]
        C, H, W = image.shape
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        H, W, C = image.shape
        img_np = image
    
    # Convert depth map to numpy
    if isinstance(depth_map, torch.Tensor):
        depth_map_np = depth_map.cpu().numpy()
        if depth_map_np.ndim > 2:
            depth_map_np = depth_map_np.squeeze()
    else:
        depth_map_np = depth_map
        if depth_map_np.ndim > 2:
            depth_map_np = depth_map_np.squeeze()
    
    # Filter depth by confidence
    if depth_conf is not None:
        if isinstance(depth_conf, torch.Tensor):
            conf_map = depth_conf.cpu().numpy()
            if conf_map.ndim > 2:
                conf_map = conf_map.squeeze()
        else:
            conf_map = depth_conf
            if conf_map.ndim > 2:
                conf_map = conf_map.squeeze()
        
        if conf_threshold == 1.0:
            mask = ~np.isnan(depth_map_np) & (depth_map_np > 0)
            filtered_depth = depth_map_np.copy()
        else:
            flat_conf = conf_map.flatten()
            threshold = np.percentile(flat_conf, (1-conf_threshold)*100)
            mask = conf_map > threshold
            filtered_depth = depth_map_np.copy()
            filtered_depth[~mask] = np.nan
        
        valid_depths = filtered_depth[mask]
        mean_depth = np.nanmean(valid_depths)
        
        look_at_depth_factor = getattr(args, 'look_at_depth', 1.0) if args is not None else 1.0
        adjusted_depth = mean_depth * look_at_depth_factor
    else:
        mask = ~np.isnan(depth_map_np) & (depth_map_np > 0)
        filtered_depth = depth_map_np.copy()
        filtered_depth[~mask] = np.nan
        
        valid_depths = filtered_depth[mask]
        mean_depth = np.nanmean(valid_depths)
        
        look_at_depth_factor = getattr(args, 'look_at_depth', 1.0) if args is not None else 1.0
        adjusted_depth = mean_depth * look_at_depth_factor
    
    # Ensure extrinsic is 4x4
    if extrinsic.shape[0] == 3 and extrinsic.shape[1] == 4:
        cam_extrinsic = np.eye(4)
        cam_extrinsic[:3, :] = extrinsic
    else:
        cam_extrinsic = extrinsic.copy()
    
    # Generate camera sequence
    dir_lower = direction.lower()
    if dir_lower in ["up", "down"]:
        actual_degree = degree if dir_lower == "up" else -degree
        camera_list = get_look_up_camera_seq(cam_extrinsic, actual_degree, frame_num, adjusted_depth)
    elif dir_lower in ["left", "right"]:
        actual_degree = degree if dir_lower == "right" else -degree
        camera_list = get_look_right_camera_seq(cam_extrinsic, actual_degree, frame_num, adjusted_depth)
    elif dir_lower == "forward":
        camera_list = get_look_forward_camera_seq(cam_extrinsic, degree, frame_num, adjusted_depth)
    elif dir_lower == "backward":
        camera_list = get_look_backward_camera_seq(cam_extrinsic, degree, frame_num, adjusted_depth)
    elif dir_lower == "up_pan":
        camera_list = get_up_pan_camera_seq(cam_extrinsic, degree, frame_num, adjusted_depth)
    elif dir_lower == "down_pan":
        camera_list = get_down_pan_camera_seq(cam_extrinsic, degree, frame_num, adjusted_depth)
    elif dir_lower == "left_pan":
        camera_list = get_left_pan_camera_seq(cam_extrinsic, degree, frame_num, adjusted_depth)
    elif dir_lower == "right_pan":
        camera_list = get_right_pan_camera_seq(cam_extrinsic, degree, frame_num, adjusted_depth)
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    
    # Initialize results
    warped_images = []
    warped_masks = []
    camera_interpolations = []
    
    # Add original image
    original_img = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    original_mask = np.ones((H, W), dtype=np.uint8)
    
    warped_images.append(original_img)
    warped_masks.append(original_mask)
    camera_interpolations.append({
        "type": "original",
        "camera_name": "original",
        "direction": direction,
        "angle": 0.0
    })
    
    # Create pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())], axis=-1)
    
    # Unproject to 3D
    cam_coords = np.linalg.inv(intrinsic) @ pixels.T
    depths = filtered_depth.flatten()
    
    valid_depth_mask = ~np.isnan(depths) & (depths > 0)
    points_3d = np.zeros_like(cam_coords)
    points_3d[:, valid_depth_mask] = cam_coords[:, valid_depth_mask] * depths[valid_depth_mask]
    
    # Transform to world coordinates
    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3]
    R_inv = R.T
    t_inv = -R.T @ t
    
    points_world = R_inv @ points_3d + t_inv[:, np.newaxis]
    
    # Pre-compute crack filling params (avoid re-creating per frame)
    if fill_cracks:
        default_crack_params = create_default_crack_params(args)
        if crack_params is not None:
            default_crack_params.update(crack_params)
        disable_depth_aware_fill = getattr(args, 'disable_depth_aware_fill', False) if args is not None else False
        depth_segments = getattr(args, 'depth_segments', 5) if args is not None else 5
    
    # Pre-flatten coordinate arrays for reuse
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # Generate warped views
    print(f"\nGenerating {len(camera_list)-1} warped views...")
    for i, new_cam in enumerate(camera_list):
        if i == 0:
            continue
        
        # Update progress bar
        print_progress_bar(i, len(camera_list)-1, prefix='Warping:', suffix=f'Frame {i}/{len(camera_list)-1}')
            
        warped_img = np.zeros((H, W, C), dtype=np.float32)
        warped_mask = np.zeros((H, W), dtype=np.float32)
        
        # Forward warping
        new_R = new_cam[:3, :3]
        new_t = new_cam[:3, 3]
        new_points_cam = new_R @ points_world + new_t[:, np.newaxis]
        
        valid_z_mask = np.abs(new_points_cam[2, :]) > 1e-6
        valid_z_mask = valid_z_mask & valid_depth_mask
        
        warped_zbuffer = np.ones((H, W)) * float('inf')
        warped_depth = np.full((H, W), np.nan, dtype=np.float32)
        
        if np.sum(valid_z_mask) > 0:
            warped_pixels = np.zeros((3, new_points_cam.shape[1]))
            warped_pixels[:, valid_z_mask] = intrinsic @ (new_points_cam[:, valid_z_mask] / new_points_cam[2, valid_z_mask])
            warped_pixels = warped_pixels[:2, :].T
            
            warped_u = warped_pixels[:, 0]
            warped_v = warped_pixels[:, 1]
            
            valid_mask = (warped_u >= 0) & (warped_u < W) & (warped_v >= 0) & (warped_v < H) & valid_z_mask
            
            u_nearest = np.round(warped_u[valid_mask]).astype(np.int32)
            v_nearest = np.round(warped_v[valid_mask]).astype(np.int32)
            
            u_nearest = np.clip(u_nearest, 0, W - 1)
            v_nearest = np.clip(v_nearest, 0, H - 1)
            
            valid_x = x_flat[valid_mask]
            valid_y = y_flat[valid_mask]
            valid_z = new_points_cam[2, valid_mask]
            
            colors = img_np[valid_y, valid_x]
            
            # Vectorized z-buffer: sort far-to-near, last write (nearest) wins
            sorted_indices = np.argsort(-valid_z)
            v_sorted = v_nearest[sorted_indices]
            u_sorted = u_nearest[sorted_indices]
            colors_sorted = colors[sorted_indices]
            z_sorted = valid_z[sorted_indices]
            
            warped_img[v_sorted, u_sorted] = colors_sorted
            warped_mask[v_sorted, u_sorted] = 1.0
            warped_zbuffer[v_sorted, u_sorted] = z_sorted
            warped_depth[v_sorted, u_sorted] = z_sorted
        
        binary_mask = (warped_mask > 0).astype(np.uint8)
        
        if warped_img.max() <= 1.0:
            warped_img = (warped_img * 255).astype(np.uint8)
        else:
            warped_img = warped_img.astype(np.uint8)
        
        # Apply crack filling
        if fill_cracks and i > 0:
            warped_img_float = warped_img.astype(np.float32) / 255.0
            
            use_depth_aware_fill = (
                'warped_depth' in locals() and
                warped_depth is not None and
                not disable_depth_aware_fill and
                np.sum(~np.isnan(warped_depth)) > 100
            )
            
            if use_depth_aware_fill:
                filled_img_float, filled_mask, filled_depth = depth_aware_crack_filling(
                    warped_image=warped_img_float,
                    warped_mask=binary_mask,
                    warped_depth=warped_depth,
                    crack_params=default_crack_params,
                    num_segments=depth_segments
                )
                warped_depth = filled_depth
            else:
                filled_img_float, filled_mask = fill_small_cracks(
                    warped_image=warped_img_float,
                    warped_mask=binary_mask,
                    original_depth=filtered_depth,
                    depth_conf=conf_map if depth_conf is not None else None,
                    depth_threshold=default_crack_params['depth_threshold'],
                    max_crack_size=default_crack_params['max_crack_size'],
                    min_valid_neighbors=default_crack_params['min_valid_neighbors']
                )
            
            warped_img = (filled_img_float * 255).astype(np.uint8)
            binary_mask = filled_mask
        
        warped_images.append(warped_img)
        warped_masks.append(binary_mask)
        
        current_angle = degree * (i+1) / frame_num
        camera_interpolations.append({
            "type": "single_view_warped",
            "direction": direction,
            "angle": current_angle,
            "camera_name": f"{direction}_{current_angle:.2f}_deg"
        })
    
    print("\nWarping complete!")
    return warped_images, warped_masks, camera_interpolations

