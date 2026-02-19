import numpy as np
import cv2
from scipy.interpolate import griddata
from PIL import Image
from typing import List, Union
import os

import torch

try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer import (
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        AlphaCompositor,
        NormWeightedCompositor,
        look_at_view_transform,
        FoVOrthographicCameras,
        camera_conversions
    )
except ImportError:
    print("Pytorch3d not installed.")


def mask_to_numpy(mask_path):
    # Open the video file
    cap = cv2.VideoCapture(mask_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to float32 and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_normalized = frame.astype(np.float32) / 255.0
        frames.append(frame_normalized)
    
    cap.release()
    return np.array(frames)


def video_to_numpy(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to float32 and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_normalized = frame.astype(np.float32) / 255.0
        frames.append(frame_normalized)
    
    cap.release()
    return np.array(frames)


def export_to_video(
    video_frames: Union[List[np.ndarray], List[Image.Image]], output_path: str = None , fps: int = 10, H=None, W=None
) -> str:
    os.makedirs(f'{output_path}/images', exist_ok=True)
    output_video_path = os.path.join(output_path, f'video.mp4')

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = video_frames[0].shape[0:2] if H is None and W is None else (H, W)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        if H is not None and W is not None:
            img = cv2.resize(img, (W, H))
        cv2.imwrite(f"{output_path}/images/frames_{i:02d}.png", img)
        video_writer.write(img)
    return output_video_path


def resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def project_points_to_image_pytorch(points, features, extrinsic_matrix, intrinsics, image_size, morph=True):
    """
    Project 3D points onto an image plane using the provided camera parameters.
    
    Args:
        points (torch.Tensor): A tensor of shape (N, 3) representing the 3D points.
        features (torch.Tensor): A tensor of shape (N, F) representing the features for each point.
        extrinsic_matrix (torch.Tensor): A tensor of shape (4, 4) representing the camera extrinsic matrix.
        intrinsics (torch.Tensor): A tensor of shape (3, 3) representing the camera intrinsic matrix.
        image_size (torch.Tensor): A tensor of shape (2,) representing the size of the output image.
    
    Returns:
        torch.Tensor: A tensor of shape (H, W, F) representing the feature image.
    """
    point_cloud = Pointclouds(points=[points], features=[features])
    
    cameras = camera_conversions._cameras_from_opencv_projection(
        R=extrinsic_matrix[:3, :3].unsqueeze(0).float(), 
        tvec=extrinsic_matrix[:3, 3].unsqueeze(0).float(),
        camera_matrix=intrinsics.unsqueeze(0).float(),
        image_size=image_size.unsqueeze(0)
    )   


    # cameras = PerspectiveCameras(R=extrinsic_matrix[:3, :3].unsqueeze(0), 
    #                              T=extrinsic_matrix[:3, 3].unsqueeze(0), 
    #                             #  focal_length=intrinsics[0, 0].unsqueeze(0), 
    #                             #  principal_point=intrinsics[:2, 2].unsqueeze(0),
    #                                 focal_length=2.5,
    #                              device=points.device)
    
    
    # cameras = PerspectiveCameras(R=extrinsic_matrix[:3, :3].unsqueeze(0), 
    #                              T=extrinsic_matrix[:3, 3].unsqueeze(0), 
    #                             #  focal_length=intrinsics[0, 0].unsqueeze(0), 
    #                             #  principal_point=intrinsics[:2, 2].unsqueeze(0),
    #                                 focal_length=2.5,
    #                              device=points.device)
    
    
    raster_settings = PointsRasterizationSettings(
        image_size=(int(image_size[0]), int(image_size[1])), 
        radius = 0.005,
        points_per_pixel = 10,
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(point_cloud)
    points_idx = fragments.idx[0,:,:,0] # (H, W)

    image = features[points_idx]
    mask = torch.where(points_idx==-1, 0, 1).cpu().numpy()

    # renderer = PointsRenderer(
    #     rasterizer=rasterizer,
    #     compositor=AlphaCompositor()
    # )

    # image = renderer(point_cloud)[0]
    
    # mask = torch.where(image[..., -1] > 0, 1, 0).cpu().numpy()

    if morph: 
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    image[mask == 0] = 0

    return image.cpu().numpy(), mask[...,None]

def project_points_to_image_numpy(points, features, extrinsic_matrix, intrinsics, image_size):
    """
    Projects 3D points with colors to the 2D image plane and renders the image with occlusion handling.

    :param points: Nx3 numpy array of 3D points (x, y, z) in the world coordinate system.
    :param features: NxD numpy array of features corresponding to each 3D point.
    :param extrinsic_matrix: 4x4 numpy array representing the camera extrinsic matrix.
    :param intrinsics: 3x3 numpy array representing the camera intrinsics matrix.
    :param image_size: Tuple (height, width) representing the size of the output image.
    :return: Rendered image as a numpy array of shape (height, width, 3) and a mask of the pixels that are visible.
    """
    # Convert points to homogeneous coordinates by adding a fourth column of ones
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform points from world coordinates to camera coordinates
    points_camera = extrinsic_matrix @ points_homogeneous.T

    # Remove the last row (homogeneous coordinate) and transpose to Nx3
    points_camera = points_camera[:3, :].T

    # Project points onto the image plane
    height, width = image_size
    u = intrinsics[0, 0] * (points_camera[:, 0] / points_camera[:, 2]) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (points_camera[:, 1] / points_camera[:, 2]) + intrinsics[1, 2]
    depth = points_camera[:, 2]

    # Convert to integer pixel coordinates
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Initialize an empty image and depth buffer
    image = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf)
    mask = np.zeros((height, width), dtype=bool)

    # Clip the coordinates to be within the image bounds
    valid_indices = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_indices]
    v = v[valid_indices]
    depth = depth[valid_indices]
    features = features[valid_indices]

    # image[v, u] = colors

    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing='xy')
    grid = np.stack((x,y), axis=-1).reshape(-1,2)
    uv = np.stack((u,v), axis=-1)
    image = griddata(uv, features, grid, method='linear', fill_value=0).reshape(height,width,-1)
    image = image.clip(0, 1).astype(np.float32)

    # Assign colors to the corresponding pixels using depth buffer
    for i in range(len(u)):
        x, y, z = u[i], v[i], depth[i]
        if z < depth_buffer[y, x]:
            depth_buffer[y, x] = z
            image[y, x] = features[i]
            mask[y, x] = True

    # perform opening operation to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    image[mask == 0] = 0

    return image, mask


def look_at(camera_pos, target, up):
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    
    up = np.cross(forward, right)
    
    R = np.vstack([right, up, forward]).T
    return R


def get_look_up_camera(extrinsic, up_degree, look_at_depth):
    '''
    Get the camera matrix where the camera look at the scene with a certain degree up.
    :param extrinsic: 4x4 numpy array representing the camera extrinsic matrix.
    :param up_degree: The degree to look up.
    :param look_at_depth: The depth to look at.
    '''
    t = extrinsic.copy()[:3, 3]
    R = extrinsic.copy()[:3, :3]

    # Get the look at point
    look_at_point = t + R @ np.array([0, 0, look_at_depth])

    # Get the new camera position
    camera_pos = t
    camera_pos[1] = camera_pos[1] + np.sin(np.deg2rad(up_degree)) * look_at_depth
    camera_pos[2] = camera_pos[2] +(1 - np.cos(np.deg2rad(up_degree))) * look_at_depth

    # Get the new camera matrix
    new_R = look_at(camera_pos, look_at_point, np.array([0, 1, 0]))

    new_extrinsic = np.eye(4)
    new_extrinsic[:3, :3] = new_R
    new_extrinsic[:3, 3] = camera_pos

    return new_extrinsic


def get_look_right_camera(extrinsic, right_degree, look_at_depth):
    '''
    Get the camera matrix where the camera look at the scene with a certain degree right.
    :param extrinsic: 4x4 numpy array representing the camera extrinsic matrix.
    :param right_degree: The degree to look right.
    :param look_at_depth: The depth to look at.
    '''
    t = extrinsic.copy()[:3, 3]
    R = extrinsic.copy()[:3, :3]

    # Get the look at point
    look_at_point = t + R @ np.array([0, 0, look_at_depth])

    # Get the new camera position
    camera_pos = t
    camera_pos[0] = camera_pos[0] + np.sin(np.deg2rad(-right_degree)) * look_at_depth
    camera_pos[2] = camera_pos[2] +(1 - np.cos(np.deg2rad(-right_degree))) * look_at_depth

    # Get the new camera matrix
    new_R = look_at(camera_pos, look_at_point, np.array([0, 1, 0]))

    new_extrinsic = np.eye(4)
    new_extrinsic[:3, :3] = new_R
    new_extrinsic[:3, 3] = camera_pos

    return new_extrinsic


def get_look_up_camera_seq(extrinsics, max_degree, frame_num, look_at_depth):
    '''
    look at a point and the camera will rotate up to max_degree.
    '''
    cams = []

    for degree in np.linspace(0, max_degree, frame_num):
        new_cam = get_look_up_camera(extrinsics, degree, look_at_depth)
        cams.append(new_cam)

    return cams


def get_look_right_camera_seq(extrinsics, max_degree, frame_num, look_at_depth):
    '''
    look at a point and the camera will rotate right to max_degree.
    '''
    cams = []

    for degree in np.linspace(0, max_degree, frame_num):
        new_cam = get_look_right_camera(extrinsics, degree, look_at_depth)
        cams.append(new_cam)

    return cams


def get_circle_camera_seq(extrinsics, radius, frame_num, look_at_depth, direction='right'):
    '''
    look at a point and the camera will rotate around the point with a certain radius.
    '''
    
    t = extrinsics.copy()[:3, 3]
    R = extrinsics.copy()[:3, :3]

    # Get the look at point
    look_at_point = t + R @ np.array([0, 0, look_at_depth])


    cams = []

    for degree in np.linspace(0, 2*np.pi, frame_num):
        camera_pos = extrinsics[:3, 3].copy()
        if direction == 'right':
            camera_pos[0] = camera_pos[0] + radius * (np.cos(degree)-1)
        elif direction == 'left':
            camera_pos[0] = camera_pos[0] - radius * (np.cos(degree)-1)
        else:
            raise ValueError('direction should be either right or left.')
        camera_pos[1] = camera_pos[1] + radius * np.sin(degree)

        new_R = look_at(camera_pos, look_at_point, np.array([0, 1, 0]))

        new_extrinsic = np.eye(4)
        new_extrinsic[:3, :3] = new_R
        new_extrinsic[:3, 3] = camera_pos

        cams.append(new_extrinsic)
        

    return cams


def apply_zoom_to_camera_seq(cams, zoom_mode, rate, look_at_depth):

    if zoom_mode == 'none':
        return cams
    
    if not (0.0 < rate <= 1.0):
        raise ValueError("rate must be between 0.0 and 1.0")
    
    zoomed_cams = []
    frame_num = len(cams)
    
    for i, cam in enumerate(cams):
        camera_pos = cam[:3, 3].copy()
        R = cam[:3, :3].copy()
        
        look_at_point = camera_pos + R @ np.array([0, 0, look_at_depth])
        
        distance_vector = camera_pos - look_at_point
        
        progress = i / (frame_num - 1) if frame_num > 1 else 0.0
        
        if zoom_mode == 'zoom_out':
            current_rate = 1.0 - progress * (1.0 - rate)
        elif zoom_mode == 'zoom_in':
            current_rate = 1.0 + progress * (1.0 / rate - 1.0)
        else:
            current_rate = 1.0
        
        new_camera_pos = look_at_point + distance_vector * current_rate
        
        new_R = look_at(new_camera_pos, look_at_point, np.array([0, 1, 0]))
        
        new_cam = np.eye(4)
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_camera_pos
        
        zoomed_cams.append(new_cam)
    
    return zoomed_cams


def get_stable_look_up_camera_seq(extrinsics, max_degree, frame_num, look_at_depth, stable_frame=17):
    cams = []
    stable_frame = min(stable_frame, frame_num)  
    
    for i in range(frame_num):
        if i < stable_frame:
            if stable_frame > 1:
                degree = (i / (stable_frame - 1)) * max_degree
            else:
                degree = max_degree
        else:
            degree = max_degree
        
        new_cam = get_look_up_camera(extrinsics, degree, look_at_depth)
        cams.append(new_cam)
    
    return cams


def get_stable_look_right_camera_seq(extrinsics, max_degree, frame_num, look_at_depth, stable_frame=17):
    cams = []
    stable_frame = min(stable_frame, frame_num)
    
    for i in range(frame_num):
        if i < stable_frame:
            if stable_frame > 1:
                degree = (i / (stable_frame - 1)) * max_degree
            else:
                degree = max_degree
        else:
            degree = max_degree
        
        new_cam = get_look_right_camera(extrinsics, degree, look_at_depth)
        cams.append(new_cam)
    
    return cams


def apply_stable_zoom_to_camera_seq(cams, zoom_mode, rate, look_at_depth, stable_frame=17):
    if zoom_mode == 'none':
        return cams
    
    if not (0.0 < rate <= 1.0):
        raise ValueError("rate must be between 0.0 and 1.0")
    
    zoomed_cams = []
    frame_num = len(cams)
    stable_frame = min(stable_frame, frame_num)
    
    for i, cam in enumerate(cams):
        camera_pos = cam[:3, 3].copy()
        R = cam[:3, :3].copy()
        
        look_at_point = camera_pos + R @ np.array([0, 0, look_at_depth])
        distance_vector = camera_pos - look_at_point
        
        if i < stable_frame:
            if stable_frame > 1:
                progress = i / (stable_frame - 1)
            else:
                progress = 1.0
        else:
            progress = 1.0
        
        if zoom_mode == 'zoom_out':
            zoom_factor = 1.0 - progress * (1.0 - rate)
        elif zoom_mode == 'zoom_in':
            zoom_factor = 1.0 + progress * (1.0 / rate - 1.0)
        else:
            zoom_factor = 1.0
        
        new_camera_pos = look_at_point + distance_vector * zoom_factor
        new_R = look_at(new_camera_pos, look_at_point, np.array([0, 1, 0]))
        
        new_cam = cam.copy()
        new_cam[:3, :3] = new_R
        new_cam[:3, 3] = new_camera_pos
        
        zoomed_cams.append(new_cam)
    
    return zoomed_cams


def detect_depth_edges(depth_map, edge_threshold=0.1, kernel_size=3):
    """
    Detect edge regions in depth map
    
    Args:
        depth_map (np.ndarray): Depth map (H, W)
        edge_threshold (float): Edge detection threshold
        kernel_size (int): Sobel kernel size
    
    Returns:
        np.ndarray: Edge mask, True indicates edge regions
    """
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    edge_mask = gradient_magnitude > edge_threshold
    
    return edge_mask


def filter_edge_points(points, features, depth_2d, edge_threshold=0.1, edge_dilation=3, 
                      depth_jump_threshold=0.3, neighbor_check_radius=2):
    """
    Filter point cloud near depth edges to reduce warping artifacts
    
    Args:
        points (torch.Tensor): 3D point cloud (N, 3)
        features (torch.Tensor): Point features (N, F)  
        depth_2d (torch.Tensor): Original depth map (H, W)
        edge_threshold (float): Edge detection threshold
        edge_dilation (int): Edge region dilation in pixels
        depth_jump_threshold (float): Depth discontinuity threshold
        neighbor_check_radius (int): Neighbor check radius
    
    Returns:
        tuple: Filtered (points, features)
    """
    H, W = depth_2d.shape
    device = points.device
    
    depth_np = depth_2d.cpu().numpy()
    edge_mask = detect_depth_edges(depth_np, edge_threshold)
    
    if edge_dilation > 0:
        kernel = np.ones((edge_dilation*2+1, edge_dilation*2+1), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    if depth_jump_threshold > 0 and neighbor_check_radius > 0:
        from scipy import ndimage
        depth_min = ndimage.minimum_filter(depth_np, size=neighbor_check_radius*2+1)
        depth_max = ndimage.maximum_filter(depth_np, size=neighbor_check_radius*2+1)
        depth_variation = depth_max - depth_min
        
        jump_mask = depth_variation > depth_jump_threshold
        edge_mask = edge_mask | jump_mask
    
    edge_mask_torch = torch.from_numpy(edge_mask).to(device)
    edge_mask_flat = edge_mask_torch.flatten()
    
    if points.shape[0] != H * W:
        print(f"Warning: Point count ({points.shape[0]}) doesn't match pixel count ({H*W}), using original projection")
        return points, features
    
    valid_mask = ~edge_mask_flat
    filtered_points = points[valid_mask]
    filtered_features = features[valid_mask]
    
    return filtered_points, filtered_features


def project_points_to_image_pytorch_with_edge_filter(points, features, extrinsic_matrix, intrinsics, 
                                                    image_size, depth_2d=None, 
                                                    enable_edge_filter=True, 
                                                    edge_threshold=0.1, 
                                                    edge_dilation=3,
                                                    depth_jump_threshold=0.3,
                                                    neighbor_check_radius=2,
                                                    morph=True,
                                                    verbose=False):
    """
    Point cloud projection with edge filtering
    """
    original_point_count = points.shape[0]
    
    if enable_edge_filter and depth_2d is not None:
        points, features = filter_edge_points(
            points, features, depth_2d, 
            edge_threshold, edge_dilation, 
            depth_jump_threshold, neighbor_check_radius
        )
        
        filtered_count = points.shape[0]
        removed_count = original_point_count - filtered_count
        removal_ratio = removed_count / original_point_count
        
        if verbose:
            print(f"  Edge filtering: original={original_point_count}, filtered={filtered_count}, "
                  f"removed={removed_count} ({removal_ratio:.1%})")
    
    return project_points_to_image_pytorch(points, features, extrinsic_matrix, 
                                         intrinsics, image_size, morph)