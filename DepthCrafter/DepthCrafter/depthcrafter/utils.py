from typing import Union, List
import tempfile
import numpy as np
import PIL.Image
import matplotlib.cm as cm
import mediapy
import torch
from decord import VideoReader, cpu
import os
import glob
import cv2

dataset_res_dict = {
    "sintel": [448, 1024],
    "scannet": [640, 832],
    "KITTI": [384, 1280],
    "bonn": [512, 640],
    "NYUv2": [448, 640],
}


def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open"):
    if os.path.isdir(video_path):
        print("==> processing image sequence: ", video_path)
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(video_path, ext)))
            image_files.extend(glob.glob(os.path.join(video_path, ext.upper())))
        
        if not image_files:
            raise ValueError(f"No images found in path {video_path}")
        
        image_files.sort()
        print(f"==> Found {len(image_files)} images")
        
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            try:
                pil_image = PIL.Image.open(image_files[0])
                first_image = np.array(pil_image)
                if len(first_image.shape) == 3 and first_image.shape[2] == 3:
                    pass
                else:
                    raise ValueError(f"Unsupported image format: {image_files[0]}")
            except Exception as e:
                raise ValueError(f"Cannot read image {image_files[0]}: {e}")
        else:
            first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        original_height, original_width = first_image.shape[:2]
        
        if dataset == "open":
            print("==> original image shape: ", (len(image_files), original_height, original_width, 3))
            height = round(original_height / 64) * 64
            width = round(original_width / 64) * 64
            if max(height, width) > max_res:
                scale = max_res / max(original_height, original_width)
                height = round(original_height * scale / 64) * 64
                width = round(original_width * scale / 64) * 64
        else:
            height = dataset_res_dict[dataset][0]
            width = dataset_res_dict[dataset][1]
        

        if target_fps == -1:
            fps = 30  
            stride = 1
        else:
            fps = target_fps
            stride = max(1, round(30 / target_fps)) 
        
        frames_idx = list(range(0, len(image_files), stride))
        print(f"==> downsampled shape: {len(frames_idx), height, width, 3}, with stride: {stride}")
        
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        
        print(f"==> final processing shape: {len(frames_idx), height, width, 3}")
        
        frames = []
        for idx in frames_idx:
            img = cv2.imread(image_files[idx])
            if img is None:
 
                try:
                    pil_image = PIL.Image.open(image_files[idx])
                    img = np.array(pil_image)
                    if len(img.shape) != 3 or img.shape[2] != 3:
                        raise ValueError(f"Format erorr: {image_files[idx]}")
     
                except Exception as e:
                    print(f"Warning: skip unreadable image {image_files[idx]}: {e}")
                    continue
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (width, height))
            frames.append(img)
        
        frames = np.array(frames).astype("float32") / 255.0
        return frames, fps
    
    else:

        if dataset == "open":
            print("==> processing video: ", video_path)
            vid = VideoReader(video_path, ctx=cpu(0))
            print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
            original_height, original_width = vid.get_batch([0]).shape[1:3]
            height = round(original_height / 64) * 64
            width = round(original_width / 64) * 64
            if max(height, width) > max_res:
                scale = max_res / max(original_height, original_width)
                height = round(original_height * scale / 64) * 64
                width = round(original_width * scale / 64) * 64
        else:
            height = dataset_res_dict[dataset][0]
            width = dataset_res_dict[dataset][1]

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        print(
            f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
        )
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        print(
            f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
        )
        frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

        return frames, fps


def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    crf: int = 18,
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
    return output_video_path


def save_frames(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_folder_path: str,
    prefix: str = "frame"
) -> str:

    os.makedirs(output_folder_path, exist_ok=True)
    
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    
    for i, frame in enumerate(video_frames):
        
        if len(frame.shape) == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        output_path = os.path.join(output_folder_path, f"{prefix}_{i:04d}.png")
        cv2.imwrite(output_path, frame_bgr)
    
    print(f"==> Save {len(video_frames)} frames to {output_folder_path}")
    return output_folder_path


class ColorMapper:

    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):

        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        image = self.colormap[image]
        return image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res
