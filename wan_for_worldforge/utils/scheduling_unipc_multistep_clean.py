# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import deprecate, is_scipy_available
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

if is_scipy_available():
    import scipy.stats


class VideoMotionPCASelector:
    """
    Video motion trajectory PCA channel selector.
    Filters VAE latent channels most correlated with reference video motion using optical flow.
    Used for Flow-guided Latent Fusion (FLF) channel selection.
    """
    
    def __init__(self):
        self._flf_initialized = False

    def _sanitize_data(self, data: torch.Tensor, name: str = "data") -> torch.Tensor:
        """Clean NaN and Inf values from data."""
        if not torch.isfinite(data).all():
            finite_mask = torch.isfinite(data)
            if finite_mask.any():
                finite_mean = data[finite_mask].mean()
                data = torch.where(torch.isfinite(data), data, finite_mean)
            else:
                data = torch.zeros_like(data)
        return data

    def _validate_mask(self, mask: torch.Tensor, video_shape: tuple) -> torch.Tensor:
        """Validate and standardize mask format to [B, 1, T, H, W]."""
        if mask is None:
            return None
            
        mask = self._sanitize_data(mask, "mask")
        mask = torch.clamp(mask, 0.0, 1.0)
        
        B, C, T, H, W = video_shape
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 4:
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(1)
            else:
                mask = mask.unsqueeze(0)
        
        if mask.shape[1] != 1:
            mask = mask[:, 0:1]
        
        if mask.shape != (B, 1, T, H, W):
            mask = F.interpolate(
                mask.view(1, 1, -1, mask.shape[-2], mask.shape[-1]),
                size=(T, H, W),
                mode='nearest'
            ).view(B, 1, T, H, W)
        
        return mask

    def extract_motion_trajectory_robust(self, video: torch.Tensor, mask: torch.Tensor = None, 
                                        use_optical_flow: bool = True) -> torch.Tensor:
        """
        Extract motion trajectory using optical flow or temporal differences.
        
        Args:
            video: [1, 3, T, H, W] reference video
            mask: [1, 1, T, H, W] valid pixel mask
            use_optical_flow: Whether to use optical flow algorithm
            
        Returns:
            motion_features: [1, T-1, 2, H, W] optical flow or [1, T-1, 3, H, W] difference features
        """
        if video.dim() != 5:
            raise ValueError(f"Video must be 5D tensor [B, C, T, H, W], got {video.dim()}D")
        
        batch_size, channels, frames, height, width = video.shape
        if frames < 2:
            raise ValueError(f"Video needs at least 2 frames, got {frames}")
        
        video = video.to(dtype=torch.float32)
        if mask is not None:
            mask = mask.to(dtype=torch.float32)
        
        video = self._sanitize_data(video, "video")
        mask = self._validate_mask(mask, video.shape)
        
        if use_optical_flow:
            try:
                motion_features = self._extract_optical_flow_motion(video, mask)
            except Exception:
                motion_features = self._extract_multiscale_motion(video, mask)
        else:
            motion_features = self._extract_multiscale_motion(video, mask)
        
        return motion_features

    def _extract_multiscale_motion(self, video: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Extract motion using multiscale temporal differences."""
        batch_size, channels, frames, height, width = video.shape
        
        smoothed_video = self._apply_gaussian_smoothing(video)
        
        motion_components = []
        scales = [1, 2]
        weights = [0.7, 0.3]
        
        for scale, weight in zip(scales, weights):
            if frames > scale:
                if scale == 1:
                    motion_scale = smoothed_video[:, :, 1:] - smoothed_video[:, :, :-1]
                else:
                    motion_scale = (smoothed_video[:, :, scale:] - smoothed_video[:, :, :-scale]) / scale
                
                if motion_scale.shape[2] > frames - 1:
                    motion_scale = motion_scale[:, :, :frames-1]
                elif motion_scale.shape[2] < frames - 1:
                    padding_frames = frames - 1 - motion_scale.shape[2]
                    last_frame = motion_scale[:, :, -1:].repeat(1, 1, padding_frames, 1, 1)
                    motion_scale = torch.cat([motion_scale, last_frame], dim=2)
                
                motion_components.append(weight * motion_scale)
        
        motion = sum(motion_components)
        motion = motion.permute(0, 2, 1, 3, 4)
        motion = self._enhance_spatial_gradients(motion)
        
        if mask is not None:
            motion = self._apply_motion_mask(motion, mask)
        
        motion = motion.to(device=video.device, dtype=video.dtype)
        motion = self._sanitize_data(motion, "motion")
        
        return motion

    def _extract_optical_flow_motion(self, video: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Extract motion using Farneback optical flow algorithm."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")
        
        batch_size, channels, frames, height, width = video.shape
        
        video_f32 = video.to(dtype=torch.float32)
        if mask is not None:
            mask = mask.to(dtype=torch.float32)
        
        video_np = video_f32.squeeze(0).cpu().numpy()
        video_np = video_np.transpose(1, 2, 3, 0)
        
        video_min = video_np.min()
        video_max = video_np.max()
        
        if video_min >= -0.01 and video_max <= 1.01:
            video_np = (video_np * 255).astype(np.uint8)
        elif video_min >= -1.01 and video_max <= 1.01:
            video_np = ((video_np + 1.0) * 127.5).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)
        
        if mask is not None:
            mask_np = mask.squeeze().cpu().numpy()
            if mask_np.ndim == 4:
                mask_np = mask_np.squeeze(0)
            if mask_np.ndim == 3:
                mask_np = mask_np.transpose(1, 2, 0)
            else:
                mask_np = None
        else:
            mask_np = None
        
        flow_list = []
        
        for t in range(frames - 1):
            frame1 = video_np[t]
            frame2 = video_np[t + 1]
            
            if frame1.shape[2] == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            elif frame1.shape[2] == 1:
                gray1 = frame1.squeeze(-1)
                gray2 = frame2.squeeze(-1)
            else:
                gray1 = frame1[:, :, 0]
                gray2 = frame2[:, :, 0]
            
            if mask_np is not None:
                if mask_np.ndim == 3:
                    current_mask = mask_np[:, :, min(t + 1, mask_np.shape[2] - 1)]
                    current_mask = (current_mask > 0.5).astype(np.uint8) * 255
                else:
                    current_mask = (mask_np > 0.5).astype(np.uint8) * 255
                
                gray1 = cv2.bitwise_and(gray1, current_mask)
                gray2 = cv2.bitwise_and(gray2, current_mask)
            
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
            except:
                diff = frame2.astype(np.float32) - frame1.astype(np.float32)
                grad_x = cv2.Sobel(diff.mean(axis=2), cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(diff.mean(axis=2), cv2.CV_32F, 0, 1, ksize=3)
                flow = np.stack([grad_x, grad_y], axis=2)
            
            if mask_np is not None:
                if mask_np.ndim == 3:
                    current_mask = mask_np[:, :, min(t + 1, mask_np.shape[2] - 1)]
                else:
                    current_mask = mask_np
                
                flow[:, :, 0] *= current_mask
                flow[:, :, 1] *= current_mask
            
            flow_list.append(flow)
        
        flows = np.stack(flow_list, axis=0)
        flows = flows.transpose(0, 3, 1, 2)
        flows_tensor = torch.from_numpy(flows).float().unsqueeze(0)
        flows_tensor = flows_tensor.to(device=video.device, dtype=video.dtype)
        flows_tensor = self._sanitize_data(flows_tensor, "flow")
        
        return flows_tensor

    def _apply_gaussian_smoothing(self, video: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
        """Apply Gaussian smoothing to reduce noise."""
        batch_size, channels, frames, height, width = video.shape
        
        original_dtype = video.dtype
        video = video.to(dtype=torch.float32)
        
        video_reshaped = video.view(batch_size * channels * frames, 1, height, width)
        
        if height >= kernel_size and width >= kernel_size:
            coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            kernel = g.outer(g).unsqueeze(0).unsqueeze(0).to(video.device)
            
            padding = kernel_size // 2
            smoothed = F.conv2d(video_reshaped, kernel, padding=padding, groups=1)
        else:
            smoothed = video_reshaped
        
        smoothed = smoothed.view(batch_size, channels, frames, height, width)
        smoothed = smoothed.to(dtype=original_dtype)
        
        return smoothed

    def _enhance_spatial_gradients(self, motion: torch.Tensor) -> torch.Tensor:
        """Enhance spatial gradient information in motion."""
        batch_size, frames, channels, height, width = motion.shape
        
        original_dtype = motion.dtype
        motion = motion.to(dtype=torch.float32)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_x = sobel_x.to(motion.device)
        sobel_y = sobel_y.to(motion.device)
        
        enhanced_motion = []
        
        for t in range(frames):
            motion_frame = motion[:, t]
            
            if height >= 3 and width >= 3:
                gradients = []
                for c in range(channels):
                    channel_data = motion_frame[:, c:c+1]
                    grad_x = F.conv2d(channel_data, sobel_x, padding=1)
                    grad_y = F.conv2d(channel_data, sobel_y, padding=1)
                    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
                    gradients.append(grad_magnitude)
                
                enhanced_frame = torch.cat(gradients, dim=1)
                enhanced_frame = 0.7 * motion_frame + 0.3 * enhanced_frame
            else:
                enhanced_frame = motion_frame
            
            enhanced_motion.append(enhanced_frame)
        
        enhanced_motion = torch.stack(enhanced_motion, dim=1)
        enhanced_motion = enhanced_motion.to(dtype=original_dtype)
        
        return enhanced_motion

    def _apply_motion_mask(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply motion mask to filter motion features."""
        batch_size, motion_frames, channels, height, width = motion.shape
        
        mask_motion = mask[:, :, 1:]
        
        if mask_motion.shape[2] != motion_frames:
            mask_motion = F.interpolate(
                mask_motion.view(1, 1, -1, height, width),
                size=(motion_frames, height, width),
                mode='nearest'
            ).view(1, 1, motion_frames, height, width)
        
        mask_motion = mask_motion.repeat(1, channels, 1, 1, 1)
        mask_motion = mask_motion.permute(0, 2, 1, 3, 4)
        masked_motion = motion * mask_motion
        
        return masked_motion

    def extract_motion_trajectory(self, video: torch.Tensor, mask: torch.Tensor = None, 
                                 use_optical_flow: bool = True) -> torch.Tensor:
        """Compatibility method."""
        return self.extract_motion_trajectory_robust(video, mask, use_optical_flow)

    def select_motion_related_channels(self, 
                                      pred_original_sample: torch.Tensor,
                                      video_latents: torch.Tensor,
                                      mask: torch.Tensor = None,
                                      keep_channels: int = 12,
                                      current_step: int = 0,
                                      total_steps: int = 50,
                                      use_optical_flow: bool = True,
                                      static: bool = False,
                                      **kwargs) -> List[int]:
        """
        Select channels most related to motion trajectory using optical flow analysis.
        
        Args:
            pred_original_sample: [1, 16, T, h, w] predicted original sample
            video_latents: [1, 16, T, h, w] encoded reference video
            mask: [1, 1, T, h, w] valid pixel mask
            keep_channels: Number of channels to keep
            current_step: Current sampling step
            total_steps: Total sampling steps
            use_optical_flow: Whether to use optical flow
            static: Whether to use static parameters
            
        Returns:
            channels_to_replace: List of channel indices to replace
        """
        if current_step < 2:
            return []
        
        if pred_original_sample.dim() != 5 or video_latents.dim() != 5:
            return list(range(min(2, pred_original_sample.shape[1])))
        
        if mask is not None and mask.dim() == 5 and mask.shape[1] != 1:
            mask = mask[:, 0:1]
        
        aligned_video = video_latents
        aligned_video_f32 = aligned_video.to(dtype=torch.float32)
        
        video_global_min = aligned_video_f32.min()
        video_global_max = aligned_video_f32.max()
        video_global_range = video_global_max - video_global_min + 1e-8
        
        try:
            channel_motion_features = []
            for ch_idx in range(aligned_video_f32.shape[1]):
                channel_data = aligned_video_f32[:, ch_idx:ch_idx+1]
                
                if use_optical_flow:
                    try:
                        channel_rgb = channel_data.repeat(1, 3, 1, 1, 1)
                        channel_rgb = (channel_rgb - video_global_min) / video_global_range
                        channel_motion = self._extract_optical_flow_motion(channel_rgb, None)
                    except Exception:
                        channel_motion = channel_data[:, :, 1:] - channel_data[:, :, :-1]
                        channel_motion = channel_motion.permute(0, 2, 1, 3, 4)
                else:
                    channel_motion = channel_data[:, :, 1:] - channel_data[:, :, :-1]
                    channel_motion = channel_motion.permute(0, 2, 1, 3, 4)
                
                channel_motion_features.append(channel_motion)
        except Exception:
            return list(range(min(2, pred_original_sample.shape[1])))
        
        channel_correlations = self._compute_channel_correlations(
            pred_original_sample, aligned_video, None, use_optical_flow, channel_motion_features
        )
        
        if len(channel_correlations) == 0:
            return list(range(min(2, pred_original_sample.shape[1])))
        
        channel_correlations = np.array(channel_correlations)
        corr_mean = np.mean(channel_correlations)
        corr_std = np.std(channel_correlations)
        
        if current_step <= 10:
            if current_step <= 5:
                max_replace = 0
            else:
                max_replace = 1
            channels_to_replace = np.argsort(channel_correlations)[:max_replace].tolist()
        else:
            scale_std = 0.625
            max_replace = 6
            
            threshold = corr_mean - scale_std * corr_std
            below_threshold_indices = [i for i, score in enumerate(channel_correlations) if score < threshold]
            
            min_replace = 2
            
            if len(below_threshold_indices) < min_replace:
                channels_to_replace = np.argsort(channel_correlations)[:min_replace].tolist()
            elif len(below_threshold_indices) > max_replace:
                below_threshold_scores = [(idx, channel_correlations[idx]) for idx in below_threshold_indices]
                below_threshold_scores.sort(key=lambda x: x[1])
                channels_to_replace = [idx for idx, _ in below_threshold_scores[:max_replace]]
            else:
                channels_to_replace = below_threshold_indices
        
        channels_to_replace.sort()
        return channels_to_replace
    
    def _compute_channel_correlations(self, 
                                     pred_original_sample: torch.Tensor,
                                     aligned_video: torch.Tensor,
                                     aligned_mask: torch.Tensor = None,
                                     use_optical_flow: bool = True,
                                     channel_motion_features: List[torch.Tensor] = None) -> List[float]:
        """Compute channel correlations using optical flow metrics (M-EPE, M-AE, Fl-all)."""
        device = pred_original_sample.device
        correlations = []
        
        aligned_video = aligned_video.to(device, dtype=pred_original_sample.dtype)
        
        if channel_motion_features is not None and len(channel_motion_features) == pred_original_sample.shape[1]:
            ref_motion_features = channel_motion_features
        else:
            ref_motion_features = []
            for ch_idx in range(aligned_video.shape[1]):
                channel_data = aligned_video[:, ch_idx:ch_idx+1]
                channel_motion = channel_data[:, :, 1:] - channel_data[:, :, :-1]
                channel_motion = channel_motion.permute(0, 2, 1, 3, 4)
                ref_motion_features.append(channel_motion)
        
        pred_f32 = pred_original_sample.to(dtype=torch.float32)
        pred_global_min = pred_f32.min()
        pred_global_max = pred_f32.max()
        pred_global_range = pred_global_max - pred_global_min + 1e-8
        
        for channel_idx in range(pred_original_sample.shape[1]):
            try:
                channel_data = pred_original_sample[:, channel_idx:channel_idx+1]
                channel_data_f32 = channel_data.to(dtype=torch.float32)
                
                if use_optical_flow:
                    try:
                        channel_rgb = channel_data_f32.repeat(1, 3, 1, 1, 1)
                        channel_rgb = (channel_rgb - pred_global_min) / pred_global_range
                        channel_rgb = channel_rgb.to(dtype=torch.float32)
                        channel_motion = self._extract_optical_flow_motion(channel_rgb, None)
                    except Exception:
                        channel_motion = channel_data_f32[:, :, 1:] - channel_data_f32[:, :, :-1]
                        channel_motion = channel_motion.permute(0, 2, 1, 3, 4)
                        channel_motion = channel_motion.to(device=device)
                else:
                    channel_motion = channel_data_f32[:, :, 1:] - channel_data_f32[:, :, :-1]
                    channel_motion = channel_motion.permute(0, 2, 1, 3, 4)
                
                if channel_idx < len(ref_motion_features):
                    ref_channel_motion = ref_motion_features[channel_idx]
                    similarity = self._compute_flow_metrics(ref_channel_motion, channel_motion, None)
                    correlations.append(similarity)
                else:
                    correlations.append(0.0)
                
            except Exception:
                correlations.append(0.0)
        
        return correlations
    
    def _compute_flow_metrics(self, ref_motion: torch.Tensor, channel_motion: torch.Tensor, 
                             mask: torch.Tensor = None) -> float:
        """
        Compute optical flow evaluation metrics: M-EPE, M-AE, Fl-all.
        
        Args:
            ref_motion: [1, T-1, C_ref, H, W] reference motion features
            channel_motion: [1, T-1, C_chan, H, W] channel motion features
            mask: [1, 1, T, H, W] mask (optional)
            
        Returns:
            float: similarity score [0, 1]
        """
        try:
            device = ref_motion.device
            ref_motion = ref_motion.to(device)
            channel_motion = channel_motion.to(device, dtype=ref_motion.dtype)
            
            min_frames = min(ref_motion.shape[1], channel_motion.shape[1])
            if min_frames <= 0:
                return 0.0
                
            ref_aligned = ref_motion[:, :min_frames]
            chan_aligned = channel_motion[:, :min_frames]
            
            if ref_aligned.shape[-2:] != chan_aligned.shape[-2:]:
                chan_aligned = F.interpolate(
                    chan_aligned.view(-1, chan_aligned.shape[-2], chan_aligned.shape[-1]),
                    size=ref_aligned.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).view(chan_aligned.shape[0], chan_aligned.shape[1], chan_aligned.shape[2], 
                       ref_aligned.shape[-2], ref_aligned.shape[-1])
            
            if ref_aligned.shape[2] >= 2:
                ref_flow = ref_aligned[:, :, :2]
            else:
                ref_flow = ref_aligned.repeat(1, 1, 2, 1, 1)[:, :, :2]
                
            if chan_aligned.shape[2] >= 2:
                chan_flow = chan_aligned[:, :, :2]
            else:
                chan_flow = chan_aligned.repeat(1, 1, 2, 1, 1)[:, :, :2]
            
            # M-EPE (Mean End Point Error)
            flow_diff = ref_flow - chan_flow
            epe = torch.sqrt((flow_diff ** 2).sum(dim=2) + 1e-8)
            
            # M-AE (Mean Angular Error)
            dot_product = (ref_flow * chan_flow).sum(dim=2)
            ref_norm = torch.sqrt((ref_flow ** 2).sum(dim=2) + 1e-8)
            chan_norm = torch.sqrt((chan_flow ** 2).sum(dim=2) + 1e-8)
            cos_angle = dot_product / (ref_norm * chan_norm + 1e-8)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            angle_error = torch.acos(cos_angle) * 180.0 / torch.pi
            
            # Fl-all (Outlier Percentage)
            ref_motion_magnitude = ref_norm
            threshold_abs = 3.0
            threshold_rel = 0.05
            outlier_mask = (epe > threshold_abs) & (epe > ref_motion_magnitude * threshold_rel)

            if mask is not None:
                motion_mask = mask[:, :, 1:1+min_frames]
                
                if motion_mask.shape[2] != min_frames:
                    motion_mask = F.interpolate(
                        motion_mask.view(1, 1, -1, motion_mask.shape[-2], motion_mask.shape[-1]),
                        size=(min_frames, motion_mask.shape[-2], motion_mask.shape[-1]),
                        mode='nearest'
                    ).view(1, 1, min_frames, motion_mask.shape[-2], motion_mask.shape[-1])
                
                if motion_mask.shape[-2:] != epe.shape[-2:]:
                    motion_mask = F.interpolate(
                        motion_mask.view(-1, motion_mask.shape[-2], motion_mask.shape[-1]),
                        size=epe.shape[-2:],
                        mode='nearest'
                    ).view(motion_mask.shape[:3] + epe.shape[-2:])
                
                motion_mask = motion_mask.squeeze(1)
                valid_pixels = motion_mask.sum()
                
                if valid_pixels > 0:
                    m_epe = (epe * motion_mask).sum() / valid_pixels
                    m_ae = (angle_error * motion_mask).sum() / valid_pixels
                    fl_all = (outlier_mask.float() * motion_mask).sum() / valid_pixels
                else:
                    m_epe = torch.tensor(0.0, device=device)
                    m_ae = torch.tensor(0.0, device=device)
                    fl_all = torch.tensor(0.0, device=device)
            else:
                m_epe = epe.mean()
                m_ae = angle_error.mean()
                fl_all = outlier_mask.float().mean()
            
            # Normalize metrics to [0, 1] range
            norm_epe = torch.clamp(m_epe / 10.0, 0.0, 1.0)
            norm_fl = torch.clamp(fl_all / 0.5, 0.0, 1.0)
            norm_ae = torch.clamp(m_ae / 30.0, 0.0, 1.0)
            
            # Weighted error: M-EPE 45%, Fl-all 45%, M-AE 10%
            weighted_error = 0.45 * norm_epe + 0.45 * norm_fl + 0.1 * norm_ae
            
            # Convert to similarity (1 - error)
            similarity = 1.0 - weighted_error
            similarity = torch.clamp(similarity, 0.0, 1.0)
            
            return similarity.item()
            
        except Exception:
            return 0.0


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
    """Create beta schedule that discretizes alpha_t_bar function."""
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    """Rescale betas to have zero terminal SNR."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class UniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    UniPCMultistepScheduler with optical flow guided generation and resampling support.
    
    Based on UniPC (https://arxiv.org/abs/2302.04867) with extensions for:
    - Optical flow guided channel selection (FLF)
    - Guided resampling mechanism
    - Video latent fusion
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p: SchedulerMixin = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        use_flow_sigmas: Optional[bool] = False,
        flow_shift: Optional[float] = 1.0,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: Optional[str] = "zero",
        rescale_betas_zero_snr: bool = False,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("scipy required for beta sigmas")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError("Only one sigma type can be used")
            
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} not implemented")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2**-24

        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        self.init_noise_sigma = 1.0

        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(f"{solver_type} not implemented")

        self.predict_x0 = predict_x0
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.outputs_for_c = [None] * solver_order
        
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

        self.derivative_history = []
        self.dt_history = []
        self.last_lower_order_nums = 0
        self.this_order = None
        self.last_this_order = None
        
        self._pca_selector = None
        
        self.resample_sigmas = None
        self.resample_timesteps = None
        self.is_resampling = False
        self.original_step_index = None
        
    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """Set discrete timesteps for diffusion chain."""
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // (num_inference_steps + 1)
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.arange(self.config.num_train_timesteps, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"{self.config.timestep_spacing} not supported")

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        
        if self.config.use_karras_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            sigma_last = sigmas[-1] if self.config.final_sigmas_type == "sigma_min" else 0
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        elif self.config.use_exponential_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
            sigma_last = sigmas[-1] if self.config.final_sigmas_type == "sigma_min" else 0
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        elif self.config.use_beta_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
            sigma_last = sigmas[-1] if self.config.final_sigmas_type == "sigma_min" else 0
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        elif self.config.use_flow_sigmas:
            alphas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)
            sigmas = 1.0 - alphas
            sigmas = np.flip(self.config.flow_shift * sigmas / (1 + (self.config.flow_shift - 1) * sigmas))[:-1].copy()
            timesteps = (sigmas * self.config.num_train_timesteps).copy()
            sigma_last = sigmas[-1] if self.config.final_sigmas_type == "sigma_min" else 0
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5 if self.config.final_sigmas_type == "sigma_min" else 0
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")
        
        self.clear_motion_cache()
        self._compute_resample_sigmas_and_timesteps()
        
        if self.resample_sigmas is not None:
            self.resample_sigmas = self.resample_sigmas.to(device)
        if self.resample_timesteps is not None:
            self.resample_timesteps = self.resample_timesteps.to(device)

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Dynamic thresholding."""
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def _sigma_to_t(self, sigma, log_sigmas):
        log_sigma = np.log(np.maximum(sigma, 1e-10))
        dists = log_sigma - log_sigmas[:, np.newaxis]
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    def _sigma_to_alpha_sigma_t(self, sigma):
        if self.config.use_flow_sigmas:
            alpha_t = 1 - sigma
            sigma_t = sigma
        else:
            alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
            sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """Karras et al. (2022) noise schedule."""
        sigma_min = getattr(self.config, "sigma_min", None) or in_sigmas[-1].item()
        sigma_max = getattr(self.config, "sigma_max", None) or in_sigmas[0].item()

        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Exponential noise schedule."""
        sigma_min = getattr(self.config, "sigma_min", None) or in_sigmas[-1].item()
        sigma_max = getattr(self.config, "sigma_max", None) or in_sigmas[0].item()
        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    def _convert_to_beta(self, in_sigmas: torch.Tensor, num_inference_steps: int, 
                        alpha: float = 0.6, beta: float = 0.6) -> torch.Tensor:
        """Beta sampling schedule (Lee et. al, 2024)."""
        sigma_min = getattr(self.config, "sigma_min", None) or in_sigmas[-1].item()
        sigma_max = getattr(self.config, "sigma_max", None) or in_sigmas[0].item()

        sigmas = np.array([
            sigma_min + (ppf * (sigma_max - sigma_min))
            for ppf in [
                scipy.stats.beta.ppf(timestep, alpha, beta)
                for timestep in 1 - np.linspace(0, 1, num_inference_steps)
            ]
        ])
        return sigmas

    def convert_model_output(self, model_output: torch.Tensor, *args, sample: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Convert model output to x0 prediction."""
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as required argument")
        if timestep is not None:
            deprecate("timesteps", "1.0.0", 
                     "Passing `timesteps` is deprecated and has no effect")

        if self.is_resampling and self.resample_sigmas is not None:
            sigma_index = min(self.step_index, len(self.resample_sigmas) - 1)
            sigma = self.resample_sigmas[sigma_index]
        else:
            sigma = self.sigmas[self.step_index]
            
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.config.prediction_type == "flow_prediction":
                if self.is_resampling and self.resample_sigmas is not None:
                    sigma_index = min(self.step_index, len(self.resample_sigmas) - 1)
                    sigma_t = self.resample_sigmas[sigma_index]
                else:
                    sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(f"prediction_type {self.config.prediction_type} not supported")

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(f"prediction_type {self.config.prediction_type} not supported")

    def multistep_uni_p_bh_update(self, model_output: torch.Tensor, *args, 
                                  sample: torch.Tensor = None, order: int = None, **kwargs) -> torch.Tensor:
        """UniP (B(h) version) update step."""
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as required argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError("missing `order` as required argument")
        if prev_timestep is not None:
            deprecate("prev_timestep", "1.0.0", "Passing `prev_timestep` is deprecated")
            
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        if self.is_resampling and self.resample_sigmas is not None:
            current_index = min(self.step_index, len(self.resample_sigmas) - 1)
            next_index = min(self.step_index + 1, len(self.resample_sigmas) - 1)
            if next_index < len(self.resample_sigmas):
                sigma_t = self.sigmas[next_index]
            else:
                sigma_t = self.sigmas[min(self.step_index + 1, len(self.sigmas) - 1)]
            sigma_s0 = self.resample_sigmas[current_index]
        else:
            sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
            
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            
            if self.is_resampling and self.resample_sigmas is not None:
                si_index = min(max(si, 0), len(self.resample_sigmas) - 1)
                if si_index < len(self.resample_sigmas):
                    sigma_si_value = self.resample_sigmas[si_index]
                else:
                    sigma_si_value = self.sigmas[min(max(si, 0), len(self.sigmas) - 1)]
            else:
                sigma_si_value = self.sigmas[si]
                
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(sigma_si_value)
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(self, this_model_output: torch.Tensor, *args,
                                  last_sample: torch.Tensor = None, this_sample: torch.Tensor = None,
                                  order: int = None, **kwargs) -> torch.Tensor:
        """UniC (B(h) version) correction step."""
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError("missing `last_sample` as required argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError("missing `this_sample` as required argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError("missing `order` as required argument")
        if this_timestep is not None:
            deprecate("this_timestep", "1.0.0", "Passing `this_timestep` is deprecated")

        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        if self.is_resampling and self.resample_sigmas is not None:
            current_index = min(self.step_index, len(self.resample_sigmas) - 1)
            prev_index = min(max(self.step_index - 1, 0), len(self.resample_sigmas) - 1)
            
            sigma_t = self.resample_sigmas[current_index] if current_index < len(self.resample_sigmas) else self.sigmas[min(self.step_index, len(self.sigmas) - 1)]
            sigma_s0 = self.resample_sigmas[prev_index] if prev_index < len(self.resample_sigmas) else self.sigmas[min(max(self.step_index - 1, 0), len(self.sigmas) - 1)]
        else:
            sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
            
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            
            if self.is_resampling and self.resample_sigmas is not None:
                si_index = min(max(si, 0), len(self.resample_sigmas) - 1)
                sigma_si_value = self.resample_sigmas[si_index] if si_index < len(self.resample_sigmas) else self.sigmas[min(max(si, 0), len(self.sigmas) - 1)]
            else:
                sigma_si_value = self.sigmas[si]
                
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(sigma_si_value)
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
            
        x_t = x_t.to(x.dtype)
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = (schedule_timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    def _init_step_index(self, timestep):
        """Initialize step_index counter."""
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def fuse_latents(self, pred_original_sample, video_latents, mask, vae=None, static=False, **kwargs):
        """
        Fuse predicted sample with reference video in pixel space using guided channel selection.
        
        This is the core guidance mechanism that controls how reference video information
        is blended with model-generated content.
        
        Args:
            pred_original_sample: [B, C, T, H, W] predicted original sample (denoised x_0)
            video_latents: Reference video in pixel space
            mask: [B, 1, F, H, W] mask controlling guidance strength (1=use reference, 0=use generated)
            vae: VAE instance for encoding/decoding
            static: Whether to use static parameters
        """
        if mask is None or video_latents is None or vae is None:
            return pred_original_sample
            
        import logging
        logger = logging.getLogger(__name__)
        
        original_shape = pred_original_sample.shape
        original_device = pred_original_sample.device
        original_dtype = pred_original_sample.dtype
        
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(pred_original_sample.device, pred_original_sample.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            pred_original_sample.device, pred_original_sample.dtype
        )
        
        decoded_latents = pred_original_sample / latents_std + latents_mean
        decoded_latents = decoded_latents.to(dtype=torch.float32)
        
        try:
            decoded_video = vae.decode(decoded_latents, return_dict=False)[0]
        except RuntimeError:
            encode_chunk_size = 4
            decoded_chunks = []
            
            for i in range(0, decoded_latents.shape[2], encode_chunk_size):
                chunk = decoded_latents[:, :, i:i+encode_chunk_size, :, :].to(dtype=torch.float32)
                decoded_chunk = vae.decode(chunk, return_dict=False)[0]
                decoded_chunks.append(decoded_chunk)
                
            decoded_video = torch.cat(decoded_chunks, dim=2)
        
        target_shape = decoded_video.shape
        target_batch, target_channels, target_frames, target_height, target_width = target_shape
        
        if video_latents.shape != decoded_video.shape:
            import torch.nn.functional as F
            
            if video_latents.shape[0] != target_batch:
                video_latents = video_latents.repeat(target_batch, 1, 1, 1, 1)
            
            current_frames = video_latents.shape[2]
            current_height = video_latents.shape[3]
            current_width = video_latents.shape[4]
            
            if (current_frames != target_frames or 
                current_height != target_height or 
                current_width != target_width):
                
                batch_size, channels = video_latents.shape[0], video_latents.shape[1]
                
                if current_height != target_height or current_width != target_width:
                    video_reshaped = video_latents.reshape(batch_size * channels * current_frames, current_height, current_width)
                    video_spatial_resized = F.interpolate(
                        video_reshaped.unsqueeze(1),
                        size=(target_height, target_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    video_latents = video_spatial_resized.reshape(batch_size, channels, current_frames, target_height, target_width)
                
                if current_frames != target_frames:
                    video_time_reshaped = video_latents.reshape(batch_size * channels, current_frames, target_height, target_width)
                    video_time_resized = F.interpolate(
                        video_time_reshaped,
                        size=(target_frames, target_height, target_width),
                        mode='trilinear',
                        align_corners=False
                    )
                    video_latents = video_time_resized.reshape(target_shape)
        
        target_mask_shape = (target_batch, 1, target_frames, target_height, target_width)
        
        if mask.shape != target_mask_shape:
            if mask.shape[0] != target_batch:
                mask = mask.repeat(target_batch, 1, 1, 1, 1)
            
            if mask.shape[1] != 1:
                mask = mask[:, 0:1, :, :, :]
            
            current_mask_frames = mask.shape[2]
            current_mask_height = mask.shape[3]
            current_mask_width = mask.shape[4]
            
            if (current_mask_frames != target_frames or 
                current_mask_height != target_height or 
                current_mask_width != target_width):
                
                batch_size, channels = mask.shape[0], mask.shape[1]
                
                if current_mask_height != target_height or current_mask_width != target_width:
                    mask_reshaped = mask.reshape(batch_size * channels * current_mask_frames, current_mask_height, current_mask_width)
                    mask_spatial_resized = F.interpolate(
                        mask_reshaped.unsqueeze(1),
                        size=(target_height, target_width),
                        mode='nearest'
                    ).squeeze(1)
                    mask = mask_spatial_resized.reshape(batch_size, channels, current_mask_frames, target_height, target_width)
                
                if current_mask_frames != target_frames:
                    mask_time_reshaped = mask.reshape(batch_size * channels, current_mask_frames, target_height, target_width)
                    mask_time_resized = F.interpolate(
                        mask_time_reshaped,
                        size=(target_frames, target_height, target_width),
                        mode='nearest'
                    )
                    mask = mask_time_resized.reshape(target_mask_shape)
        
        video_latents = video_latents.to(decoded_video.device, decoded_video.dtype)
        mask = mask.to(decoded_video.device, decoded_video.dtype)
        video_latents = 2.0 * video_latents - 1.0
        
        if mask.shape[1] != decoded_video.shape[1]:
            mask = mask.repeat(1, decoded_video.shape[1], 1, 1, 1)
        
        fused_video = video_latents * mask + decoded_video * (1 - mask)
        fused_video = fused_video.to(dtype=torch.float32)
        
        try:
            encoded_video = vae.encode(fused_video).latent_dist.mode()
            encoded_video = (encoded_video - latents_mean) * latents_std
            
            if kwargs.get('use_pca_channel_selection') and not kwargs.get('resampling', False):
                try:
                    use_optical_flow = kwargs.get('use_optical_flow', True)
                    current_step = kwargs.get('current_step', 0)
                    total_steps = kwargs.get('total_steps', 50)
                    
                    if self._pca_selector is None:
                        self._pca_selector = VideoMotionPCASelector()
                        print("  [FLF] Flow-guided Latent Fusion channel selection enabled")
                    
                    encoded_video_aligned = encoded_video.to(pred_original_sample.device, pred_original_sample.dtype)
                    
                    channels_to_replace = self._pca_selector.select_motion_related_channels(
                        pred_original_sample=pred_original_sample,
                        video_latents=encoded_video_aligned,
                        mask=None,
                        keep_channels=12,
                        current_step=current_step,
                        total_steps=total_steps,
                        use_optical_flow=use_optical_flow,
                        static=static,
                    )
                    
                    for channel_idx in channels_to_replace:
                        if 0 <= channel_idx < encoded_video.shape[1]:
                            encoded_video[:, channel_idx, :, :, :] = pred_original_sample[:, channel_idx, :, :, :]
                    
                except Exception as e:
                    logger.error(f"FLF channel selection failed: {str(e)}")

            return encoded_video.to(original_device, original_dtype)
            
        except RuntimeError as e:
            logger.error(f"Video encoding failed: {str(e)}")
            return pred_original_sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        mask: Optional[torch.Tensor] = None,
        guided: bool = False,
        video_latents: Optional[torch.Tensor] = None,
        resampling: bool = False,
        vae: Optional[Any] = None,
        current_step: int = -1,
        resample_count: int = 2,
        is_resample_round: bool = False,
        static: bool = False,
        **kwargs,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict sample at previous timestep with optional video guidance.
        
        Args:
            model_output: Model output (noise prediction)
            timestep: Current timestep
            sample: Current sample
            mask: Guidance mask
            guided: Enable video guidance
            video_latents: Reference video for guidance
            resampling: Whether in resampling mode (r > 0)
            vae: VAE instance for pixel-space fusion
            current_step: Current step index
            resample_count: Number of resampling iterations
            is_resample_round: Whether in resample round
            static: Use static parameters
        """
        if self.num_inference_steps is None:
            raise ValueError("Run 'set_timesteps' after creating scheduler")

        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        
        if guided and video_latents is not None:
            fuse_kwargs = kwargs.copy()
            fuse_kwargs.update({
                'current_step': current_step,
                'total_steps': self.num_inference_steps if self.num_inference_steps else 50,
                'resampling': resampling,
                'static': static
            })
            model_output_convert = self.fuse_latents(model_output_convert, video_latents, mask, vae=vae, **fuse_kwargs)

        if not resampling:
            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
                self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order
            
        self.last_this_order = self.this_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0
        
        if not use_corrector:
            self.last_sample = sample
        if not is_resample_round:
            self.last_sample = sample
        if resample_count < 2:
            self.last_sample = sample
            
        if resampling:
            self.derivative_history.append(model_output)
  
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        self.last_lower_order_nums = self.lower_order_nums
        
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
            
        self._step_index += 1
        
        if not return_dict:
            return (prev_sample,)

        @dataclass
        class CustomSchedulerOutput:
            """Custom scheduler output with predicted x0."""
            prev_sample: torch.FloatTensor
            pred_x0: torch.FloatTensor
            
            def __getitem__(self, idx):
                if idx == 0:
                    return self.prev_sample
                elif idx == 1:
                    return self.pred_x0
                else:
                    raise IndexError(f"Invalid index {idx}")

        return CustomSchedulerOutput(prev_sample=prev_sample, pred_x0=model_output_convert)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Scale model input (identity for UniPC)."""
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
        r: int = 0,
        use_resample_sigma: bool = False,
    ) -> torch.Tensor:
        """Add noise to samples."""
        if use_resample_sigma and self.resample_sigmas is not None:
            sigmas = self.resample_sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
            schedule_timesteps = self.resample_timesteps.to(device=original_samples.device)
        else:
            sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
            schedule_timesteps = self.timesteps.to(original_samples.device)
            
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            timesteps = timesteps.to(original_samples.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            if use_resample_sigma:
                step_indices = [min(self.step_index, len(sigmas) - 1)] * timesteps.shape[0]
            else:
                step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        t_i = list(map(lambda x: x, step_indices))
        sigma_a = sigmas[t_i].flatten()
        while len(sigma_a.shape) < len(original_samples.shape):
            sigma_a = sigma_a.unsqueeze(-1)
        alpha_a, sigma_a = self._sigma_to_alpha_sigma_t(sigma_a)

        noisy_samples = (1-sigma_a)*original_samples + sigma_a * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
    
    def clear_motion_cache(self):
        """Clear motion analysis cache for new inference."""
        self._pca_selector = None
        
    def _compute_resample_sigmas_and_timesteps(self):
        """Compute interpolated sigma values for resampling."""
        if len(self.sigmas) < 2:
            self.resample_sigmas = None
            self.resample_timesteps = None
            return
            
        resample_sigmas = []
        for i in range(len(self.sigmas) - 1):
            sigma_curr = self.sigmas[i]
            resample_sigmas.append(sigma_curr)
        
        self.resample_sigmas = torch.stack(resample_sigmas) if resample_sigmas else None
        
        if self.resample_sigmas is not None:
            resample_timesteps = []
            full_sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
            
            for sigma in self.resample_sigmas:
                if self.config.use_flow_sigmas:
                    timestep = sigma * self.config.num_train_timesteps
                    timestep_int = torch.floor(timestep).to(torch.int64)
                else:
                    try:
                        sigma_np = sigma.cpu().numpy()
                        log_sigmas = np.log(np.maximum(full_sigmas.numpy(), 1e-10))
                        timestep_float = self._sigma_to_t(sigma_np, log_sigmas)
                        timestep_int = torch.round(torch.tensor(timestep_float)).to(torch.int64)
                    except:
                        distances = torch.abs(full_sigmas - sigma)
                        closest_idx = torch.argmin(distances)
                        timestep_int = torch.tensor(self.config.num_train_timesteps - 1 - closest_idx.item()).to(torch.int64)
                
                resample_timesteps.append(timestep_int)
            
            self.resample_timesteps = torch.stack(resample_timesteps)

    def set_resample_mode(self, enabled: bool):
        """Set resampling mode."""
        if enabled and not self.is_resampling:
            self.original_step_index = self.step_index
        self.is_resampling = enabled
        if not enabled and self.original_step_index is not None:
            self._step_index = self.original_step_index
            self.original_step_index = None
    
    def get_resample_timestep(self, step_index: int) -> torch.Tensor:
        """Get resampling timestep."""
        if self.resample_timesteps is not None and step_index < len(self.resample_timesteps):
            timestep = self.resample_timesteps[step_index]
            timestep = timestep.to(device=self.timesteps.device, dtype=self.timesteps.dtype)
            return timestep
        else:
            fallback_index = min(step_index, len(self.timesteps) - 1)
            return self.timesteps[fallback_index]
