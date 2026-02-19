# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Optical flow-based channel selector (streamlined version)
class VideoMotionChannelSelector:
    """
    Optical flow-based channel selector for motion-guided latent channel filtering.
    Compares motion similarity between predicted and reference video latents.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._check_cv2()
    
    def _check_cv2(self):
        """Check OpenCV availability."""
        try:
            import cv2
            self.use_optical_flow = True
        except ImportError:
            self.use_optical_flow = False
    
    def _sanitize_data(self, data: torch.Tensor, name: str = "data") -> torch.Tensor:
        """Clean NaN and Inf values."""
        if not torch.isfinite(data).all():
            finite_mask = torch.isfinite(data)
            if finite_mask.any():
                finite_mean = data[finite_mask].mean()
                data = torch.where(torch.isfinite(data), data, finite_mean)
            else:
                data = torch.zeros_like(data)
        return data
    
    def temporal_spatial_align(self, video: torch.Tensor, target_frames: int, target_spatial: Tuple[int, int]) -> torch.Tensor:
        """Align video to target dimensions."""
        batch_size, channels, frames, height, width = video.shape
        
        original_dtype = video.dtype
        video = video.to(dtype=torch.float32)
        video = self._sanitize_data(video, "input video")
        
        is_mask = (channels == 1)
        
        # Temporal alignment
        if frames != target_frames:
            video_permuted = video.permute(0, 1, 3, 4, 2)  # [1, C, H, W, T]
            video_reshaped = video_permuted.reshape(batch_size * channels * height * width, frames)
            
            aligned_frames = F.interpolate(
                video_reshaped.unsqueeze(1), 
                size=(target_frames,), 
                mode='linear', 
                align_corners=True if target_frames > 1 else False
            ).squeeze(1)
            
            video = aligned_frames.reshape(batch_size, channels, height, width, target_frames)
            video = video.permute(0, 1, 4, 2, 3)  # [1, C, target_frames, H, W]
        
        # Spatial alignment
        if (height, width) != target_spatial:
            video_reshaped = video.reshape(batch_size * channels * target_frames, height, width)
            interpolation_mode = 'nearest' if is_mask else 'bilinear'
            
            video_resized = F.interpolate(
                video_reshaped.unsqueeze(1),
                size=target_spatial,
                mode=interpolation_mode,
                align_corners=False if interpolation_mode == 'bilinear' else None
            ).squeeze(1)
            video = video_resized.reshape(batch_size, channels, target_frames, target_spatial[0], target_spatial[1])
        
        video = video.to(dtype=original_dtype)
        return video
    
    def _extract_optical_flow(self, video: torch.Tensor) -> torch.Tensor:
        """Extract optical flow from video"""
        try:
            import cv2
            
            video = video.to(dtype=torch.float32)
            batch_size, channels, frames, height, width = video.shape
            
            # Convert to numpy  
            video_np = video.squeeze(0).float().cpu().numpy()  # [C, T, H, W]
            video_np = video_np.transpose(1, 2, 3, 0)  # [T, H, W, C]
            
            # Normalize to [0, 255]
            if video_np.min() >= -1.1 and video_np.max() <= 1.1:
                video_np = ((video_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            else:
                video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
            
            flow_list = []
            for t in range(frames - 1):
                frame1 = video_np[t]
                frame2 = video_np[t + 1]
                
                # Convert to grayscale
                if frame1.shape[2] == 3:
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                else:
                    gray1 = frame1.squeeze(-1) if frame1.ndim == 3 else frame1
                    gray2 = frame2.squeeze(-1) if frame2.ndim == 3 else frame2
                
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        gray1, gray2, None, 
                        pyr_scale=0.5, levels=3, winsize=15, 
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                except Exception as e:
                    logger.warning(f"Optical flow failed at frame {t}: {e}, using gradient fallback")
                    # Fallback to gradient
                    diff = frame2.astype(np.float32) - frame1.astype(np.float32)
                    grad_x = cv2.Sobel(diff.mean(axis=2) if diff.ndim == 3 else diff, cv2.CV_32F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(diff.mean(axis=2) if diff.ndim == 3 else diff, cv2.CV_32F, 0, 1, ksize=3)
                    flow = np.stack([grad_x, grad_y], axis=2)
                
                flow_list.append(flow)
            
            flows = np.stack(flow_list, axis=0)  # [T-1, H, W, 2]
            flows = flows.transpose(0, 3, 1, 2)  # [T-1, 2, H, W]
            flows_tensor = torch.from_numpy(flows).float().unsqueeze(0)  # [1, T-1, 2, H, W]
            
            flows_tensor = flows_tensor.to(device=video.device, dtype=video.dtype)
            flows_tensor = self._sanitize_data(flows_tensor, "optical flow")
            
            return flows_tensor
        
        except Exception as e:
            logger.error(f"Optical flow extraction failed: {e}, using temporal difference fallback")
            return self._extract_temporal_difference(video)
    
    def _extract_temporal_difference(self, video: torch.Tensor) -> torch.Tensor:
        """Extract temporal difference as motion proxy"""
        # Simple temporal difference: [1, C, T, H, W] -> [1, T-1, C, H, W]
        motion = video[:, :, 1:] - video[:, :, :-1]
        motion = motion.permute(0, 2, 1, 3, 4)  # [1, T-1, C, H, W]
        return motion
    
    def _compute_flow_metrics(self, ref_flow: torch.Tensor, cand_flow: torch.Tensor) -> float:
        """
        Compute optical flow similarity metrics: M-EPE, M-AE, Fl-all
        """
        try:
            device = ref_flow.device
            ref_flow = ref_flow.float().to(device)
            cand_flow = cand_flow.float().to(device)
            
            # Align dimensions
            min_frames = min(ref_flow.shape[1], cand_flow.shape[1])
            if min_frames <= 0:
                return 0.0
            
            ref_aligned = ref_flow[:, :min_frames]
            cand_aligned = cand_flow[:, :min_frames]
            
            # Spatial alignment
            if ref_aligned.shape[-2:] != cand_aligned.shape[-2:]:
                cand_aligned = F.interpolate(
                    cand_aligned.view(-1, cand_aligned.shape[-2], cand_aligned.shape[-1]),
                    size=ref_aligned.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).view(cand_aligned.shape[0], cand_aligned.shape[1], cand_aligned.shape[2], 
                       ref_aligned.shape[-2], ref_aligned.shape[-1])
            
            # Extract flow channels (u, v)
            if ref_aligned.shape[2] >= 2:
                ref_flow_uv = ref_aligned[:, :, :2]
            else:
                ref_flow_uv = ref_aligned.repeat(1, 1, 2, 1, 1)[:, :, :2]
            
            if cand_aligned.shape[2] >= 2:
                cand_flow_uv = cand_aligned[:, :, :2]
            else:
                cand_flow_uv = cand_aligned.repeat(1, 1, 2, 1, 1)[:, :, :2]
            
            # 1. M-EPE (End Point Error)
            flow_diff = ref_flow_uv - cand_flow_uv
            epe = torch.sqrt((flow_diff ** 2).sum(dim=2) + 1e-8)  # [1, T, H, W]
            
            # 2. M-AE (Angle Error)
            dot_product = (ref_flow_uv * cand_flow_uv).sum(dim=2)
            ref_norm = torch.sqrt((ref_flow_uv ** 2).sum(dim=2) + 1e-8)
            cand_norm = torch.sqrt((cand_flow_uv ** 2).sum(dim=2) + 1e-8)
            
            cos_angle = dot_product / (ref_norm * cand_norm + 1e-8)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            angle_error = torch.acos(cos_angle) * 180.0 / torch.pi
            
            # 3. Fl-all (Outlier percentage)
            threshold_abs = 3.0
            threshold_rel = 0.05
            outlier_mask = (epe > threshold_abs) | (epe > ref_norm * threshold_rel)
            
            m_epe = epe.mean()
            m_ae = angle_error.mean()
            fl_all = outlier_mask.float().mean()
            
            # Normalize and compute similarity
            norm_epe = torch.clamp(m_epe / 10.0, 0.0, 1.0)
            norm_fl = torch.clamp(fl_all / 0.5, 0.0, 1.0)
            norm_ae = torch.clamp(m_ae / 30.0, 0.0, 1.0)
            
            weighted_error = 0.4 * norm_epe + 0.4 * norm_fl + 0.2 * norm_ae
            similarity = 1.0 - weighted_error
            similarity = torch.clamp(similarity, 0.0, 1.0)
            
            return similarity.item()
        except Exception as e:
            return 0.0
    
    def select_motion_related_channels(
        self,
        pred_original_sample: torch.Tensor,
        encoded_video: torch.Tensor,
        current_step: int = 0,
        total_steps: int = 50,
        static: bool = False,
        use_distill: bool = False,
        max_replace_threshold: Optional[int] = None,
    ) -> List[int]:
        """
        Select channels to replace based on motion similarity.
        
        Args:
            pred_original_sample: [1, 16, T, H, W] - predicted latents
            encoded_video: [1, 16, T, H, W] - reference encoded video
            current_step: current sampling step
            total_steps: total sampling steps
            static: static scene mode
            use_distill: whether using distillation mode (16-step sampling)
        
        Returns:
            List of channel indices to replace
        """
        # Skip early steps
        if current_step < 2:
            return []
        
        # Validate dimensions
        if pred_original_sample.dim() != 5 or encoded_video.dim() != 5:
            return []
        
        device = pred_original_sample.device
        encoded_video = encoded_video.to(device, dtype=pred_original_sample.dtype)
        
        # Compute motion similarity for each channel
        correlations = []
        for ch_idx in range(pred_original_sample.shape[1]):
            try:
                pred_channel = pred_original_sample[:, ch_idx:ch_idx+1]  # [1, 1, T, H, W]
                ref_channel = encoded_video[:, ch_idx:ch_idx+1]  # [1, 1, T, H, W]
                
                # Extract motion features
                if self.use_optical_flow:
                    try:
                        # For optical flow, expand to 3 channels
                        pred_rgb = pred_channel.repeat(1, 3, 1, 1, 1)
                        pred_rgb = (pred_rgb - pred_rgb.min()) / (pred_rgb.max() - pred_rgb.min() + 1e-8)
                        pred_motion = self._extract_optical_flow(pred_rgb)
                        
                        ref_rgb = ref_channel.repeat(1, 3, 1, 1, 1)
                        ref_rgb = (ref_rgb - ref_rgb.min()) / (ref_rgb.max() - ref_rgb.min() + 1e-8)
                        ref_motion = self._extract_optical_flow(ref_rgb)
                        
                        # Clean up
                        del pred_rgb, ref_rgb
                    except Exception as e:
                        logger.warning(f"Optical flow failed for channel {ch_idx}: {e}")
                        # Fallback to temporal difference
                        pred_motion = self._extract_temporal_difference(pred_channel.to(torch.float32))
                        ref_motion = self._extract_temporal_difference(ref_channel.to(torch.float32))
                else:
                    pred_motion = self._extract_temporal_difference(pred_channel.to(torch.float32))
                    ref_motion = self._extract_temporal_difference(ref_channel.to(torch.float32))
                
                # Compute similarity
                similarity = self._compute_flow_metrics(ref_motion, pred_motion)
                correlations.append(similarity)
                
                # Clean up
                del pred_channel, ref_channel, pred_motion, ref_motion
                
                # Force GPU sync every few channels to avoid accumulation
                if (ch_idx + 1) % 4 == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
            except Exception as e:
                correlations.append(0.0)
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if len(correlations) == 0:
            return []
        
        # Dynamic thresholding strategy
        channel_correlations = np.array(correlations)
        corr_mean = np.mean(channel_correlations)
        corr_std = np.std(channel_correlations)
        
        # Adaptive strategy based on step and distillation mode
        if use_distill:
            if current_step <= 3:
                max_replace = 1
                channels_to_replace = np.argsort(channel_correlations)[:max_replace].tolist()
            else:
                scale_std = 0.625
                max_replace = max_replace_threshold if max_replace_threshold is not None else 3
                min_replace = 1
                
                threshold = corr_mean - scale_std * corr_std
                below_threshold = [i for i, score in enumerate(channel_correlations) if score < threshold]
                
                if len(below_threshold) < min_replace:
                    channels_to_replace = np.argsort(channel_correlations)[:min_replace].tolist()
                elif len(below_threshold) > max_replace:
                    below_scores = [(idx, channel_correlations[idx]) for idx in below_threshold]
                    below_scores.sort(key=lambda x: x[1])
                    channels_to_replace = [idx for idx, _ in below_scores[:max_replace]]
                else:
                    channels_to_replace = below_threshold
        else:
            if current_step <= 5:
                max_replace = 1
                channels_to_replace = np.argsort(channel_correlations)[:max_replace].tolist()
            else:
                scale_std = 0.625
                max_replace = max_replace_threshold if max_replace_threshold is not None else 1
                min_replace = 1
                
                threshold = corr_mean - scale_std * corr_std
                below_threshold = [i for i, score in enumerate(channel_correlations) if score < threshold]
                
                if len(below_threshold) < min_replace:
                    channels_to_replace = np.argsort(channel_correlations)[:min_replace].tolist()
                elif len(below_threshold) > max_replace:
                    below_scores = [(idx, channel_correlations[idx]) for idx in below_threshold]
                    below_scores.sort(key=lambda x: x[1])
                    channels_to_replace = [idx for idx, _ in below_scores[:max_replace]]
                else:
                    channels_to_replace = below_threshold
        
        channels_to_replace = sorted(channels_to_replace)
        
        return channels_to_replace


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_x0 (`torch.FloatTensor`, optional):
            Predicted original sample (x_0) for resampling.
    """

    prev_sample: torch.FloatTensor
    pred_x0: Optional[torch.FloatTensor] = None


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation. Increasing `base_shift` reduces variation and image is more consistent
            with desired output.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors. Increasing `max_shift` encourages more variation and image may be
            more exaggerated or stylized.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to False):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to None):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to False):
            Whether to use Karras sigmas for step sizes in the noise schedule during sampling.
        use_exponential_sigmas (`bool`, defaults to False):
            Whether to use exponential sigmas for step sizes in the noise schedule during sampling.
        use_beta_sigmas (`bool`, defaults to False):
            Whether to use beta sigmas for step sizes in the noise schedule during sampling.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply. Either "exponential" or "linear".
        stochastic_sampling (`bool`, defaults to False):
            Whether to use stochastic sampling.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self._shift = shift

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        
        # Resample-related attributes
        self.derivative_history = []
        self.resample_sigmas = None
        self.resample_timesteps = None
        self.is_resampling = False
        self._channel_selector = None

    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return self._shift

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Stretches and shifts the timestep schedule to ensure it terminates at the configured `shift_terminal` config
        value.

        Reference:
        https://github.com/Lightricks/LTX-Video/blob/a01a171f8fe3d99dce2728d60a73fecf4d4238ae/ltx_video/schedulers/rf.py#L51

        Args:
            t (`torch.Tensor`):
                A tensor of timesteps to be stretched and shifted.

        Returns:
            `torch.Tensor`:
                A tensor of adjusted timesteps such that the final value equals `self.config.shift_terminal`.
        """
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `None`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `None`, the timesteps are computed
                automatically.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")

        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        # 1. Prepare default sigmas
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            timesteps = np.array(timesteps).astype(np.float32)

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
                )
            sigmas = timesteps / self.config.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)

        # 2. Perform timestep shifting. Either no shifting is applied, or resolution-dependent shifting of
        #    "exponential" or "linear" type is applied
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # 3. If required, stretch the sigmas schedule to terminate at the configured `shift_terminal` value
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        # 4. If required, convert sigmas to one of karras, exponential, or beta sigma schedules
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # 5. Convert sigmas and timesteps to tensors and move to specified device
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            timesteps = sigmas * self.config.num_train_timesteps
        else:
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)

        # 6. Append the terminal sigma value.
        #    If a model requires inverted sigma schedule for denoising but timesteps without inversion, the
        #    `invert_sigmas` flag can be set to `True`. This case is only required in Mochi
        if self.config.invert_sigmas:
            sigmas = 1.0 - sigmas
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        else:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None
        
        # Compute resample sigmas and timesteps
        self._compute_resample_sigmas_and_timesteps()
        
        # Clear derivative history for new generation
        self.derivative_history = []

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # Resample and guidance parameters
        video_ref: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        guided: bool = False,
        resampling: bool = False,
        vae: Optional[Any] = None,
        use_pca_channel_selection: bool = False,
        static: bool = False,
        current_step: int = -1,
        total_steps: int = 50,
        sample_full: Optional[torch.FloatTensor] = None,
        use_distill: bool = False,
        max_replace_threshold: Optional[int] = None,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.
            video_ref (`torch.Tensor`, *optional*):
                Reference video for guidance.
            mask (`torch.Tensor`, *optional*):
                Mask for guided fusion.
            guided (`bool`, defaults to False):
                Enable guided generation.
            resampling (`bool`, defaults to False):
                Whether this is a resample step.
            vae (`Any`, *optional*):
                VAE for encoding/decoding.
            use_pca_channel_selection (`bool`, defaults to False):
                Enable optical flow channel selection.
            static (`bool`, defaults to False):
                Static scene mode.
            current_step (`int`, defaults to -1):
                Current sampling step index.
            total_steps (`int`, defaults to 50):
                Total sampling steps.
        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma
        
        # Compute pred_x0 for resampling
        # Flow-matching: x_t = (1 - sigma_t) * x_0 + sigma_t * noise
        # Therefore: x_0 = (x_t - sigma_t * model_output) / (1 - sigma_t)
        # Simplified: x_0 ≈ sample - current_sigma * model_output
        pred_x0 = sample - current_sigma * model_output
        
        # Apply guided fusion if enabled and not resampling
        if guided and video_ref is not None and not resampling and sample_full is not None:
            try:
                # Compute pred_x0_full for the full latents (including first frame)
                pred_x0_full = sample_full - current_sigma * torch.cat([
                    torch.zeros_like(model_output[:, :, 0:1]),  # First frame is condition, no noise pred
                    model_output
                ], dim=2)
                
                # Fuse with full latents
                pred_x0_full_fused = self.fuse_latents(
                    pred_original_sample=pred_x0_full,
                    video_latents=video_ref,
                    mask=mask,
                    vae=vae,
                    use_pca_channel_selection=use_pca_channel_selection,
                    static=static,
                    current_step=current_step,
                    total_steps=total_steps,
                    max_replace_threshold=max_replace_threshold,
                    use_distill=use_distill,
                )
                
                # Extract the fused part (excluding first frame) for diffusion
                pred_x0 = pred_x0_full_fused[:, :, 1:, :, :]
                
            except Exception as e:
                logger.error(f"fuse_latents failed: {e}")
                logger.error(f"Falling back to original pred_x0")
                import traceback
                traceback.print_exc()
                # Continue with original pred_x0 on error
        
        # Record for auto-guidance
        self.derivative_history.append(model_output)

        if self.config.stochastic_sampling:
            noise = torch.randn_like(sample)
            prev_sample = (1.0 - next_sigma) * pred_x0 + next_sigma * noise
            #print(f"[Scheduler Step {self._step_index}] Mode: STOCHASTIC | next_sigma={next_sigma:.4f} | formula: (1-σ)*pred_x0 + σ*noise")
        else:
            prev_sample = sample + dt * model_output
            #print(f"[Scheduler Step {self._step_index}] Mode: DETERMINISTIC (Euler) | dt={dt:.4f} | formula: sample + dt*model_output")

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_x0=pred_x0)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_exponential
    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Constructs an exponential noise schedule."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_beta
    def _convert_to_beta(
        self, in_sigmas: torch.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> torch.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def _time_shift_exponential(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu, sigma, t):
        return mu / (mu + (1 / t - 1) ** sigma)

    def __len__(self):
        return self.config.num_train_timesteps
    
    # ========== Resample-related methods ==========
    
    def _compute_resample_sigmas_and_timesteps(self):
        """
        Compute interpolated sigmas for resampling: sigma_resample = sigma_t
        Note: May need adjustment based on Euler's sigma range (0-1 vs other ranges)
        """
        if len(self.sigmas) < 2:
            self.resample_sigmas = None
            self.resample_timesteps = None
            return
        
        # Interpolated sigma: keep current sigma for resample (can be adjusted to sqrt(sigma_t * sigma_next))
        resample_sigmas = []
        for i in range(len(self.sigmas) - 1):
            sigma_curr = self.sigmas[i]
            # TODO: May need sqrt(sigma_curr * sigma_next) for better interpolation
            sigma_resample = sigma_curr
            resample_sigmas.append(sigma_resample)
        
        self.resample_sigmas = torch.stack(resample_sigmas) if resample_sigmas else None
        
        # Compute corresponding timesteps
        if self.resample_sigmas is not None:
            self.resample_timesteps = self.resample_sigmas * self.config.num_train_timesteps
    
    def set_resample_mode(self, enabled: bool):
        """Toggle resample mode"""
        self.is_resampling = enabled
    
    def get_resample_timestep(self, step_index: int) -> torch.Tensor:
        """Get interpolated timestep for resampling"""
        if self.resample_timesteps is not None and step_index < len(self.resample_timesteps):
            return self.resample_timesteps[step_index]
        else:
            fallback_index = min(step_index, len(self.timesteps) - 1)
            return self.timesteps[fallback_index]
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        use_resample_sigma: bool = False,
    ) -> torch.Tensor:
        """
        Add noise to samples for resampling.
        Flow-matching forward process: x_t = (1 - sigma_t) * x_0 + sigma_t * noise
        """
        # Always use timesteps to compute sigma, not step_index
        # because step_index may have been incremented by step()
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        
        timesteps = timesteps.to(original_samples.device)
        
        # Get step indices from timesteps directly
        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        # sigma = sigma * 0.995
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        
        # Flow-matching noise schedule
        noisy_samples = (1.0 - sigma) * original_samples + sigma * noise
        
        return noisy_samples
    
    def fuse_latents(
        self,
        pred_original_sample: torch.Tensor,
        video_latents: torch.Tensor,
        mask: torch.Tensor,
        vae: Any,
        use_pca_channel_selection: bool = False,
        static: bool = False,
        current_step: int = 0,
        total_steps: int = 50,
        use_distill: bool = False,
        max_replace_threshold: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Fuse predicted latents with reference video in pixel space.
        Applies optical flow-based channel selection if enabled.
        
        Args:
            pred_original_sample: [1, 16, T_latent, H, W] - predicted x_0 (FULL, including first frame)
            video_latents: [1, 3, T_pixel, H*8, W*8] - reference video (pixel space)
            mask: [1, 1, T_pixel, H*8, W*8] - guidance mask
            vae: VAE for encoding/decoding
            use_pca_channel_selection: enable optical flow channel selection
            static: static scene mode
            current_step: current sampling step
            total_steps: total sampling steps
            use_distill: whether using distillation mode (16-step sampling)
        
        Returns:
            Fused latents with selective channel replacement
        """
        if mask is None or video_latents is None or vae is None:
            return pred_original_sample
        
        try:
            original_device = pred_original_sample.device
            original_dtype = pred_original_sample.dtype
            
            logger.debug(f"fuse_latents: pred_x0={pred_original_sample.shape}, video_ref={video_latents.shape}, mask={mask.shape}")
            
            # 1. Decode pred_original_sample to pixel space (FULL latents including first frame)
            
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(pred_original_sample.device, pred_original_sample.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                pred_original_sample.device, pred_original_sample.dtype
            )
            
            decoded_latents = pred_original_sample / latents_std + latents_mean
            decoded_latents = decoded_latents.to(dtype=vae.dtype)
            
            # Decode to pixel space (full latents including first frame)
            decoded_video = vae.decode(decoded_latents, return_dict=False)[0]
            logger.debug(f"Decoded: latent {decoded_latents.shape} -> pixel {decoded_video.shape}")
            
            # 2. Check dimensions match (no interpolation, direct error if mismatch)
            if video_latents.shape != decoded_video.shape:
                error_msg = (
                    f"Dimension mismatch! "
                    f"decoded_video={decoded_video.shape}, "
                    f"video_ref={video_latents.shape}. "
                    f"Expected exact match."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if mask.shape[1] != 1 or mask.shape[2] != decoded_video.shape[2] or \
               mask.shape[3] != decoded_video.shape[3] or mask.shape[4] != decoded_video.shape[4]:
                error_msg = (
                    f"Mask dimension mismatch! "
                    f"mask={mask.shape}, "
                    f"expected=[{decoded_video.shape[0]}, 1, {decoded_video.shape[2]}, "
                    f"{decoded_video.shape[3]}, {decoded_video.shape[4]}]"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 3. Fuse in pixel space
            video_latents = video_latents.to(decoded_video.device, decoded_video.dtype)
            mask = mask.to(decoded_video.device, decoded_video.dtype)
            
            # Map video_latents from [0, 1] to [-1, 1]
            video_latents = 2.0 * video_latents - 1.0
            
            # Expand mask to match channels
            if mask.shape[1] != decoded_video.shape[1]:
                mask = mask.repeat(1, decoded_video.shape[1], 1, 1, 1)
            
            # Pixel-space fusion
            fused_video = video_latents * mask + decoded_video * (1 - mask)
            # 4. Re-encode to latent space
            fused_video = fused_video.to(dtype=vae.dtype)
            
            # Encode all frames at once
            encoded_video = vae.encode(fused_video).latent_dist.mode()
            logger.debug(f"Encoded: pixel {fused_video.shape} -> latent {encoded_video.shape}")
            
            # Check encode result matches original shape
            if encoded_video.shape != pred_original_sample.shape:
                error_msg = (
                    f"Encode output shape mismatch! "
                    f"encoded={encoded_video.shape}, "
                    f"expected={pred_original_sample.shape}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Normalize
            encoded_video = (encoded_video - latents_mean) * latents_std
            
            # Apply Flow Latent Filtering (FLF) channel selection
            if use_pca_channel_selection:
                
                try:
                    # Initialize channel selector
                    if self._channel_selector is None:
                        self._channel_selector = VideoMotionChannelSelector(debug=False)
                    
                    # Select channels to replace
                    channels_to_replace = self._channel_selector.select_motion_related_channels(
                        pred_original_sample=pred_original_sample,
                        encoded_video=encoded_video,
                        current_step=current_step,
                        total_steps=total_steps,
                        static=static,
                        use_distill=use_distill,
                        max_replace_threshold=max_replace_threshold,
                    )
                    
                    # Replace selected channels
                    for ch_idx in channels_to_replace:
                        if 0 <= ch_idx < encoded_video.shape[1]:
                            encoded_video[:, ch_idx, :, :, :] = pred_original_sample[:, ch_idx, :, :, :]
                    
                    logger.debug(f"FLF: replaced {len(channels_to_replace)} channels")
                    
                    # Clean up GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Channel selection failed: {e}, skipping channel replacement")
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return encoded_video.to(original_device, original_dtype)
        
        except Exception as e:
            logger.error(f"fuse_latents complete failure: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return pred_original_sample
    
