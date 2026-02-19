# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

import numpy as np

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> import numpy as np
        >>> from diffusers import AutoencoderKLWan
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import CLIPVisionModel
        >>> from utils.pipeline_wan_i2v_clean import WanImageToVideoPipeline

        >>> model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        >>> image_encoder = CLIPVisionModel.from_pretrained(
        ...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
        ... )
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanImageToVideoPipeline.from_pretrained(
        ...     model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image("path/to/image.jpg")
        >>> max_area = 480 * 832
        >>> aspect_ratio = image.height / image.width
        >>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        >>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        >>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        >>> image = image.resize((width, height))
        
        >>> prompt = "High quality video with smooth camera motion"
        >>> negative_prompt = "Low quality, blurry, distorted"

        >>> output = pipe(
        ...     image=image,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, 
                    sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of encoder_output")


class WanImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan with optical flow guided generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods.

    Args:
        tokenizer: T5Tokenizer from google/umt5-xxl
        text_encoder: T5EncoderModel from google/umt5-xxl
        image_encoder: CLIPVisionModel (clip-vit-huge-patch14)
        transformer: WanTransformer3DModel for denoising
        scheduler: UniPCMultistepScheduler
        vae: AutoencoderKLWan for encoding/decoding
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(self, image: PipelineImageInput, device: Optional[torch.device] = None):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Encode prompt into text encoder hidden states."""
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` type mismatch with `prompt`: {type(negative_prompt)} != {type(prompt)}")
            elif batch_size != len(negative_prompt):
                raise ValueError(f"`negative_prompt` batch size {len(negative_prompt)} != prompt batch size {batch_size}")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError("Cannot forward both `image` and `image_embeds`")
        if image is None and image_embeds is None:
            raise ValueError("Must provide either `image` or `image_embeds`")
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` must be torch.Tensor or PIL.Image, got {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(f"`callback_on_step_end_tensor_inputs` must be in {self._callback_tensor_inputs}")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`")
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `prompt` or `prompt_embeds`")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` must be str or list, got {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` must be str or list, got {type(negative_prompt)}")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} != batch size {batch_size}")

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
        )
        video_condition = video_condition.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        video_ref: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        guided: bool = False,
        resample_steps: int = 1,
        guide_steps: int = 20,
        omega: float = 1.8,
        omega_resample: float = 1.0,
        resample_round: int = 20,
        use_pca_channel_selection: bool = False,
        static: bool = False,
    ):
        r"""
        Generate video from image using optical flow guided generation.

        Args:
            image: Input image to condition generation
            prompt: Text prompt to guide generation
            negative_prompt: Negative text prompt
            height: Video height (default: 480)
            width: Video width (default: 832)
            num_frames: Number of frames to generate (default: 81)
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: Classifier-free guidance scale (default: 5.0)
            num_videos_per_prompt: Number of videos per prompt (default: 1)
            generator: Random generator for reproducibility
            latents: Pre-generated noisy latents
            prompt_embeds: Pre-generated text embeddings
            negative_prompt_embeds: Pre-generated negative text embeddings
            image_embeds: Pre-generated image embeddings
            output_type: Output format ("pil" or "np")
            return_dict: Whether to return WanPipelineOutput
            attention_kwargs: Additional attention processor kwargs
            callback_on_step_end: Callback function for each step
            callback_on_step_end_tensor_inputs: Tensor inputs for callback
            max_sequence_length: Maximum sequence length for text
            video_ref: Reference video for guided generation
            mask: Mask for guidance (1=use reference, 0=use generated)
            guided: Enable guided generation
            resample_steps: Number of resampling iterations (default: 1)
            guide_steps: Number of steps to apply guidance (default: 20)
            omega: Auto-guidance parameter (default: 1.8)
            omega_resample: Resampling auto-guidance parameter (default: 1.0)
            resample_round: Number of rounds for resampling (default: 20)
            use_pca_channel_selection: Enable PCA-based optical flow channel selection (FLF)
            static: Use static parameters

        Returns:
            WanPipelineOutput or tuple with generated video frames

        Examples:
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt, negative_prompt, image, height, width,
            prompt_embeds, negative_prompt_embeds, image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(f"`num_frames - 1` must be divisible by {self.vae_scale_factor_temporal}, rounding")
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        latents, condition = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        latents_original = latents.clone()
        
        if video_ref is not None and guided:
            if not isinstance(video_ref, torch.Tensor):
                video_ref = torch.tensor(video_ref)
            if video_ref.dtype != torch.float32:
                video_ref = video_ref.to(dtype=torch.float32)
            video_ref = video_ref.to(device=device)

        if mask is not None and guided:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            if mask.dim() == 3:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 4 and mask.shape[1] != 1:
                mask = mask[:, 0:1, :, :].unsqueeze(0)
            elif mask.dim() == 4 and mask.shape[0] == 1:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 5 and mask.shape[1] != 1:
                mask = mask[:, 0:1, :, :, :]
            mask = mask.to(device)
            if mask.shape[1] != 1:
                mask = mask[:, 0:1, :, :, :]
        
        if not hasattr(self.scheduler, "derivative_history"):
            self.scheduler.derivative_history = []

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                self.scheduler.derivative_history = []
                
                round_idx = 0
                saved_noise_pred = None
                pred_original_sample = None
                
                for r in range(resample_steps):
                    if r > 0:
                        self.scheduler.set_resample_mode(True)
                        resample_timestep = self.scheduler.get_resample_timestep(i)
                        timestep_for_transformer = resample_timestep.expand(latents.shape[0])
                        timestep_for_transformer = timestep_for_transformer.to(device=latents.device)
                    else:
                        self.scheduler.set_resample_mode(False)
                        timestep_for_transformer = t.expand(latents.shape[0])
                        
                    if r > 0:
                        self.scheduler._step_index -= 1
                        if self.scheduler.lower_order_nums > 0 and self.scheduler.last_lower_order_nums < self.scheduler.config.solver_order:
                            self.scheduler.lower_order_nums -= 1
                        self.scheduler.this_order = self.scheduler.last_this_order
                        
                    self._current_timestep = t
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)

                    if round_idx != 10:
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep_for_transformer,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        
                        if self.do_classifier_free_guidance:
                            noise_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep_for_transformer,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_image=image_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                            noise_pred = noise_pred + guidance_scale * (noise_pred - noise_uncond)
                            
                            if r < 1:
                                self.scheduler.derivative_history.append(noise_pred)
                        saved_noise_pred = noise_pred
                    else:
                        noise_pred = saved_noise_pred
                    
                    scheduler_output = self.scheduler.step(
                        noise_pred, 
                        t, 
                        latents, 
                        mask=mask,
                        guided=guided and i < guide_steps and r < resample_steps,
                        video_latents=video_ref,
                        vae=self.vae,
                        resampling=r > 0,
                        return_dict=True,
                        current_step=i,
                        resample_count=resample_steps,
                        is_resample_round=i<resample_round,
                        use_pca_channel_selection=use_pca_channel_selection,
                        static=static,
                    )
                    
                    if hasattr(scheduler_output, "pred_x0"):
                        pred_original_sample = scheduler_output.pred_x0
                    
                    if i >= resample_round:
                        break
                    
                    if r < resample_steps - 1 and pred_original_sample is not None:
                        if generator is not None:
                            noise = torch.randn(pred_original_sample.shape, generator=generator)
                            noise = noise.to(device=device)
                        else:
                            noise = torch.randn(pred_original_sample.shape, device=device)
                        
                        timestep_for_noise = self.scheduler.get_resample_timestep(i)
                        if timestep_for_noise.dim() == 0:
                            timestep_for_noise = timestep_for_noise.unsqueeze(0)
                        timestep_for_noise = timestep_for_noise.to(device=latents.device)
                        
                        latents = self.scheduler.add_noise(
                            pred_original_sample,
                            noise,
                            timestep_for_noise,
                            r,
                            use_resample_sigma=True
                        )
                    
                    round_idx += 1
                
                if len(self.scheduler.derivative_history) > 1:
                    noise_pred_history = self.scheduler.derivative_history
                    noise_pred_good = noise_pred_history[-1]
                    noise_pred_worse = noise_pred_history[0]
                    
                    dot_product = torch.sum(noise_pred_good * noise_pred_worse, dim=list(range(1, noise_pred_good.dim())), keepdim=True)
                    norm_good = torch.sqrt(torch.sum(noise_pred_good**2, dim=list(range(1, noise_pred_good.dim())), keepdim=True))
                    norm_worse = torch.sqrt(torch.sum(noise_pred_worse**2, dim=list(range(1, noise_pred_worse.dim())), keepdim=True))
                    cos_theta = dot_product / (norm_good * norm_worse + 1e-8)
                    
                    angle_rad = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
                    sin_theta = torch.sin(angle_rad)
                    magnitude_ratio = norm_good / (norm_worse + 1e-8)
                    
                    if i >= guide_steps:
                        omega = omega_resample
                        
                    noise_pred_better = noise_pred_good + omega * sin_theta * (noise_pred_good - (magnitude_ratio*cos_theta) * noise_pred_worse)
                    
                    self.scheduler._step_index -= 1
                    if self.scheduler.lower_order_nums > 0 and self.scheduler.last_lower_order_nums < self.scheduler.config.solver_order:
                        self.scheduler.lower_order_nums -= 1

                    noise_pred_convert = self.scheduler.convert_model_output(noise_pred_better, sample=latents)
                    
                    use_corrector = (
                        self.scheduler.step_index > 0 
                        and self.scheduler.step_index - 1 not in self.scheduler.disable_corrector 
                        and self.scheduler.last_sample is not None
                    )
                    
                    self.scheduler.last_sample = latents
                    self.scheduler.model_outputs[-1] = noise_pred_convert
                    
                    latents = self.scheduler.multistep_uni_p_bh_update(
                        model_output=noise_pred_better,
                        sample=latents,
                        order=self.scheduler.this_order
                    )
                    
                    self.scheduler._step_index += 1
                    if self.scheduler.lower_order_nums >= 0 and self.scheduler.lower_order_nums < self.scheduler.config.solver_order:
                        self.scheduler.lower_order_nums += 1
                    
                    latents = latents.to(dtype=transformer_dtype)
                else:
                    latents = scheduler_output.prev_sample
                    
                self.scheduler.set_resample_mode(False)
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()
        
        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
