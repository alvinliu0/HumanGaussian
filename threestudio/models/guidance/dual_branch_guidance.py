from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

try:
    from models.pipeline_rgbdepth import StableDiffusionPipeline
except:
    from .models.pipeline_rgbdepth import StableDiffusionPipeline

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

rgb_mean=0.14654
rgb_std=1.03744
whole_mean=-0.2481
whole_std=1.45647
depth_mean=0.21360
depth_std=1.20629

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

@threestudio.register("dual-branch-guidance")
class StableDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-base"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        use_anpg: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        model_key: str = "/path/to/pretrained/texture-structure_joint_model"
        vae_key: str = "stabilityai/sd-vae-ft-mse"
        lw_depth: float = 0.5
        guidance_rescale: float = 0.0
        original_size: int = 1024
        target_size: int = 1024
        grad_clip_pixel: bool = False
        grad_clip_threshold: float = 0.1

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Texture-Structure Joint Model ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        try:
            from models.unet_rgbdepth import UNet2DConditionModel
        except:
            from .models.unet_rgbdepth import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(self.cfg.model_key, subfolder="unet_ema").to(self.weights_dtype)

        vae = AutoencoderKL.from_pretrained(self.cfg.vae_key).to(self.weights_dtype)

        # Create model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            unet=unet, 
            vae=vae,
            torch_dtype=self.weights_dtype,
        ).to(torch_dtype=self.weights_dtype, torch_device=self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            # self.scheduler = DDIMScheduler.from_pretrained(
            #     self.cfg.pretrained_model_name_or_path,
            #     subfolder="scheduler",
            #     torch_dtype=self.weights_dtype,
            # )
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path, 
                subfolder="scheduler", 
                torch_dtype=self.weights_dtype,
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
            )
            self.scheduler.config.prediction_type = "v_prediction"
            self.scheduler.config['prediction_type'] = "v_prediction"
            self.scheduler.config.rescale_betas_zero_snr = True
            self.scheduler.config['rescale_betas_zero_snr'] = True
            self.scheduler.config.timestep_spacing = "trailing"
            self.scheduler.config['timestep_spacing'] = "trailing"

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        # self.controlnet = ControlNetModel.from_pretrained(
        #     self.cfg.controlnet_model, 
        #     torch_dtype=self.weights_dtype
        # ).to(self.device)

        threestudio.info(f"Loaded Texture-Structure Joint Model !")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        # control_image: Float[Tensor, "..."],
        noisy_latents_with_cond: Float[Tensor, "..."],
        noisy_latents_list: List,
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        unet_added_conditions: Dict,
    ) -> Float[Tensor, "..."]:
        input_dtype = noisy_latents_with_cond.dtype

        return self.unet(
            noisy_latents_with_cond.to(self.weights_dtype),
            noisy_latents_list,
            t.to(self.weights_dtype), 
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            added_cond_kwargs=unet_added_conditions,
        ).sample.to(input_dtype)

        # down_samples, mid_sample = self.controlnet(
        #     latents.to(self.weights_dtype),
        #     t.to(self.weights_dtype),
        #     encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        #     controlnet_cond=control_image.to(self.weights_dtype), 
        #     return_dict=False,
        # )

        # return self.unet(
        #     latents.to(self.weights_dtype),
        #     t.to(self.weights_dtype),
        #     encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        #     down_block_additional_residuals=down_samples, 
        #     mid_block_additional_residual=mid_sample,
        # ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def compute_grad_anpg(
        self,
        control_image: Float[Tensor, "B C 512 512"],
        latents: Float[Tensor, "B 4 64 64"],
        midas_depth_latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        res: Int = 512,
        # guidance_rescale: Float = 0.7,
    ):
        batch_size = elevation.shape[0]

        # if prompt_utils.use_perp_neg:
        #     add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        #     add_time_ids = torch.cat([add_time_ids] * 4 * batch_size)
        #     unet_added_conditions = {
        #         "time_ids": add_time_ids
        #     }

        #     (
        #         text_embeddings,
        #         neg_guidance_weights,
        #     ) = prompt_utils.get_text_embeddings_perp_neg(
        #         elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        #     )
        #     with torch.no_grad():
        #         noise = torch.randn_like(latents)
        #         latents_noisy = self.scheduler.add_noise(latents, noise, t)
        #         latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

        #         midas_depth_noise = torch.randn_like(midas_depth_latents)
        #         midas_depth_latents_noisy = self.scheduler.add_noise(midas_depth_latents, midas_depth_noise, t)
        #         midas_depth_latent_model_input = torch.cat([midas_depth_latents_noisy] * 4, dim=0)

        #         noise_all = torch.cat([noise, midas_depth_noise], dim=1)

        #         # control_image_input = torch.cat([control_image] * 4, dim=0)
        #         whole_latents = self.encode_images(control_image.to(self.weights_dtype))
        #         whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
        #         whole_latents_input = torch.cat([whole_latents] * 4)

        #         noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
        #         noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

        #         noise_pred = self.forward_unet(
        #             noisy_latents_with_cond,
        #             noisy_latents_list,
        #             torch.cat([t] * 4),
        #             encoder_hidden_states=text_embeddings,
        #             unet_added_conditions=unet_added_conditions,
        #         )  # (4B, 3, 64, 64)

        #     noise_pred_text = noise_pred[:batch_size]
        #     noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
        #     noise_pred_neg = noise_pred[batch_size * 2 :]

        #     e_pos = noise_pred_text - noise_pred_uncond
        #     accum_grad = 0
        #     n_negative_prompts = neg_guidance_weights.shape[-1]
        #     for i in range(n_negative_prompts):
        #         e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
        #         accum_grad += neg_guidance_weights[:, i].view(
        #             -1, 1, 1, 1
        #         ) * perpendicular_component(e_i_neg, e_pos)

        #     noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
        #         e_pos + accum_grad
        #     )
        #     if self.cfg.guidance_rescale > 0.0:
        #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text + accum_grad, guidance_rescale=self.cfg.guidance_rescale)

        # else:

        add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        add_time_ids = torch.cat([add_time_ids] * 3 * batch_size)
        unet_added_conditions = {
            "time_ids": add_time_ids
        }
        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3, dim=0)

            midas_depth_noise = torch.randn_like(midas_depth_latents)
            midas_depth_latents_noisy = self.scheduler.add_noise(midas_depth_latents, midas_depth_noise, t)
            midas_depth_latent_model_input = torch.cat([midas_depth_latents_noisy] * 3, dim=0)

            noise_all = torch.cat([noise, midas_depth_noise], dim=1)

            # control_image_input = torch.cat([control_image] * 2, dim=0)
            whole_latents = self.encode_images(control_image.to(self.weights_dtype))
            whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
            whole_latents_input = torch.cat([whole_latents] * 3)

            noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
            noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

            noise_pred = self.forward_unet(
                noisy_latents_with_cond,
                noisy_latents_list,
                torch.cat([t] * 3),
                encoder_hidden_states=text_embeddings,
                unet_added_conditions=unet_added_conditions,
            )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_neg, noise_pred_null = noise_pred.chunk(3)

            # Eq.6 in Noise-free Score Distillation, Katzir et al., arXiv preprint arXiv:2310.17590, 2023.
            delta_c = self.cfg.guidance_scale * (noise_pred_text - noise_pred_null)
            mask = (t < 200).int().view(batch_size, 1, 1, 1)
            delta_d = mask * noise_pred_null + (1 - mask) * (noise_pred_null - noise_pred_neg)
            # noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            #     noise_pred_text - noise_pred_uncond
            # )
            # if self.cfg.guidance_rescale > 0.0:
            #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.cfg.guidance_rescale)

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        # grad = w * (noise_pred - noise)
        # grad = w * (noise_pred - noise_all)
        grad = w * (delta_c + delta_d)
        if self.cfg.grad_clip_pixel:
            grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
            grad = grad_norm.clamp(max=self.cfg.grad_clip_threshold) * grad / grad_norm

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "midas_depth_latents_noisy": midas_depth_latents_noisy,
            "noise_pred": noise_pred,
            "control_image": control_image,
        }

        return grad, guidance_eval_utils

    def compute_grad_sds(
        self,
        control_image: Float[Tensor, "B C 512 512"],
        latents: Float[Tensor, "B 4 64 64"],
        midas_depth_latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        res: Int = 512,
        # guidance_rescale: Float = 0.7,
    ):
        batch_size = elevation.shape[0]

        # if prompt_utils.use_perp_neg:
        #     add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        #     add_time_ids = torch.cat([add_time_ids] * 4 * batch_size)
        #     unet_added_conditions = {
        #         "time_ids": add_time_ids
        #     }

        #     (
        #         text_embeddings,
        #         neg_guidance_weights,
        #     ) = prompt_utils.get_text_embeddings_perp_neg(
        #         elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        #     )
        #     with torch.no_grad():
        #         noise = torch.randn_like(latents)
        #         latents_noisy = self.scheduler.add_noise(latents, noise, t)
        #         latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

        #         midas_depth_noise = torch.randn_like(midas_depth_latents)
        #         midas_depth_latents_noisy = self.scheduler.add_noise(midas_depth_latents, midas_depth_noise, t)
        #         midas_depth_latent_model_input = torch.cat([midas_depth_latents_noisy] * 4, dim=0)

        #         noise_all = torch.cat([noise, midas_depth_noise], dim=1)

        #         # control_image_input = torch.cat([control_image] * 4, dim=0)
        #         whole_latents = self.encode_images(control_image.to(self.weights_dtype))
        #         whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
        #         whole_latents_input = torch.cat([whole_latents] * 4)

        #         noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
        #         noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

        #         noise_pred = self.forward_unet(
        #             noisy_latents_with_cond,
        #             noisy_latents_list,
        #             torch.cat([t] * 4),
        #             encoder_hidden_states=text_embeddings,
        #             unet_added_conditions=unet_added_conditions,
        #         )  # (4B, 3, 64, 64)

        #     noise_pred_text = noise_pred[:batch_size]
        #     noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
        #     noise_pred_neg = noise_pred[batch_size * 2 :]

        #     e_pos = noise_pred_text - noise_pred_uncond
        #     accum_grad = 0
        #     n_negative_prompts = neg_guidance_weights.shape[-1]
        #     for i in range(n_negative_prompts):
        #         e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
        #         accum_grad += neg_guidance_weights[:, i].view(
        #             -1, 1, 1, 1
        #         ) * perpendicular_component(e_i_neg, e_pos)

        #     noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
        #         e_pos + accum_grad
        #     )
        #     if self.cfg.guidance_rescale > 0.0:
        #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text + accum_grad, guidance_rescale=self.cfg.guidance_rescale)
        # else:

        add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        add_time_ids = torch.cat([add_time_ids] * 2 * batch_size)
        unet_added_conditions = {
            "time_ids": add_time_ids
        }
        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        text_embeddings = text_embeddings[:batch_size * 2]
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            midas_depth_noise = torch.randn_like(midas_depth_latents)
            midas_depth_latents_noisy = self.scheduler.add_noise(midas_depth_latents, midas_depth_noise, t)
            midas_depth_latent_model_input = torch.cat([midas_depth_latents_noisy] * 2, dim=0)

            noise_all = torch.cat([noise, midas_depth_noise], dim=1)

            # control_image_input = torch.cat([control_image] * 2, dim=0)
            whole_latents = self.encode_images(control_image.to(self.weights_dtype))
            whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
            whole_latents_input = torch.cat([whole_latents] * 2)

            noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
            noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

            noise_pred = self.forward_unet(
                noisy_latents_with_cond,
                noisy_latents_list,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                unet_added_conditions=unet_added_conditions,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        if self.cfg.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.cfg.guidance_rescale)

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        # grad = w * (noise_pred - noise)
        grad = w * (noise_pred - noise_all)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "midas_depth_latents_noisy": midas_depth_latents_noisy,
            "noise_pred": noise_pred,
            "control_image": control_image,
        }

        return grad, guidance_eval_utils

    def compute_grad_sjc(
        self,
        control_image: Float[Tensor, "B C 512 512"],
        latents: Float[Tensor, "B 4 64 64"],
        midas_depth_latents: Float[Tensor, "B 4 H W"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        res: Int = 512,
        # guidance_rescale: Float = 0.7,
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        # if prompt_utils.use_perp_neg:
        #     add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        #     add_time_ids = torch.cat([add_time_ids] * 4 * batch_size)
        #     unet_added_conditions = {
        #         "time_ids": add_time_ids
        #     }

        #     (
        #         text_embeddings,
        #         neg_guidance_weights,
        #     ) = prompt_utils.get_text_embeddings_perp_neg(
        #         elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        #     )
        #     with torch.no_grad():
        #         noise = torch.randn_like(latents)
        #         y = latents
        #         zs = y + sigma * noise
        #         scaled_zs = zs / torch.sqrt(1 + sigma**2)

        #         midas_depth_noise = torch.randn_like(midas_depth_latents)
        #         midas_depth_y = midas_depth_latents
        #         midas_depth_zs = midas_depth_y + sigma * midas_depth_noise
        #         midas_depth_scaled_zs = midas_depth_zs / torch.sqrt(1 + sigma**2)

        #         # pred noise
        #         latent_model_input = torch.cat([scaled_zs] * 4, dim=0)
        #         midas_depth_latent_model_input = torch.cat([midas_depth_scaled_zs] * 4, dim=0)

        #         # control_image_input = torch.cat([control_image] * 2, dim=0)
        #         whole_latents = self.encode_images(control_image.to(self.weights_dtype))
        #         whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
        #         whole_latents_input = torch.cat([whole_latents] * 4)

        #         noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
        #         noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

        #         noise_pred = self.forward_unet(
        #             noisy_latents_with_cond,
        #             noisy_latents_list,
        #             torch.cat([t] * 4),
        #             encoder_hidden_states=text_embeddings,
        #             unet_added_conditions=unet_added_conditions,
        #         )   # (4B, 3, 64, 64)

        #     noise_pred_text = noise_pred[:batch_size]
        #     noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
        #     noise_pred_neg = noise_pred[batch_size * 2 :]

        #     e_pos = noise_pred_text - noise_pred_uncond
        #     accum_grad = 0
        #     n_negative_prompts = neg_guidance_weights.shape[-1]
        #     for i in range(n_negative_prompts):
        #         e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
        #         accum_grad += neg_guidance_weights[:, i].view(
        #             -1, 1, 1, 1
        #         ) * perpendicular_component(e_i_neg, e_pos)

        #     noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
        #         e_pos + accum_grad
        #     )
        #     if self.cfg.guidance_rescale > 0.0:
        #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text + accum_grad, guidance_rescale=self.cfg.guidance_rescale)
        # else:

        add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        add_time_ids = torch.cat([add_time_ids] * 2 * batch_size)
        unet_added_conditions = {
            "time_ids": add_time_ids
        }
        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            y = latents

            zs = y + sigma * noise
            scaled_zs = zs / torch.sqrt(1 + sigma**2)

            midas_depth_noise = torch.randn_like(midas_depth_latents)
            midas_depth_y = midas_depth_latents
            midas_depth_zs = midas_depth_y + sigma * midas_depth_noise
            midas_depth_scaled_zs = midas_depth_zs / torch.sqrt(1 + sigma**2)

            # pred noise
            latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
            midas_depth_latent_model_input = torch.cat([midas_depth_scaled_zs] * 2, dim=0)

            # control_image_input = torch.cat([control_image] * 2, dim=0)
            whole_latents = self.encode_images(control_image.to(self.weights_dtype))
            whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
            whole_latents_input = torch.cat([whole_latents] * 2)

            noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
            noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

            noise_pred = self.forward_unet(
                noisy_latents_with_cond,
                noisy_latents_list,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                unet_added_conditions=unet_added_conditions,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        if self.cfg.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.cfg.guidance_rescale)

        Ds = torch.cat([zs, midas_depth_zs], dim=1) - torch.cat([sigma] * 2, dim=1) * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - torch.cat([y, midas_depth_y], dim=1)) / torch.cat([sigma] * 2, dim=1)
        else:
            grad = -(Ds - torch.cat([zs, midas_depth_zs], dim=1)) / torch.cat([sigma] * 2, dim=1)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "midas_depth_latents_noisy": midas_depth_scaled_zs,
            "noise_pred": noise_pred,
            "control_image": control_image,
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        control_images: Float[Tensor, "B H W C"],
        rgb: Float[Tensor, "B H W C"],
        depth: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        # guidance_rescale=0.7,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        
        control_images = control_images.permute(0, 3, 1, 2)
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        midas_depth_BCHW = depth.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        midas_depth_latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
            midas_depth_latents = F.interpolate(
                midas_depth_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            midas_depth_BCHW_512 = F.interpolate(
                midas_depth_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512.to(self.weights_dtype))
            midas_depth_latents = self.encode_images(midas_depth_BCHW_512.to(self.weights_dtype))
            midas_depth_latents = (midas_depth_latents - depth_mean) / depth_std * rgb_std + rgb_mean
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                control_images, latents, midas_depth_latents, t, prompt_utils, elevation, azimuth, camera_distances, res=rgb_BCHW.shape[-1],
            )
        elif self.cfg.use_anpg:
            grad, guidance_eval_utils = self.compute_grad_anpg(
                control_images, latents, midas_depth_latents, t, prompt_utils, elevation, azimuth, camera_distances, res=rgb_BCHW.shape[-1],
            )
        else:
            grad, guidance_eval_utils = self.compute_grad_sds(
                control_images, latents, midas_depth_latents, t, prompt_utils, elevation, azimuth, camera_distances, res=rgb_BCHW.shape[-1],
            )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        rgb_grad, midas_depth_grad = grad[:, :4], grad[:, 4:8]
        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        # target = (latents - grad).detach()
        target = (latents - rgb_grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        midas_depth_target = (midas_depth_latents - midas_depth_grad).detach()
        midas_depth_sds = self.cfg.lw_depth * F.mse_loss(midas_depth_latents, midas_depth_target, reduction='sum') / batch_size
        # print(loss_sds, midas_depth_sds)
        loss_sds += midas_depth_sds

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils, res=rgb_BCHW.shape[-1])
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        control_image,
        latents_noisy,
        midas_depth_latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
        res=1024, 
        # guidance_rescale=0.7,
    ):
        batch_size = latents_noisy.shape[0]

        # if use_perp_neg:
        #     add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        #     add_time_ids = torch.cat([add_time_ids] * 4 * batch_size)
        #     unet_added_conditions = {
        #         "time_ids": add_time_ids
        #     }
        #     # pred noise
        #     latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
        #     midas_depth_latent_model_input = torch.cat([midas_depth_latents_noisy] * 4, dim=0)

        #     # control_image_input = torch.cat([control_image] * 4, dim=0)
        #     whole_latents = self.encode_images(control_image.to(self.weights_dtype))
        #     whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
        #     whole_latents_input = torch.cat([whole_latents] * 4)

        #     noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
        #     noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

        #     noise_pred = self.forward_unet(
        #         noisy_latents_with_cond,
        #         noisy_latents_list,
        #         torch.cat([t.reshape(1)] * 4).to(self.device),
        #         encoder_hidden_states=text_embeddings,
        #         unet_added_conditions=unet_added_conditions,
        #     )  # (4B, 3, 64, 64)

        #     noise_pred_text = noise_pred[:batch_size]
        #     noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
        #     noise_pred_neg = noise_pred[batch_size * 2 :]

        #     e_pos = noise_pred_text - noise_pred_uncond
        #     accum_grad = 0
        #     n_negative_prompts = neg_guidance_weights.shape[-1]
        #     for i in range(n_negative_prompts):
        #         e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
        #         accum_grad += neg_guidance_weights[:, i].view(
        #             -1, 1, 1, 1
        #         ) * perpendicular_component(e_i_neg, e_pos)

        #     noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
        #         e_pos + accum_grad
        #     )
        #     if self.cfg.guidance_rescale > 0.0:
        #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text + accum_grad, guidance_rescale=self.cfg.guidance_rescale)
        # else:

        add_time_ids = torch.tensor([list((self.cfg.original_size, self.cfg.original_size) + (0, 0) + (self.cfg.target_size, self.cfg.target_size))]).to(self.weights_dtype).to(self.device)
        add_time_ids = torch.cat([add_time_ids] * 2 * batch_size)
        unet_added_conditions = {
            "time_ids": add_time_ids
        }
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        midas_depth_latent_model_input = torch.cat([midas_depth_latents_noisy] * 2, dim=0)

        # control_image_input = torch.cat([control_image] * 2, dim=0)
        whole_latents = self.encode_images(control_image.to(self.weights_dtype))
        whole_latents = (whole_latents - whole_mean) / whole_std * rgb_std + rgb_mean
        whole_latents_input = torch.cat([whole_latents] * 2)

        # print(latent_model_input.shape, midas_depth_latent_model_input.shape, whole_latents_input.shape)

        noisy_latents_with_cond = torch.cat([latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)
        noisy_latents_list = [torch.cat([midas_depth_latent_model_input, whole_latents_input], dim=1).to(self.weights_dtype)]

        noise_pred = self.forward_unet(
            noisy_latents_with_cond,
            noisy_latents_list,
            torch.cat([t.reshape(1)] * 2).to(self.device),
            encoder_hidden_states=text_embeddings,
            unet_added_conditions=unet_added_conditions,
        )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        if self.cfg.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.cfg.guidance_rescale)

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        midas_depth_latents_noisy,
        noise_pred,
        control_image,
        use_perp_neg=False,
        neg_guidance_weights=None,
        res=1024, 
        # guidance_rescale=0.7,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)
        midas_depth_latents_noisy_tmp = (midas_depth_latents_noisy - rgb_mean) / rgb_std * depth_std + depth_mean
        midas_depth_imgs_noisy = self.decode_latents(midas_depth_latents_noisy_tmp[:bs]).permute(0, 2, 3, 1)

        rgb_pred = noise_pred[:, :4]
        midas_depth_pred = noise_pred[:, 4:8]

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                rgb_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        # depth part
        midas_depth_latents_1step = []
        midas_depth_pred_1orig = []
        for b in range(bs):
            midas_depth_step_output = self.scheduler.step(
                midas_depth_pred[b : b + 1], t[b], midas_depth_latents_noisy[b : b + 1], eta=1
            )
            midas_depth_latents_1step.append(midas_depth_step_output["prev_sample"])
            midas_depth_pred_1orig.append(midas_depth_step_output["pred_original_sample"])
        midas_depth_latents_1step = torch.cat(midas_depth_latents_1step)
        midas_depth_pred_1orig = torch.cat(midas_depth_pred_1orig)
        midas_depth_latents_1step = (midas_depth_latents_1step - rgb_mean) / rgb_std * depth_std + depth_mean
        midas_depth_imgs_1step = self.decode_latents(midas_depth_latents_1step).permute(0, 2, 3, 1)
        midas_depth_pred_1orig = (midas_depth_pred_1orig - rgb_mean) / rgb_std * depth_std + depth_mean
        midas_depth_imgs_1orig = self.decode_latents(midas_depth_pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        midas_depth_latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            midas_depth_latents = midas_depth_latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    control_image[b : b + 1], latents, midas_depth_latents, t, text_emb, use_perp_neg, neg_guid, res=res,
                )
                rgb_pred = noise_pred[:, :4]
                midas_depth_pred = noise_pred[:, 4:8]
                # get prev latent
                latents = self.scheduler.step(rgb_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
                midas_depth_latents = self.scheduler.step(midas_depth_pred, t, midas_depth_latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)
            midas_depth_latents_final.append(midas_depth_latents)

        latents_final = torch.cat(latents_final)
        midas_depth_latents_final = torch.cat(midas_depth_latents_final)
        midas_depth_latents_final = (midas_depth_latents_final - rgb_mean) / rgb_std * depth_std + depth_mean
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)
        midas_depth_imgs_final = self.decode_latents(midas_depth_latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
            "midas_depth_imgs_noisy": midas_depth_imgs_noisy,
            "midas_depth_imgs_1step": midas_depth_imgs_1step,
            "midas_depth_imgs_1orig": midas_depth_imgs_1orig,
            "midas_depth_imgs_final": midas_depth_imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
