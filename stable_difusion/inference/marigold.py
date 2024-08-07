from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_util import resize_max_res, chw2hwc, colorize_depth_maps
from utils.ensemble import ensemble_depths


class DepthPipelineOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty (MAD, median absolute deviation) coming from ensembling.
    """
    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class DepthEstimationPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(self,
                 unet: UNet2DConditionModel,
                 vae: AutoencoderKL,
                 scheduler: DDIMScheduler,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(self,
                 input_image: Image,
                 denosing_steps: int = 10,
                 ensemble_size: int = 10,
                 processing_res: int = 768,
                 match_input_res: bool = True,
                 batch_size: int = 0,
                 color_map: str = "Spectral",
                 show_progress_bar: bool = True,
                 ensemble_kwargs: Dict = None,
                 ) -> DepthPipelineOutput:

        device = self.device
        input_size = input_image.size

        if not match_input_res:
            assert (
                processing_res is not None
            ), " Value Error: `resize_output_back` is only valid with "

        assert processing_res >= 0
        assert denosing_steps >= 1
        assert ensemble_size >= 1

        if processing_res > 0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )

        input_image = input_image.convert("RGB")
        image = np.array(input_image)

        rgb = np.transpose(image, (2, 0, 1))
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype).to(device)

        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)

        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = 1

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        depth_pred_ls = []

        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader

        for batch in iterable_bar:
            (batched_image,) = batch
            depth_pred_raw = self.single_infer(
                input_rgb=batched_image,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())

        depth_preds = torch.cat(depth_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()

        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)

        depth_pred = depth_pred.cpu().numpy().astype(np.float32)

        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size)
            depth_pred = np.asarray(pred_img)

        depth_pred = depth_pred.clip(0, 1)

        depth_pred_np = depth_pred.squeeze()
        depth_pred_rescaled = (depth_pred_np * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_pred_rescaled)

        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)

        return DepthPipelineOutput(
            depth_np=depth_pred,
            depth_colored=depth_image,
            uncertainty=pred_uncert,
        )

    def __encode_empty_text(self):
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(self, input_rgb: torch.Tensor,
                     num_inference_steps: int,
                     show_pbar: bool, ):

        device = input_rgb.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        rgb_latent = self.encode_RGB(input_rgb)

        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )

        if self.empty_text_embed is None:
            self.__encode_empty_text()

        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )

            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample

            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample

        torch.cuda.empty_cache()
        depth = self.decode_depth(depth_latent)
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        return depth

    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        depth_latent = depth_latent / self.depth_latent_scale_factor
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean