import argparse
import os
import torch
import logging
import numpy as np
from PIL import Image
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils import check_min_version
from inference.marigold import DepthEstimationPipeline

# Ensure the version of diffusers is compatible
check_min_version("0.26.0.dev0")

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Depth Estimation using Stable Diffusion")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="prs-eth/marigold-v1-0",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        default="dataset_rgb/test/rgb/00009_saqib.png",
        help="Path to the input image for inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_inference",
        help="Directory to save the inference results.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoint-20000",
        help="Directory containing the checkpoint to resume from.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()

    # Load the models
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder')
    unet = UNet2DConditionModel.from_pretrained(args.checkpoint_dir, subfolder="unet", in_channels=8, sample_size=96, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    pipeline = DepthEstimationPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
    )

    pipeline = pipeline.to(accelerator.device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass

    # Run inference
    with torch.no_grad():
        input_image_pil = Image.open(args.input_image_path)

        pipe_out = pipeline(
            input_image_pil,
            denosing_steps=20,
            ensemble_size=10,
            processing_res=768,
            match_input_res=True,
            batch_size=1,
            color_map="jet",
            show_progress_bar=True,
        )

        depth_pred: np.ndarray = pipe_out.depth_np
        depth_colored: Image.Image = pipe_out.depth_colored

        # Save results
        rgb_name_base = os.path.splitext(os.path.basename(args.input_image_path))[0]
        pred_name_base = rgb_name_base + "_pred"

        npy_save_path = os.path.join(args.output_dir, f"{pred_name_base}.npy")
        if os.path.exists(npy_save_path):
            logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
        np.save(npy_save_path, depth_pred)

        colored_save_path = os.path.join(args.output_dir, f"{pred_name_base}_colored.png")
        if os.path.exists(colored_save_path):
            logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")
        depth_colored.save(colored_save_path)

    logger.info(f"Inference results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
