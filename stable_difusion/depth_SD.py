import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers.utils import ContextManagers
from accelerate import Accelerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging
import random 
import timeit
import os
from typing import Dict, Union
from diffusers.optimization import get_scheduler
import xformers
import math
from accelerate.logging import get_logger
import transformers
import datasets
from accelerate.state import AcceleratorState
import diffusers
import accelerate
import shutil

from utils.image_util import pyramid_noise_like
from PIL import Image
# from dataloader import CustomDataset
from dataset_prep import CustomDataset
from inference.marigold_pipeline import DepthEstimationPipeline
from accelerate.utils import ProjectConfiguration, set_seed

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae,text_encoder,tokenizer,unet,accelerator,scheduler,epoch,OUTPUT_DIR,BACKBONE,
                   input_image_path=None
                   ):
    
    denoise_steps = 10
    ensemble_size = 10
    processing_res = 768
    match_input_res = True
    batch_size = 1
    color_map="Spectral"
    
    logger.info("Running validation ... ")
    pipeline = DepthEstimationPipeline.from_pretrained(pretrained_model_name_or_path=BACKBONE,
                                                   vae=accelerator.unwrap_model(vae),
                                                   text_encoder=accelerator.unwrap_model(text_encoder),
                                                   tokenizer=tokenizer,
                                                   unet = accelerator.unwrap_model(unet),
                                                   scheduler = accelerator.unwrap_model(scheduler),
                                                   )

    pipeline = pipeline.to(accelerator.device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass  

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        input_image_pil = Image.open(input_image_path)

        pipe_out = pipeline(input_image_pil,
             denosing_steps=denoise_steps,
             ensemble_size= ensemble_size,
             processing_res = processing_res,
             match_input_res = match_input_res,
             batch_size = batch_size,
             color_map = color_map,
             show_progress_bar = True,
             )

        depth_pred: np.ndarray = pipe_out.depth_np
        depth_colored: Image.Image = pipe_out.depth_colored
        
        # savd as npy
        rgb_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
        pred_name_base = rgb_name_base + "_pred"
        
        npy_save_path = os.path.join(OUTPUT_DIR, f"{pred_name_base}.npy")
        np.save(npy_save_path, depth_pred)

        # Colorize
        colored_save_path = os.path.join(
            OUTPUT_DIR, f"{pred_name_base}_{epoch}_grayscale.png"
        )
        depth_colored.save(colored_save_path)
        
        del depth_colored
        del pipeline
        torch.cuda.empty_cache()

def main():
    
    RANDOM_SEED = 42
    TRAIN_BATCH_SIZE = 1
    TEST_BATCH_SIZE = 1
    LEARNING_RATE = 1e-5                        
    NUM_EPOCHS = 10
    MIXED_PRECISION = "fp16"
    GRADIENT_ACCUMULATION_STEPS = 1
    LATENT_FACTOR = 0.18215
    WARMUP_STEPS = 0
    CHECKPT_SAVE_STEP = 5000
    BACKBONE = "stabilityai/stable-diffusion-2"
    
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_WEIGHT_DECAY = 1e-2
    ADAM_EPSILON = 1e-08
    MAX_GRAD_NORM = 1.0
    
    DATA_PATH = "custom_dataset_rgb"
    OUTPUT_DIR = "sd_output"
    TEST_IMAGE = "custom_dataset_rgb/test/image/02849.png"
    TRAIN_LIST = "filenames/train_filenames_256.txt"
    TEST_LIST = "filenames/test_filenames_256.txt"
    tracker_project_name = "Depth_Estimation_SD"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logging_dir = os.path.join(OUTPUT_DIR, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=OUTPUT_DIR, logging_dir=logging_dir) 
    
    accelerator = Accelerator(
    mixed_precision=MIXED_PRECISION,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    log_with="tensorboard",
    project_config=accelerator_project_config,
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(accelerator.state, main_process_only=True)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if RANDOM_SEED is not None:
        set_seed(RANDOM_SEED)
    
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    noise_scheduler_inference = DDIMScheduler.from_pretrained(BACKBONE, subfolder="scheduler")
    noise_scheduler = DDPMScheduler(
            beta_end = 0.012,
            beta_schedule = "scaled_linear",
            beta_start = 0.00085,
            clip_sample = False,
            num_train_timesteps = 1000,
            prediction_type = "v_prediction",
            steps_offset = 1
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(BACKBONE, subfolder="tokenizer")
    logger.info("loading the noise scheduler and the tokenizer from {}".format(BACKBONE),main_process_only=True)

    vae = AutoencoderKL.from_pretrained(BACKBONE, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(BACKBONE, subfolder="text_encoder")
    # unet = UNet2DConditionModel.from_pretrained(BACKBONE, subfolder="unet")

    unet = UNet2DConditionModel.from_pretrained(BACKBONE,subfolder="unet",
                                                in_channels=8, sample_size=96,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    
    optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPSILON,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    with accelerator.main_process_first():
        train_dataset = CustomDataset(DATA_PATH, weight_dtype, test=False)
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=TRAIN_BATCH_SIZE,
                                  num_workers = 4, 
                                  shuffle=True)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
    overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps= WARMUP_STEPS * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    
    unet, optimizer, train_loader,lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader,lr_scheduler
    )
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    if overrode_max_train_steps:
        max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    if accelerator.is_main_process:
        accelerator.init_trackers(tracker_project_name)
    
    total_batch_size = TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_loader)}")
    print(f"  Num Epochs = {NUM_EPOCHS}")
    print(f"  Instantaneous batch size per device = {TRAIN_BATCH_SIZE}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    if accelerator.is_main_process:
        unet.eval()
        log_validation(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            accelerator=accelerator,
            scheduler=noise_scheduler_inference,
            epoch=0,
            OUTPUT_DIR=OUTPUT_DIR,
            BACKBONE=BACKBONE,
            input_image_path=TEST_IMAGE)
    
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            
            with accelerator.accumulate(unet):
                
                thermal_images = batch['image']
                depth_maps = batch['depth']
                
                h_img = vae.encoder(thermal_images.to(weight_dtype))
                moments_img = vae.quant_conv(h_img)
                mean_img, logvar_img = torch.chunk(moments_img, 2, dim=1)
                image_latent = mean_img * LATENT_FACTOR

                h_depth = vae.encoder(depth_maps.to(weight_dtype))
                moments_depth = vae.quant_conv(h_depth)
                mean_depth, logvar_depth = torch.chunk(moments_depth, 2, dim=1)
                depth_latent = mean_depth * LATENT_FACTOR
                
                prompt = ""
                text_inputs = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                text_input_ids = text_inputs.input_ids.to(text_encoder.device)
                empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)
                
                batch_empty_text_embed = empty_text_embed.repeat(
                    (depth_latent.shape[0], 1, 1)
                )
                
                noise = pyramid_noise_like(depth_latent)
                bsz = depth_latent.shape[0]
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=depth_latent.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(depth_latent, noise, timesteps)
                concatenated_noisy_latents = torch.cat([image_latent, noisy_latents], dim=1)
            
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(depth_latent, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                model_pred = unet(concatenated_noisy_latents, 
                                  timesteps, 
                                  encoder_hidden_states=batch_empty_text_embed).sample
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                train_loss += loss.item() / GRADIENT_ACCUMULATION_STEPS
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), MAX_GRAD_NORM)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            
            if global_step % CHECKPT_SAVE_STEP == 0:
                if accelerator.is_main_process:          
                    save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
        
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
        if accelerator.is_main_process:
                
            log_validation(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                accelerator=accelerator,
                scheduler=noise_scheduler_inference,
                epoch=epoch,
                OUTPUT_DIR=OUTPUT_DIR,
                BACKBONE=BACKBONE,
                input_image_path=TEST_IMAGE
            )
    
    accelerator.wait_for_everyone()
    accelerator.end_training()        
    
    
if __name__ == "__main__":
    main()