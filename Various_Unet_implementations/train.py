import logging
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
from backboned_unet import Unet
from unet import UNet

from custom_loss import DepthEstimationLosses
from dataloader import CustomDataset
from ssim_loss import ssim
import utils

# Available Backbones
# 'resnet18'
# 'resnet34'
# 'resnet50'
# 'resnet101'
# 'resnet152'
# 'vgg16'
# 'vgg19'
# 'densenet121'
# 'densenet161'
# 'densenet169'
# 'densenet201'
# 'unet_encoder'

def main():
    # Configuration settings
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 10
    DATA_PATH = "monocular_dataset"
    FILENAMES = 'filenames/train_filenames_256.txt' 
    MODEL_SAVE_PATH = "trained_models/Unet_resnet152.pth"
    LOGGING_PATH = 'logs'

    # Initialize logging
    logging.basicConfig(filename='train_logs/Unet_resnet152.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Prepare for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CustomDataset(DATA_PATH, FILENAMES)
    total_size = len(train_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    # Data loading
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(train_dataset, [train_size, val_size], generator=generator)
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model setup
    model = Unet(backbone_name='resnet152', classes=1).to(device)
    optimizer = optim.AdamW([{'params': model.get_pretrained_parameters(), 'lr':1e-5},
                              {'params': model.get_random_initialized_parameters()}], lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    loss_functions = DepthEstimationLosses()

    # TensorBoard setup
    writer = SummaryWriter(LOGGING_PATH)

    logging.info("Training Started.")
    # Main training and validation loop
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        model.train()
        train_running_loss = 0.0
        for batch, sample_batched in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            
            optimizer.zero_grad()
            
            img = sample_batched['image'].to(device)
            depth = sample_batched['depth'].to(device)
            
            masks = depth > 0
            masked_true_depths = depth * masks
    
            pred = model(img)
            masked_pred_depths = pred * masks
            # pred = 1/preds[0]
            
            grad_loss = loss_functions.edge_aware_smoothness_loss(masked_pred_depths, masked_true_depths)
            l1_loss = loss_functions.mae_loss(masked_pred_depths, masked_true_depths)
            l_ssim = torch.clamp((1 - ssim(masked_pred_depths, masked_true_depths, val_range = 1.0)) * 0.5, 0, 1)
            loss = (1.0 * l_ssim) + (0.1 * l1_loss) + (1.0 * grad_loss)
            
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

            if batch % 10 == 0:  
                writer.add_scalar('Step/Loss_train', loss.item(), epoch * len(train_dataloader) + batch)

        scheduler.step()
        train_loss = train_running_loss / len(train_dataloader)
        writer.add_scalar('Epoch/Loss_train', train_loss, epoch)
        writer.add_scalar('Epoch/Learning_rate', scheduler.get_last_lr()[0], epoch)

        model.eval()
        val_running_loss = 0.0
        errors = []
        with torch.no_grad():
            for batch, sample_batched in enumerate(tqdm(val_dataloader, desc="Validation", leave=False)):
                img = sample_batched['image'].to(device)
                depth = sample_batched['depth'].to(device)

                masks = depth > 0
                masked_true_depths = depth * masks
        
                pred = model(img)
                masked_pred_depths = pred * masks
                
                grad_loss = loss_functions.edge_aware_smoothness_loss(masked_pred_depths, masked_true_depths)
                l1_loss = loss_functions.mae_loss(masked_pred_depths, masked_true_depths)
                l_ssim = torch.clamp((1 - ssim(masked_pred_depths, masked_true_depths, val_range = 1.0)) * 0.5, 0, 1)
                loss = (1.0 * l_ssim) + (0.1 * l1_loss) + (1.0 * grad_loss)
                
                val_running_loss += loss.item()
                batch_errors = utils.compute_errors(depth, pred)
                errors.append(batch_errors)

                if epoch % 1 == 0:  # Adjust visualization frequency as needed
                    writer.add_images('Validation/Input_Images', img, epoch)
                    writer.add_images('Validation/True_Depth', depth, epoch)
                    writer.add_images('Validation/Pred_Depth', pred, epoch)

        val_loss = val_running_loss / len(val_dataloader)
        writer.add_scalar('Epoch/Loss_validation', val_loss, epoch)
        mean_errors = {metric: np.mean([x[metric] for x in errors]) for metric in errors[0]}
        for metric, value in mean_errors.items():
            writer.add_scalar(f'Epoch/Metrics/{metric}', value, epoch)
            
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Meterics: {mean_errors}")
        logging.info(f"-"*50)
                     
    # Cleanup
    writer.close()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    torch.cuda.empty_cache()
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
