import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
from backboned_unet import Unet
from dataloader import CustomDataset

import os
import utils
import numpy as np
from PIL import Image


def main():
    
    DATAROOT = "monocular_dataset"
    FILENAME = "filenames/test_filenames_256.txt"
    
    model = Unet(backbone_name='densenet201', classes=1).cuda()
    model.load_state_dict(torch.load('./trained_models/Unet_densenet201.pth'))

    test_data = CustomDataset(DATAROOT, FILENAME)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test(test_loader, model)


def test(test_loader, model):
    model.eval()

    totalNumber = 0
    
    save_dir = "predicted_depth/Unet_densenet201"
    os.makedirs(save_dir, exist_ok=True)

    errors = []
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            image, depth = sample_batched['image'], sample_batched['depth']

            depth = depth.cuda()
            image = image.cuda()
            
            masks = depth > 0
            masked_true_depths = depth * masks

            output = model(image)
            masked_pred_depths = output * masks
            # output = F.interpolate(output, size=[depth.size(2),depth.size(3)], mode='bilinear')

            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors.append(utils.compute_errors(masked_pred_depths, masked_true_depths))
            
            for j in range(batchSize):
                save_path = os.path.join(save_dir, f"depth_map_{i*batchSize + j}.png")
                depth_map = output[j].squeeze().cpu().detach().numpy()
                depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
                depth_map_img = Image.fromarray(depth_map_normalized.astype(np.uint8))
                depth_map_img.save(save_path)

        mean_errors = {metric: np.mean([x[metric] for x in errors]) for metric in errors[0]}
        metrics_path = os.path.join(save_dir, "computed_metrics_custom.txt")
        with open(metrics_path, "w") as metrics_file:
            metrics_file.write(f"Total Number of Images Tested: {totalNumber}\n")
            for metric, value in mean_errors.items():
                metrics_file.write(f"{metric}: {value}\n")
                
        print(f'Total Number of Images Tested: {totalNumber}')
        print(mean_errors)


if __name__ == '__main__':
    main()
