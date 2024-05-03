import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

class DepthEstimationLosses(nn.Module):
    def __init__(self):
        super(DepthEstimationLosses, self).__init__()
        
    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)
        
    def mae_loss(self, pred, target):
        return F.l1_loss(pred, target)

    def edge_aware_smoothness_loss(self, predicted_depth, original_image):
        # Calculate gradients for the predicted depth
        depth_gradients_x = predicted_depth[:, :, :, :-1] - predicted_depth[:, :, :, 1:]
        depth_gradients_y = predicted_depth[:, :, :-1, :] - predicted_depth[:, :, 1:, :]

        # Calculate gradients for the original image
        image_gradients_x = original_image[:, :, :, :-1] - original_image[:, :, :, 1:]
        image_gradients_y = original_image[:, :, :-1, :] - original_image[:, :, 1:, :]

        # Compute the weights: exponential of the negative image gradients
        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), dim=1, keepdim=True))

        # Compute the smoothness loss
        smoothness_loss_x = torch.mean(torch.abs(depth_gradients_x) * weights_x)
        smoothness_loss_y = torch.mean(torch.abs(depth_gradients_y) * weights_y)

        # Return the sum of both directional smoothness losses
        return smoothness_loss_x + smoothness_loss_y


