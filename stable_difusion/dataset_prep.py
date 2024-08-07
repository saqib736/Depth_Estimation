import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F

def preprocess_depth(depth, dtype):
    depth = np.asarray(depth)
    
    # Calculate percentiles for the current depth map
    d_min = np.percentile(depth, 2)
    d_max = np.percentile(depth, 98)
    
    # Handle the case where d_min equals d_max to avoid division by zero
    if d_max == d_min:
        depth_normalized = np.zeros_like(depth)  # Set normalized depth to zero if no variation
    else:
        # Normalize the depth values to [-1, 1]
        depth_normalized = (depth - d_min) / (d_max - d_min) * 2 - 1
    
    # Clamp the values to ensure they are within [-1, 1]
    depth_clamped = np.clip(depth_normalized, -1, 1)
    
    # Convert to tensor
    depth_tensor = torch.from_numpy(depth_clamped).to(dtype)
    
    # Add channel dimension and repeat to create a 3-channel tensor
    depth_tensor = depth_tensor.unsqueeze(0)
    depth_tensor = depth_tensor.repeat(3, 1, 1)
    
    return depth_tensor

def preprocess_image(img, dtype):
    img = img.convert("RGB")
    image_array = np.asarray(img)
    image_transposed = np.transpose(image_array, (2, 0, 1)) 
    image_normalized = image_transposed / 255.0 * 2.0 - 1.0
    image_tensor = torch.from_numpy(image_normalized).to(dtype)
    image_tensor = torch.clamp(image_tensor, min=-1, max=1)
    return image_tensor

class CustomDataset(Dataset):
    def __init__(self, root_path, dtype, test=False):
        self.root_path = root_path
        self.dtype = dtype

        self.images = sorted([os.path.join(root_path, "test/thermal", i) for i in os.listdir(os.path.join(root_path, "test/thermal"))]) if test \
                      else sorted([os.path.join(root_path, "train/thermal", i) for i in os.listdir(os.path.join(root_path, "train/thermal"))])
        self.depths = sorted([os.path.join(root_path, "test/depth", i) for i in os.listdir(os.path.join(root_path, "test/depth"))]) if test \
                      else sorted([os.path.join(root_path, "train/depth", i) for i in os.listdir(os.path.join(root_path, "train/depth"))])

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        depth = Image.open(self.depths[index])
        # depth = np.load(self.depths[index])

        img = preprocess_image(img, self.dtype)
        depth = preprocess_depth(depth, self.dtype)

        img = F.resize(img, (432, 768))
        depth = F.resize(depth, (432, 768))

        sample = {'image': img, 'depth': depth}
        return sample

    def __len__(self):
        return len(self.images)
