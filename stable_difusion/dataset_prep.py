import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F

def preprocess_depth(depth, dtype):
    depth = np.asarray(depth)
    d_min = np.percentile(depth, 2)  
    d_max = np.percentile(depth, 98)
    depth_normalized = (depth - d_min) / (d_max - d_min) * 2 - 1
    depth_clamped = np.clip(depth_normalized, -1, 1)  
    depth_tensor = torch.from_numpy(depth_clamped).to(dtype)
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

        self.images = sorted([os.path.join(root_path, "test/image", i) for i in os.listdir(os.path.join(root_path, "test/image"))]) if test \
                      else sorted([os.path.join(root_path, "train/image", i) for i in os.listdir(os.path.join(root_path, "train/image"))])
        self.depths = sorted([os.path.join(root_path, "test/depth_map", i) for i in os.listdir(os.path.join(root_path, "test/depth_map"))]) if test \
                      else sorted([os.path.join(root_path, "train/depth_map", i) for i in os.listdir(os.path.join(root_path, "train/depth_map"))])

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        depth = Image.open(self.depths[index])

        img = preprocess_image(img, self.dtype)
        depth = preprocess_depth(depth, self.dtype)

        img = F.resize(img, (432, 768))
        depth = F.resize(depth, (432, 768))

        sample = {'image': img, 'depth': depth}
        return sample

    def __len__(self):
        return len(self.images)
