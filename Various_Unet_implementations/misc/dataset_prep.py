import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

class NormalizeDepth(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, depth):
        # Normalize depth image
        depth = F.to_tensor(depth)  # Convert to tensor (if not already done)
        depth = torch.clamp(depth / self.max_depth, 0, 1)  # Normalize and clamp
        return depth

class CustomDataset(Dataset):
    def __init__(self, root_path, test=False, max_depth=5.0):  # Use an appropriate max depth
        self.root_path = root_path

        if test:
            self.images = sorted([root_path+"/test_thermal/"+i for i in os.listdir(root_path+"/test_thermal/")])
            self.depths = sorted([root_path+"/test_depth/"+i for i in os.listdir(root_path+"/test_depth/")])
        else: 
            self.images = sorted([root_path+"/train_rgb/"+i for i in os.listdir(root_path+"/train_rgb/")])
            self.depths = sorted([root_path+"/train_depth/"+i for i in os.listdir(root_path+"/train_depth/")])

        self.rgb_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # Add normalization for RGB if necessary
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            NormalizeDepth(max_depth),  # Normalize depth values
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        depth = Image.open(self.depths[index])  # Assuming depth is already single channel

        return self.rgb_transform(img), self.depth_transform(depth)
    
    def __len__(self):
        return len(self.images)
