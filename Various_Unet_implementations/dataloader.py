import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

class CustomDataset(Dataset):
    def __init__(self, root_path, filenames_file):
        self.root_path = root_path
        full_path = os.path.join(filenames_file)

        self.images, self.depths = [], []
        with open(full_path, 'r') as f:
            for line in f:
                thermal_path, depth_path = line.strip().split()
                self.images.append(os.path.join(root_path, thermal_path))
                self.depths.append(os.path.join(root_path, depth_path))

        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        depth = Image.open(self.depths[index])
        
        img = self.rgb_transform(img)
        depth = self.depth_transform(depth)
        
        sample = {'image': img, 'depth': depth}
        
        return sample

    def __len__(self):
        return len(self.images)
