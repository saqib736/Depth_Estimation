import cv2
import torch
import numpy as np
from PIL import Image

from pathlib import Path
from torch.utils.data import Dataset

from dataset.utils import *


class SceneFlowDataset(Dataset):
    def __init__(
        self,
        image_list,
        root,
        crop_size=None,
        training=False,
        augmentation=False,
    ):
        super().__init__()
        with open(image_list, "rt") as fp:
            self.file_list = [Path(line.strip()) for line in fp]
        self.root = Path(root)
        self.crop_size = crop_size
        self.training = training
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Get the base and file names
        base_path = self.root / self.file_list[index].parents[1]  # Move up two levels to get to 'train' or 'test'
        file_name = self.file_list[index].name
        
        # Construct paths for left, right, disparity, and slant images
        left_path = self.root / self.file_list[index]
        right_path = base_path / "right" / file_name
        disparity_path = base_path / "disparity" / self.file_list[index].with_suffix('.tiff').name
        slant_path = base_path / "slant" / self.file_list[index].with_suffix('.npy').name

        # Load data
        data = {
            "left": np2torch(cv2.imread(str(left_path), cv2.IMREAD_COLOR), bgr=True),
            "right": np2torch(cv2.imread(str(right_path), cv2.IMREAD_COLOR), bgr=True),
            "disp": np2torch(np.array(Image.open(str(disparity_path)))),  # -1 means load as is (including alpha channel if present)
            "dxy": np2torch(np.load(str(slant_path)), t=False),
        }
        # Add cropping, padding, and augmentation as necessary
        if self.crop_size is not None:
            data = crop_and_pad(data, self.crop_size, self.training)
        if self.training and self.augmentation:
            data = augmentation(data, self.training)
        return data

if __name__ == "__main__":
    import torchvision
    from colormap import apply_colormap, dxy_colormap

    dataset = SceneFlowDataset(
        "lists/train_new.txt",
        "stereodata",
        training=True,
    )
    
    print(dataset[0])
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for ids, data in enumerate(loader):
        disp = data["disp"]
        disp = torch.clip(disp / 192 * 255, 0, 255).long()
        disp = apply_colormap(disp)

        dxy = data["dxy"]
        dxy = dxy_colormap(dxy)
        output = torch.cat((data["left"], data["right"], disp, dxy), dim=0)
        torchvision.utils.save_image(output, "{:06d}.png".format(ids), nrow=1)
