import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import imageio
import numpy as np
from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, Unet_dict

from dataset_prep import CustomDataset
from unet import UNet

# class NormalizeDepth(object):
#     def __init__(self, max_depth):
#         self.max_depth = max_depth

#     def __call__(self, depth):
#         # Normalize depth image
#         depth = torch.clamp(depth / self.max_depth, 0, 1)  # Normalize and clamp
#         return depth

def single_image_inference(image_pth, depth_pth, model_pth, device):
    model = U_Net().to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()  # Set the model to inference mode

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    depth_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # NormalizeDepth(5.0)  # Normalize depth values
        ])

    # Load and transform the RGB image
    img = Image.open(image_pth).convert('RGB')  
    img_transformed = transform(img).float().to(device)
    img_transformed = img_transformed.unsqueeze(0)  

    gt_depth = Image.open(depth_pth).convert('RGB')   
    gt_depth_transformed = depth_transform(gt_depth).float().to(device)  
    gt_depth_normalized = gt_depth_transformed.squeeze().cpu().detach()

    with torch.no_grad():  # No need to track gradients
        pred_depth = model(img_transformed)

    # Reformat for plotting
    img = img_transformed.squeeze().cpu().detach().permute(1, 2, 0)
    pred_depth = pred_depth.squeeze().cpu().detach()

    # output_depth = np.array(pred_depth)
    # imageio.imwrite('output/pred_combineloss.tiff', output_depth)  

    # Assuming gt_depth_normalized and pred_depth are your depth tensors
    gt_min, gt_max = gt_depth_normalized.min(), gt_depth_normalized.max()
    pred_min, pred_max = pred_depth.min(), pred_depth.max()

    # Use the broader of the two ranges for both plots to ensure consistency
    vmin = min(gt_min, pred_min)
    vmax = max(gt_max, pred_max)
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)  # Original image
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    gt_display = axes[1].imshow(gt_depth_normalized, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth Depth')
    axes[1].axis('off')

    pred_display = axes[2].imshow(pred_depth, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')

    # Instead of plt.show(), use plt.savefig()
    plt.savefig('output/newdata_newgrad2.png', bbox_inches='tight')



if __name__ == "__main__":
    SINGLE_IMG_PATH = "./monocular_dataset/meng/data_256/test/thermal/02250.png"
    MODEL_PATH = "./checkpoints/model_unet.pth"
    GT_DEPTH_PATH = "./monocular_dataset/meng/data_256/test/depth/02250.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_inference(SINGLE_IMG_PATH, GT_DEPTH_PATH, MODEL_PATH, device)