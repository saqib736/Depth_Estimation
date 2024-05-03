import matplotlib
import matplotlib.cm
import numpy as np
import torch

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=0.0, vmax=1.0, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

def compute_errors(pred, true):
    """
    Compute depth estimation errors between true and predicted depth maps.

    Parameters:
    - true (torch.Tensor): The ground truth depth map.
    - pred (torch.Tensor): The predicted depth map.

    Returns:
    - dict: A dictionary containing the computed error metrics.
    """
    # Ensure inputs are tensors and have the same shape
    if not (isinstance(true, torch.Tensor) and isinstance(pred, torch.Tensor)):
        raise TypeError("Inputs must be PyTorch tensors")
    if true.shape != pred.shape:
        raise ValueError("True and predicted tensors must have the same shape")
    
    errors = {'AbsRel': 0, 'SqRel': 0, 'RMSE': 0, 'RMSE_Log': 0,
                'A1': 0,  'A2': 0, 'A3': 0, 'Log10': 0}
    
    # Ensure no zero values in denominators
    pred = pred.clamp(min=1e-3)
    true = true.clamp(min=1e-3)
    
    # Calculate errors
    thresh = torch.max((true / pred), (pred / true))
    errors['A1'] = (thresh < 1.25     ).float().mean()
    errors['A2'] = (thresh < 1.25 ** 2).float().mean()
    errors['A3'] = (thresh < 1.25 ** 3).float().mean()
    
    errors['AbsRel'] = torch.mean(torch.abs(true - pred) / true)
    errors['SqRel'] = torch.mean(((true - pred) ** 2) / true)
    
    errors['RMSE'] = torch.sqrt(torch.mean((true - pred) ** 2))
    errors['RMSE_Log'] = torch.sqrt(torch.mean((torch.log(true) - torch.log(pred)) ** 2))
    
    errors['Log10'] = torch.mean(torch.abs(torch.log10(true) - torch.log10(pred)))
    
    errors['AbsRel'] = float(errors['AbsRel'].data.cpu().numpy())
    errors['SqRel'] = float(errors['SqRel'].data.cpu().numpy())
    errors['RMSE'] = float(errors['RMSE'].data.cpu().numpy())
    errors['RMSE_Log'] = float(errors['RMSE_Log'].data.cpu().numpy())
    errors['A1'] = float(errors['A1'].data.cpu().numpy())
    errors['A2'] = float(errors['A2'].data.cpu().numpy())
    errors['A3'] = float(errors['A3'].data.cpu().numpy())
    errors['Log10'] = float(errors['Log10'].data.cpu().numpy())
    
    return errors