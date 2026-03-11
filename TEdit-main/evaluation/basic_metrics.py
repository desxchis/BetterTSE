import torch
    
    
def MSE(pred, gt, mask=None):
    diff = pred-gt
    if mask is not None:
        diff = diff*mask
    return diff**2

def MAE(pred, gt, mask=None):
    diff = pred - gt
    if mask is not None:
        diff = diff*mask
    return torch.abs(diff)
