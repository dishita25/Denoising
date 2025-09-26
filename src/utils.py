import torch.nn.functional as F
import torch 
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure

def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

def apply_blindspot_mask(img, mask_ratio):
    """
    Randomly masks out pixels (blind-spot).
    img: [B, C, H, W]
    mask_ratio: fraction of pixels to mask
    """
    B, C, H, W = img.shape
    mask = torch.rand(B, 1, H, W, device=img.device) < mask_ratio  # True where masked
    noisy_masked = img.clone()
    noisy_masked[mask.expand_as(noisy_masked)] = 0.0   # replace with 0 (could also use neighbor fill)
    return noisy_masked, mask

def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)

    return pred

def add_noise(x,noise_level, noise_type = 'gauss'):

    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape)
        noisy = torch.clamp(noisy,0,1)

    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x)/noise_level

    return noisy

def test(model, noisy_img, clean_img):
    from src.loss import mse 
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img),0,1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(clean_img.device)
        score = ssim(pred, clean_img)

    return PSNR, score
