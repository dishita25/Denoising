import torch 
from src.utils import apply_blindspot_mask, pair_downsampler
import torch.nn.functional as F

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(noisy_img, model, mask_ratio=0.6, blind_spot_weight = 1):
    # --- Blind-spot masking ---
    masked_img, mask = apply_blindspot_mask(noisy_img, mask_ratio)
    pred_masked = noisy_img - model(masked_img)   # predict clean signal

    # Supervised only on masked pixels (self-supervised target = noisy_img itself)
    loss_blindspot = F.mse_loss(pred_masked[mask.expand_as(pred_masked)],
                                noisy_img[mask.expand_as(noisy_img)])

    # --- Original consistency losses ---
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    # --- Combine ---
    loss = loss_res + loss_cons + blind_spot_weight * loss_blindspot
    return loss


def zsn2n_loss_func(noisy_stft, model):
    # Creates temporal subsamples of STFT
    noisy1, noisy2 = pair_downsampler(noisy_stft)
    
    # Residual consistency loss
    loss_res = 0.5 * (mse_loss(denoised1, noisy2) + mse_loss(denoised2, noisy1))
    
    # Denoised consistency loss  
    loss_cons = 0.5 * (mse_loss(denoised1, full_denoised1) + mse_loss(denoised2, full_denoised2))
    
    return loss_res + loss_cons