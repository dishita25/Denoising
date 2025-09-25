import torch 
import torch.nn.functional as F
import numpy as np
from src.loss import loss_func
from src.loss import mse

def train(model, optimizer, noisy_img):

  loss = loss_func(noisy_img, model)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.item()

def test(model, noisy_img, clean_img):

    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img),0,1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)

    return PSNR
