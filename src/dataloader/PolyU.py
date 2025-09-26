import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from utils import test
# transforms
to_tensor = T.ToTensor()

def evaluate_polyu(model, dataset_path, device="cuda"):

    model.eval()

    psnrs = []
    center_crop = T.CenterCrop((256, 256))
    
    # get all mean images
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith("_mean.JPG")])

    for fname in files:
        clean_img = Image.open(os.path.join(dataset_path, fname)).convert("RGB")
        clean_img = center_crop(clean_img)
        # get corresponding real image
        noisy_fname = fname.replace("_mean.JPG", "_real.JPG")
        noisy_img = Image.open(os.path.join(dataset_path, noisy_fname)).convert("RGB")
        noisy_img = center_crop(noisy_img)

        clean_tensor = to_tensor(clean_img).unsqueeze(0).to(device)
        noisy_tensor = to_tensor(noisy_img).unsqueeze(0).to(device)

        psnr = test(model, noisy_tensor, clean_tensor)
        psnrs.append(psnr)

    avg_psnr = np.mean(psnrs)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    return avg_psnr