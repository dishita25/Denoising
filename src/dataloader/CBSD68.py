import os
import torchvision.transforms as T
from PIL import Image
import numpy as np
from utils import test

# transforms
to_tensor = T.ToTensor()

## change to actual cbsd68 

def evaluate_artificial(model, dataset_name, noise_level, dataset_path, device="cuda"):
    """
    Evaluate model on dataset with real and mean images.
    Args:
        model: trained denoiser model
        dataset_path: root folder (contains <dataset>_noisy_XX and original_png)
        device: "cuda" or "cpu"
    Returns:
        avg_psnr: float
    """
    model.eval()
    noise_folder = f"{dataset_name}_noisy_{noise_level}"
    noisy_path = os.path.join(dataset_path, noise_folder)
    clean_path = os.path.join(dataset_path, "original_png")

    psnrs = []
    center_crop = T.CenterCrop((256, 256))
    files = sorted(os.listdir(clean_path))

    for fname in files:
        clean_img = Image.open(os.path.join(clean_path, fname)).convert("RGB")
        clean_img = center_crop(clean_img)
        noisy_img = Image.open(os.path.join(noisy_path, fname)).convert("RGB")
        noisy_img = center_crop(noisy_img)

        clean_tensor = to_tensor(clean_img).unsqueeze(0).to(device)
        noisy_tensor = to_tensor(noisy_img).unsqueeze(0).to(device)

        psnr = test(model, noisy_tensor, clean_tensor)
        psnrs.append(psnr)

    avg_psnr = np.mean(psnrs)
    print(f"Noise Level {noise_level} → Average PSNR: {avg_psnr:.2f} dB")
    return avg_psnr