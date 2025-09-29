import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as T

class NoisyImageDataset(Dataset):
    def __init__(self, noisy_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.transform = transform
        
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        noisy_img = Image.open(noisy_path).convert("RGB")
        
        if self.transform:
            noisy_img = self.transform(noisy_img)
        
        return noisy_img