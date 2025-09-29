import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

# Kernel Generator (one kernel per image)
class SEKG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, kernel_size * kernel_size, 1)  # output = 1 kernel per image
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        b, c, h, w = x.size()
        feat = self.global_pool(x)                       # [b, c, 1, 1]
        kernel = self.kernel_gen(feat).view(b, 1, self.kernel_size, self.kernel_size)  # [b,1,k,k]
        # normalize kernel so it behaves like a blur kernel
        kernel = kernel / (kernel.sum(dim=(2, 3), keepdim=True) + 1e-6)
        return kernel  # [b,1,k,k]


class network(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.sekg = SEKG(in_channels, kernel_size)

    def forward(self, x):
        b, c, h, w = x.size()
        kernels = self.sekg(x) 

        outputs = []
        for i in range(b):
            k = kernels[i]  # [1,k,k]
            k = k.expand(c, 1, self.kernel_size, self.kernel_size)  # one kernel per channel
            out = F.conv2d(x[i].unsqueeze(0), k, padding=self.padding, groups=c)
            outputs.append(out)

        return torch.cat(outputs, dim=0)  # [b,c,h,w]


# ------------------- TEST -------------------
if __name__ == "__main__":
    model = network(in_channels=3, kernel_size=3, padding=1)
    img = torch.randn(2, 3, 64, 64)  # batch=2, RGB 64x64
    out = model(img)
    print("Input:", img.shape)
    print("Output:", out.shape)
