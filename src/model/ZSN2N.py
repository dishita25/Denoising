import torch.nn as nn
import torch
import torch.nn.functional as F

# Spatial + Channel Attention
class SEKG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                                 padding=1, dilation=1, groups=in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                 padding=(kernel_size - 1) // 2) 

    def forward(self, x):
        b, c, h, w = x.size()
        sa_x = self.conv_sa(x)  # spatial attention
        y = self.avg_pool(x)    # [B, C, 1, 1]
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)) \
                     .transpose(-1, -2).unsqueeze(-1)  # channel attention
        return sa_x + ca_x


# Adaptive Kernel Generator
class AFG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, kernel_size)
        # predict kernels per channel (for RGB: 3 kernels)
        self.conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        feat = self.sekg(x)
        kernels = self.conv(feat)  # [B, 3*K*K, H, W]
        kernels = kernels.view(b, 3, self.kernel_size * self.kernel_size, h, w)
        return kernels


# Apply predicted kernels dynamically
class DynamicFiltering(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, img, kernels):
        """
        img: [B, 3, H, W]
        kernels: [B, 3, K*K, H, W]
        """
        b, c, h, w = img.size()
        k = self.kernel_size

        # unfold image to patches
        patches = F.unfold(img, kernel_size=k, padding=self.pad)  # [B, 3*K*K, H*W]
        patches = patches.view(b, c, k*k, h, w)  # [B, 3, K*K, H, W]

        # weighted sum with kernels
        out = (patches * kernels).sum(2)  # [B, 3, H, W]
        return out


# Final Model
class DenoiseNet(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.feature_extractor = nn.Conv2d(3, in_channels, 3, 1, 1)  # RGB -> feature
        self.kernel_gen = AFG(in_channels, kernel_size)
        self.filter = DynamicFiltering(kernel_size)

    def forward(self, x):
        feat = self.feature_extractor(x)
        kernels = self.kernel_gen(feat)
        out = self.filter(x, kernels) 
        return out
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    model = DenoiseNet(in_channels=64, kernel_size=3)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)