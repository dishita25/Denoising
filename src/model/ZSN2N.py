import torch.nn.functional as F
import torch.nn as nn

class SEKG(nn.Module):
    def __init__(self, in_channels, kernel_size=3):  # Remove num_kernels parameter
        super().__init__()
        
        # Global average pooling to get image-level features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Generate kernels for all input channels
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels * kernel_size * kernel_size, 1)
        )
        
        self.kernel_size = kernel_size
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Get global image features
        global_feat = self.global_pool(x)  # [b, c, 1, 1]
        
        # Generate kernels for all channels
        kernels = self.kernel_gen(global_feat)  # [b, c*k*k, 1, 1]
        kernels = kernels.view(b, c, self.kernel_size, self.kernel_size)  # [b, c, k, k]
        
        # Normalize kernel so it behaves like a blur kernel
        kernels = kernels / (kernels.abs().sum(dim=(-2, -1), keepdim=True) + 1e-8)
        
        return kernels


# Adaptive Filter Generation 
class AFG(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(AFG, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, kernel_size)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        kernels = self.sekg(input_x)  # [b, c, k, k] - one kernel per image
        # Expand to match spatial dimensions for the dynamic conv operation
        kernels = kernels.unsqueeze(-1).unsqueeze(-1)  # [b, c, k, k, 1, 1]
        kernels = kernels.expand(-1, -1, -1, -1, h, w)  # [b, c, k, k, h, w]
        kernels = kernels.contiguous().view(b, c, self.kernel_size*self.kernel_size, h, w)
        return kernels

# Dynamic convolution
class DyConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DyConv, self).__init__()
        self.kernel_size = kernel_size
        self.afg = AFG(in_channels, kernel_size)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=1, stride=1)
        
    def forward(self, input_x):
        b, c, h, w = input_x.size()
        filter_x = self.afg(input_x)                               
        unfold_x = self.unfold(input_x).reshape(b, c, -1, h, w)    
        out = (unfold_x * filter_x).sum(2)                         
        return out
    
class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48, kernel_size=3):
        super(network, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = DyConv(in_channels=n_chan, kernel_size=kernel_size)      
        self.proj1 = nn.Conv2d(n_chan, chan_embed, 1) 
        self.conv2 = DyConv(in_channels=chan_embed, kernel_size=kernel_size)

        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)  

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.proj1(x)    
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
