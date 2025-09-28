import torch.nn.functional as F
import torch.nn as nn

class SEKG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=in_channels)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) 

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        # spatial attention
        sa_x = self.conv_sa(input_x)  
        # channel attention
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out  = sa_x + ca_x
        return out

# Adaptive Filter Generation 
class AFG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(AFG, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels*kernel_size*kernel_size, 1, 1, 0)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        x = self.sekg(input_x)
        x = self.conv(x)
        filter_x = x.reshape([b, c, self.kernel_size*self.kernel_size, h, w])
        return filter_x
    
# Dynamic convolution
class DyConv(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(DyConv, self).__init__()
        self.kernel_size = kernel_size
        self.afg = AFG(in_channels, kernel_size)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        
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

# model = network(n_chan=3, chan_embed=48, kernel_size=3)
# x = torch.randn(2, 3, 32, 32)
# out = model(x)

# print("Input shape :", x.shape)
# print("Output shape:", out.shape)

# original network
# class network(nn.Module):
#     def __init__(self, n_chan, chan_embed=48):
#         super(network, self).__init__()

#         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
#         self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)

#         self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.act(self.conv2(x))
#         x = self.conv3(x)
#         return x