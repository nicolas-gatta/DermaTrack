import torch
import torch.nn as nn
import numpy as np

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels = 64, out_channels = 64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = "same", bias = False),
            nn.BatchNorm2d(num_features = out_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = "same", bias = False),
            nn.BatchNorm2d(num_features = out_channels)
        )

    def forward(self, x):
        return x + self.block(x) 

# Sub-Pixel Convolution Layers (Increase Image resolution) each block is 2x factor
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels = 64, out_channels = 256):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = "same"),
            nn.PixelShuffle(upscale_factor = 2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class SRResNet(nn.Module):
    def __init__(self, channels = 64, num_residual_blocks = 16, up_scale = 4):
        super(SRResNet, self).__init__()
        
        upscale_block = int(np.log2(up_scale))
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 9, stride = 1, padding = "same"),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )

        self.post_residual = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = "same", bias = False),
            nn.BatchNorm2d(num_features = channels)
        )

        self.upsampling = nn.Sequential(
            *[UpsampleBlock(in_channels = channels, out_channels = channels * 4) for _ in range(upscale_block)]
        )

        self.final = nn.Conv2d(in_channels = channels, out_channels = 3, kernel_size = 9, stride = 1, padding = "same")

    def forward(self, x):
        start = self.initial(x)
        out = self.residual_blocks(start)
        out = self.post_residual(out)
        out = self.upsampling(start + out)  
        out = self.final(out)
        return torch.tanh(out)