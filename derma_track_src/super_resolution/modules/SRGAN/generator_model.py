import torch
import torch.nn as nn
import numpy as np

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x) # ElementWise Sum

# Sub-Pixel Convolution Layers (Increase Image resolution) each block is 2x factor
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class SRGAN_Generator(nn.Module):
    def __init__(self, channels = 64, num_residual_blocks = 16, up_scale = 4):
        super(SRGAN_Generator, self).__init__()
        
        upscale_block = int(np.log2(up_scale))
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(
            *([ResidualBlock(channels)] * num_residual_blocks)
        )

        self.post_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.upsampling = nn.Sequential(
            *([UpsampleBlock(channels, channels * 4)] * upscale_block)
        )

        self.final = nn.Conv2d(channels, 3, kernel_size = 9, stride=1, padding=4)

    def forward(self, x):
        start = self.initial(x)
        out = self.residual_blocks(start)
        out = self.post_residual(out)
        out = self.upsampling(start + out)  
        out = self.final(out)
        return torch.tanh(out)