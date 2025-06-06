import torch
import torch.nn as nn
import numpy as np

# Residual Block
class ResidualBlock(nn.Module):
    """
    ResidualBlock implements a basic residual block as used in SRResNet architectures.
    """
    
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
# We need to use the 256 for the out channels because the PixelShuffle will decrease the channels by scale factor^2
# Since we need 64 channels as the ouput of the upsamble bloc, 64 * 4 = 256
class UpsampleBlock(nn.Module):
    """
    UpsampleBlock implements block for upsampling feature maps using a convolutional layer followed by pixel shuffling.
    """
    
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
    """
    SRResNet: Super-Resolution Residual Network for Image Super-Resolution.
    
    Args:
        channels (int): Number of feature channels in the intermediate layers. Default is 64.
        num_residual_blocks (int): Number of residual blocks in the network. Default is 16.
        up_scale (int): Upscaling factor for the output image. Must be a power of 2. Default is 4.
        
    Attributes:
        initial (nn.Sequential): Initial convolutional layer followed by PReLU activation.
        residual_blocks (nn.Sequential): Sequence of residual blocks for feature extraction.
        post_residual (nn.Sequential): Convolutional and batch normalization layers after residual blocks.
        upsampling (nn.Sequential): Sequence of upsampling blocks to increase spatial resolution.
        final (nn.Conv2d): Final convolutional layer to produce the output image.
    """
    
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