import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Interpolate(nn.Module):
    """
    A custom PyTorch module for upsampling input tensors using interpolation.
    Args:
        scale_factor (int or float, optional): Multiplier for spatial size. Default is 2.
        mode (str, optional): Algorithm used for upsampling. Default is "bicubic".
            
    Forward Args:
        x (torch.Tensor): Input tensor to be upsampled.
        
    Returns:
        torch.Tensor: Upsampled tensor.
    """
    
    def __init__(self, scale_factor = 2, mode = "bicubic"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor = self.scale_factor, mode=self.mode)

# Dense Block 
class ResidualDenseBlock(nn.Module):
    """
    ResidualDenseBlock implements a residual dense block as used in super-resolution networks like ESRGAN.
    """
    
    def __init__(self, num_features = 64, growth_channels = 32, bias = True):
        super(ResidualDenseBlock, self).__init__()
        
        self.dense1 = nn.Sequential(
            nn.Conv2d(in_channels = num_features, out_channels = growth_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )
        
        self.dense2 = nn.Sequential(
            nn.Conv2d(in_channels = num_features + growth_channels, out_channels = growth_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )
                
        self.dense3 = nn.Sequential(
            nn.Conv2d(in_channels = num_features + (2 * growth_channels), out_channels = growth_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace = True)
        )
                        
        self.dense4 = nn.Sequential(
            nn.Conv2d(in_channels = num_features + (3 * growth_channels), out_channels = growth_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )
        
        self.conv1 = nn.Conv2d(in_channels = num_features + (4 * growth_channels), out_channels = num_features, kernel_size = 3, stride = 1, padding = 1, bias=bias)

    def forward(self, x):
        d1 = self.dense1(x)
        d2 = self.dense2(torch.cat(tensors = [x, d1], dim = 1))
        d3 = self.dense3(torch.cat(tensors = [x, d1, d2], dim = 1))
        d4 = self.dense4(torch.cat(tensors = [x, d1, d2, d3], dim = 1))
        final = self.conv1(torch.cat(tensors = [x, d1, d2, d3, d4], dim = 1))
        return (final * 0.2) + x
    
class ResidualInResidualDenseBlock(nn.Module):
    """
    A Residual-in-Residual Dense Block (RRDB) as used in ESRGAN and similar super-resolution networks.
    """
    
    def __init__(self, num_features = 64, growth_channels = 32, bias = True):
        super(ResidualInResidualDenseBlock, self).__init__()
        
        self.RRDB1 = ResidualDenseBlock(num_features = num_features, growth_channels = growth_channels, bias = bias)
        self.RRDB2 = ResidualDenseBlock(num_features = num_features, growth_channels = growth_channels, bias = bias)
        self.RRDB3 = ResidualDenseBlock(num_features = num_features, growth_channels = growth_channels, bias = bias)
    
    def forward(self, x):
        out = self.RRDB1(x)
        out = self.RRDB2(out)
        out = self.RRDB3(out)
        return (out * 0.2) + x
        
# Sub-Pixel Convolution Layers (Increase Image resolution) each block is 2x factor
class UpsampleBlock(nn.Module):
    """
    UpsampleBlock that performs upsampling.
    """
    
    def __init__(self, in_channels = 64, out_channels = 64):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            Interpolate(scale_factor = 2, mode = "bicubic"),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

    def forward(self, x):
        return self.block(x)

class RRDBNet(nn.Module):
    """
    RRDBNet: Residual-in-Residual Dense Block Network for Image Super-Resolution.
    
    Args:
        in_channels (int): Number of input image channels. Default is 3 (RGB).
        out_channels (int): Number of output image channels. Default is 3 (RGB).
        num_features (int): Number of feature maps in the intermediate layers. Default is 64.
        growth_channels (int): Number of growth channels in dense blocks. Default is 32.
        bias (bool): Whether to use bias in convolutional layers. Default is True.
        num_blocks (int): Number of RRDB blocks. Default is 23.
        up_scale (int): Upscaling factor (should be a power of 2, e.g., 2, 4, 8). Default is 4.
        
    Attributes:
        initial (nn.Conv2d): Initial convolutional layer.
        RRDB_layer (nn.Sequential): Sequence of RRDB blocks.
        post_RRDB (nn.Conv2d): Convolutional layer after RRDB blocks.
        upsampling (nn.Sequential): Sequence of upsampling blocks.
        conv (nn.Conv2d): Convolutional layer after upsampling.
        conv_final (nn.Conv2d): Final convolutional layer to produce output image.

    Returns:
        torch.Tensor: Super-resolved image tensor of shape (N, out_channels, H * up_scale, W * up_scale).
    """
    
    def __init__(self, in_channels = 3, out_channels = 3, num_features = 64, growth_channels = 32, bias = True, num_blocks = 23, up_scale = 4):
        super(RRDBNet, self).__init__()
        
        upscale_block = int(np.log2(up_scale))
        
        self.initial = nn.Conv2d(in_channels = in_channels, out_channels = num_features, kernel_size = 3, stride = 1, padding = 1, bias = True)
        
        self.RRDB_layer = nn.Sequential(*[ResidualInResidualDenseBlock(num_features = num_features, growth_channels = growth_channels, bias = bias) for _ in range(num_blocks)])
        
        self.post_RRDB = nn.Conv2d(in_channels = num_features, out_channels = num_features, kernel_size = 3, stride = 1, padding = 1, bias = True)
        
        self.upsampling = nn.Sequential(*[UpsampleBlock(in_channels = num_features, out_channels = num_features) for _ in range(upscale_block)])
        
        self.conv = nn.Conv2d(in_channels = num_features, out_channels = num_features, kernel_size = 3, stride = 1, padding = 1, bias = True)
        
        self.conv_final = nn.Conv2d(in_channels = num_features, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True)

    def forward(self, x):
        initial = self.initial(x)
        out = self.RRDB_layer(initial)
        out = self.post_RRDB(out)
        out = self.upsampling(initial + out)
        out = self.conv(out)
        return self.conv_final(out)
        
        