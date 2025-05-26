import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Interpolate(nn.Module):
    def __init__(self, scale_factor = 2, mode = "nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor = self.scale_factor, mode=self.mode)

# Dense Block 
class ResidualDenseBlock(nn.Module):
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
    def __init__(self, in_channels = 64, out_channels = 64):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            Interpolate(scale_factor = 2, mode = "nearest"),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

    def forward(self, x):
        return self.block(x)

class RRDBNet(nn.Module):
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
        
        