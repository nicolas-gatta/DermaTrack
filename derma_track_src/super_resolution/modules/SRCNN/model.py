import torch.nn as nn
import torch.optim as optim


class SRCNN(nn.Module):
    
    def __init__(self, in_channels = 3, out_channels = 64, kernels_size: list = [9, 5, 5]):
        super(SRCNN, self).__init__()
        
        # Feature Extraction Layer (9 x 9 x 3 x 64)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernels_size[0], padding = kernels_size[0] // 2)
        
        # Non Linear Mapping Layer (5 x 5 x 64 x 32)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels // 2, kernel_size = kernels_size[1], padding = kernels_size[1] // 2)
        
        # Reconstruction Layer (5 x 5 x 32 x 3)
        self.conv3 = nn.Conv2d(in_channels = out_channels // 2, out_channels = in_channels, kernel_size = kernels_size[2], padding = kernels_size[2] // 2)
        
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        
        # Feature Extraction Layer
        out = self.relu(self.conv1(x))
        
        # Non Linear Mapping Layer 
        out = self.relu(self.conv2(out))
        
        # Reconstruction Layer
        out = self.conv3(out)
        
        return out