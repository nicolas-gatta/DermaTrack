import torch.nn as nn
import torch.optim as optim


class SRCNN(nn.Module):
    
    def __init__(self, num_channels = 1):
        super(SRCNN, self).__init__()
        
        # Feature Extraction Layer
        self.conv1 = nn.Conv2d(in_channels = num_channels, out_channels = 64, kernel_size=9, padding=9 // 2)
        
        # Non Linear Mapping Layer 
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size=5, padding=5 // 2)
        
        # Reconstruction Layer
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = num_channels, kernel_size=5, padding=5 // 2)
        
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        
        # Feature Extraction Layer
        out = self.relu(self.conv1(x))
        
        # Non Linear Mapping Layer 
        out = self.relu(self.conv2(out))
        
        # Reconstruction Layer
        out = self.conv3(out)
        
        return out