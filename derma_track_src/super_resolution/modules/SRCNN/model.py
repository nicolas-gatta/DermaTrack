import torch
import torch.nn as nn
import torch.optim as optim


class SRCNN(nn.Module):
    
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.layers = nn.Sequential(
            
            # Feature Extraction Layer
            nn.Conv2d(num_channels, 128, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            
            # Non Linear Mapping Layer 
            nn.Conv2d(128, 64, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
            
            # Reconstruction Layer
            nn.Conv2d(64, num_channels, kernel_size=5, padding=5 // 2),
        )
        

    def forward(self, x):
        return self.layers(x)