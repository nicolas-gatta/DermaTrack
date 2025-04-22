import torch.nn as nn

from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        self.normalize = transforms.Normalize([ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ])
        
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:36])
        
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
            
        self.feature_extractor.eval()
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(self.normalize(sr))
        hr_features = self.feature_extractor(self.normalize(hr))
        return self.criterion(sr_features, hr_features.detach())