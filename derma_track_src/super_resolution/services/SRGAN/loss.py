import torch.nn as nn

from torchvision import models

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        self.feature_extractor = nn.Sequential(*list(vgg.features))
        
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
   
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.criterion(sr_features, hr_features.detach())