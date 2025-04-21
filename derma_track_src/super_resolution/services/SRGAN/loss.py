import torch.nn as nn

from torchvision import models

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:19])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return nn.functional.mse_loss(sr_features, hr_features.detach())