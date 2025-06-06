import torch
import torch.nn as nn

from torchvision import models, transforms

class VGGLoss(nn.Module):
    """
    VGGLoss computes the perceptual loss between two images using features extracted from a pre-trained VGG19 network.
    """
    
    def __init__(self):
        super(VGGLoss, self).__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        self.normalize = transforms.Normalize([ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ])
        
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:36])
        
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
            
        self.feature_extractor.eval()
        self.criterion = nn.MSELoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor):
        """
        Computes the perceptual loss between super-resolved (SR) and high-resolution (HR) images.
        
        Args:
            sr (torch.Tensor): The super-resolved image tensor.
            hr (torch.Tensor): The high-resolution ground truth image tensor.
            
        Returns:
            torch.Tensor: The computed perceptual loss between the feature representations of SR and HR images.
        """
        
        sr_features = self.feature_extractor(self.normalize(sr))
        hr_features = self.feature_extractor(self.normalize(hr))
        
        return self.criterion(sr_features, hr_features.detach())