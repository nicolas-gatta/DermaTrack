import torch
import torch.nn as nn

class L1_Charbonnier_loss(nn.Module):
    """
    Class implementation of the L1 Charbonnier loss, a differentiable variant of the L1 loss that is less sensitive to outliers.
    """
    
    def __init__(self, epsilon = 1e-3):
        super(L1_Charbonnier_loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt((diff ** 2) + (self.epsilon ** 2))
        return loss.mean()