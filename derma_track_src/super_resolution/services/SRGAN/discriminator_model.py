import torch.nn as nn
import torch

class DiscriminatorBlock(nn.Module):
    """
    DiscriminatorBlock for the discriminator network in SRGAN
    """
    
    def __init__(self, in_channels, out_channels, stride):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(num_features = out_channels),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

    def forward(self, x):
        return self.block(x)


class SRGANDiscriminator(nn.Module):
    """
    SRGANDiscriminator is a PyTorch neural network module implementing the discriminator architecture for SRGAN (Super-Resolution Generative Adversarial Network).
    
    Args:
        crop_size (int, optional): The spatial size of the input image patches. Default is 96.
        
    Attributes:
        initial (nn.Sequential): Initial convolutional layer followed by LeakyReLU activation.
        blocks (nn.Sequential): Sequence of DiscriminatorBlock layers for feature extraction and downsampling.
        fc (nn.Sequential): Fully connected layers for final classification, outputting a probability via Sigmoid.
    """
    
    def __init__(self, crop_size = 96):
        super(SRGANDiscriminator, self).__init__()
        
        layer_channel_size = [64, 128, 128, 256, 256, 512, 512]
        
        stride_size = [1,2] * 3
        
        num_down_sampling_layer = len(set(layer_channel_size))
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = "same"),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )
        
        self.blocks = nn.Sequential(
            DiscriminatorBlock(in_channels = 64, out_channels = 64, stride = 2),
            *[DiscriminatorBlock(in_channels = in_channel, out_channels = out_channel, stride = stride)
										for in_channel, out_channel, stride in zip(layer_channel_size, layer_channel_size[1:], stride_size)]
            )

        # There is multiple stride of 2 which down sample by a factor of two (4 in total)
        # So using the crop size we have 96 -> 48 -> 24 -> 12 -> 6 (4 division by 2)
        # So it give us for the in features after the flatten 512 x 6 x 6 = 18432
        # Can be calculted using the crop size / (2^num down block)^2
        self.fc = nn.Sequential(
            nn.Linear(in_features = int(512 * ((crop_size / (2 ** num_down_sampling_layer)) ** 2)), out_features = 1024),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Linear(in_features = 1024, out_features = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.initial(x)
        out = self.blocks(out)
        out = torch.flatten(out, start_dim=1)
        return self.fc(out)
