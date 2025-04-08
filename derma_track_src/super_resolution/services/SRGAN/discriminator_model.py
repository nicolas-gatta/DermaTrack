import torch.nn as nn
import torch

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(num_features = out_channels),
            nn.LeakyReLU(negative_slope = 0.2)
        )

    def forward(self, x):
        return self.block(x)


class SRGANDiscriminator(nn.Module):
    def __init__(self, crop_size = 96):
        super(SRGANDiscriminator, self).__init__()
        
        layer_channel_size = [64, 128, 128, 256, 256, 512, 512]
        
        stride_size = [1,2] * 3
        
        num_down_sampling_layer = 4
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = "same"),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

        self.blocks = nn.Sequential(
            DiscriminatorBlock(in_channels = 64, out_channels = 64, stride = 2),
            *[DiscriminatorBlock(in_channels = in_channel, out_channels = out_channel, stride = stride)
										for in_channel, out_channel, stride in zip(layer_channel_size, layer_channel_size[1:], stride_size)]
            )

        self.fc = nn.Sequential(
            nn.Linear(in_features = int(512 * (crop_size / (2 ** num_down_sampling_layer)) ** 2), out_features = 1024),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(in_features = 1024, out_features = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.initial(x)
        out = self.blocks(out)
        out = torch.flatten(out, start_dim=1)
        return self.fc(out)
