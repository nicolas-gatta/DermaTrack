import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SRGAN_Discriminator(nn.Module):
    def __init__(self):
        super(SRGAN_Discriminator, self).__init__()
        
        layer_channel_size = [64, 128, 128, 256, 256, 512, 512]
        
        stride_size = [1,2] * 3
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.blocks = nn.Sequential(
            DiscriminatorBlock(64, 64, 2),
            *[DiscriminatorBlock(in_channel, out_channel, stride)
										for in_channel, out_channel, stride in zip(layer_channel_size, layer_channel_size[1:], stride_size)]
            )

        self.fc = nn.Sequential(
            nn.Linear(512 * (4 * 4), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)  # Outputs logits
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.initial(x)
        out = self.blocks(out)
        out = self.fc(out)
        return self.sigmoid(out)
