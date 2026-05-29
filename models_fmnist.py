import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block for Fashion-MNIST.

    I use GroupNorm with affine=False instead of BatchNorm. This is useful here
    because P0 is a prior over trainable parameters. Non-affine normalization
    improves optimization without adding trainable normalization parameters.
    """
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels, affine=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels, affine=False)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.gn2(x)

        x = x + residual
        x = F.relu(x, inplace=True)
        return x


class DownsampleBlock(nn.Module):
    """
    Strided convolution block used to reduce spatial resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.gn = nn.GroupNorm(groups, out_channels, affine=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        return x
    
class SmallCNN(nn.Module):
    """
    Stronger residual CNN for Fashion-MNIST.

    I keep the class name `SmallCNN` so the rest of your script still works,
    including the `model_factory=SmallCNN` call inside the logZ estimator.
    """
    def __init__(self, width: int = 64):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width, affine=False),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(width, groups=8),
            ResidualBlock(width, groups=8),
        )

        self.stage2 = nn.Sequential(
            DownsampleBlock(width, 2 * width, groups=8),      # 28 -> 14
            ResidualBlock(2 * width, groups=8),
            ResidualBlock(2 * width, groups=8),
        )

        self.stage3 = nn.Sequential(
            DownsampleBlock(2 * width, 4 * width, groups=8),  # 14 -> 7
            ResidualBlock(4 * width, groups=8),
            ResidualBlock(4 * width, groups=8),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4 * width, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)