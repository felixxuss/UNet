import torch.nn as nn
from torch.nn.functional import relu

from unet.modules.Conv2D import Conv2d


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=0, up=False, down=False,
                 dropout=0, eps=1e-5, bilinear=False, pooling=False):
        """
        Args:
            in_channels:        Number of input channels for this block.
            out_channels:       Number of output channels for this block.
            kernel:             Kernel size for the first convolution.
            up/down:            Whether the first convolution is in up- or down-sampling mode.
            dropout:            Dropout probability for dropout before the second conv.
            eps:                Epsilon parameter for BatchNorm2d.
            bilinear/pooling:   Down-/up-sampling parameters for the first convolution.
        """
        super().__init__()
        self.out_channels = out_channels
        self.dropout = dropout

        self.norm0 = nn.BatchNorm2d(in_channels, eps)
        self.conv0 = Conv2d(in_channels, out_channels, kernel,
                            up=up, down=down, bilinear=bilinear, pooling=pooling)
        self.norm1 = nn.BatchNorm2d(out_channels, eps)
        self.conv1 = Conv2d(out_channels, out_channels, 3)
        self.res = not (out_channels != in_channels or up or down)

    def forward(self, x):
        orig = x
        x = self.conv0(relu(self.norm0(x)))  # in-conv
        x = relu(self.norm1(x))
        x = self.conv1(nn.functional.dropout(
            x, p=self.dropout, training=self.training))  # optional dropout
        if self.res:
            x = x + orig  # optional residual connection
        return x
