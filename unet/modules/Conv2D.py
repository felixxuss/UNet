import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel=1, bias=True,
                 up=False, down=False, bilinear=False, pooling=False):
        """
        Args:
            in_channels:        Number of input channels for this block.
            out_channels:       Number of output channels for this block.
            kernel:             Kernel size for the first convolution.
            bias:               Bias parameter for the convolution.
            up/down:            Whether the first convolution is in up- or down-sampling mode.
            bilinear/pooling:   Down-/Upsampling parameters for the first convolution.
        """
        super().__init__()
        assert not (up and down), 'up and down cannot be both True'
        assert not (kernel and (up or down)
                    ), 'Cannot use kernel with up/down sampling'
        assert kernel or up or down, 'Must use kernel or up or down sampling'

        # Down-sampling case
        if down:
            if pooling:
                self.conv = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(
                    in_channels, out_channels, 3, padding=1, bias=bias))
            else:
                self.conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        # Up-sampling case
        elif up:
            if bilinear:
                self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(
                    in_channels, out_channels,  kernel_size=3, padding=1, bias=bias))
            else:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        # Regular case
        elif kernel:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel, padding=kernel // 2, bias=bias)

    def forward(self, x):
        return self.conv(x)
