import torch
import torch.nn as nn

from unet.modules.UNetBlock import UNetBlock
from unet.modules.Conv2D import Conv2d


class UNet(torch.nn.Module):
    def __init__(self, resolution, out_channels, in_channels=3, kernel_size=3, base_channels=32, channel_mult=[1, 2, 4], num_blocks=1, dropout=0.10, bilinear=False, pooling=False):
        """
        Args:
            resolution:     Image resolution at input/output.
            in_channels:    Number of color channels at input.
            out_channels:   Number of color channels at output.
            kernel_size:    Convolution kernel size.
            base_channels:  Base multiplier for the number of channels.
            channel_mult:   Per-resolution multipliers for the number of channels.
            num_blocks:     Number of residual blocks per resolution.
            dropout:        Dropout probability of intermediate activations.
            bilinear:       Whether to apply bilinear up-sampling instead of transpose convs
            pooling:        Whether to use max pooling instead of strided convs
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        # Encoder
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = resolution >> level  # really nice way of computing the resolution at each level
            if level == 0:
                cin = cout
                cout = base_channels
                self.enc[f'{res}x{res}_in_conv'] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=kernel_size)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, dropout=dropout, pooling=pooling)
            for idx in range(num_blocks):
                cin = cout
                cout = base_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, kernel=kernel_size, dropout=dropout)
        skips = [block.out_channels for name, block in self.enc.items(
        ) if f'block{num_blocks-1}' in name]  # Last UNetBlock in each level

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_bottleneck0'] = UNetBlock(
                    in_channels=cout, out_channels=cout, kernel=kernel_size, dropout=dropout)
                self.dec[f'{res}x{res}_bottleneck1'] = UNetBlock(
                    in_channels=cout, out_channels=cout, kernel=kernel_size, dropout=dropout)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, dropout=dropout, bilinear=bilinear)
            for idx in range(num_blocks):
                cin = cout if idx != 0 else cout + skips.pop()  # First UNetBlock in each level
                cout = base_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, kernel=kernel_size, dropout=dropout)
            if level == 0:
                self.dec[f'{res}x{res}_out_norm'] = nn.BatchNorm2d(
                    num_features=cout, eps=1e-6)
                self.dec[f'{res}x{res}_out_conv'] = nn.Sequential(
                    nn.ReLU(), Conv2d(in_channels=cout, out_channels=out_channels, kernel=3))

    def forward(self, x):
        # Encoder
        skips = []
        for name, block in self.enc.items():
            x = block(x)
            if f'block{self.num_blocks-1}' in name:  # Last UNetBlock in each level
                skips.append(x)
        # Decoder.
        for name, block in self.dec.items():
            if 'block0' in name:  # First UNetBlock in each level
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x
