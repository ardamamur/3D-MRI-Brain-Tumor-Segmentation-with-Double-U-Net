"""
author : ardamamur
date : 03.01.2023
"""
import torch
import torch.nn as nn

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features

        # Downsampling path
        self.down_convs = nn.ModuleList()
        self.down_convs.append(self.create_conv_block(self.in_channels, self.init_features))
        self.down_convs.append(self.create_conv_block(self.init_features, self.init_features * 2))
        self.down_convs.append(self.create_conv_block(self.init_features * 2, self.init_features * 4))
        self.down_convs.append(self.create_conv_block(self.init_features * 4, self.init_features * 8))

        # Upsampling path
        self.up_convs = nn.ModuleList()
        self.up_convs.append(self.create_conv_block(self.init_features * 8 + self.init_features * 4, self.init_features * 4))
        self.up_convs.append(self.create_conv_block(self.init_features * 4 + self.init_features * 2, self.init_features * 2))
        self.up_convs.append(self.create_conv_block(self.init_features * 2 + self.init_features, self.init_features))

        # Output layer
        self.out_conv = nn.Conv3d(self.init_features, self.out_channels, kernel_size=1)

    def create_conv_block(self, in_channels, out_channels):
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True), nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsampling path
        down_features = []
        for i, down_conv in enumerate(self.down_convs):
            x = down_conv(x)
            if i != len(self.down_convs) - 1:
                down_features.append(x)
                x = nn.MaxPool3d(kernel_size=2, stride=2)(x)

        # Upsampling path
        x = nn.ConvTranspose3d(self.init_features * 8, self.init_features * 4, kernel_size=2, stride=2)(x)
        x = torch.cat([x, down_features.pop()], dim=1)
        for i, up_conv in enumerate(self.up_convs):
            x = up_conv(x)
            if i != len(self.up_convs) - 1:
                x = torch.cat([x, down_features.pop()], dim=1)

        # Output layer
        x = self.out_conv(x)

        return x

class NoNewNet(nn.Module):

    def __init__(self, in_channels, out_channels, num_fmaps=32, fmap_inc_rule=lambda x: 2 ** x, depth=5):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fmaps = num_fmaps
        self.fmap_inc_rule = fmap_inc_rule
        self.depth = depth
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        self.up_interpolates = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i in range(depth):
            #down convolutions
            self.down_convs.append(nn.Sequential(
                 nn.Conv3d(in_channels if i==0 else self.num_fmaps * self.fmap_inc_rule(i), self.num_fmaps * self.fmap_inc_rule(i+1), kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(self.num_fmaps * self.fmap_inc_rule(i+1)),
                nn.ReLU(inplace=True)
            ))
            #up convolutions
            self.up_convs.append(nn.Sequential(
                nn.Conv3d(self.num_fmaps * self.fmap_inc_rule(i), self.num_fmaps * self.fmap_inc_rule(i+1), kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(self.num_fmaps * self.fmap_inc_rule(i+1)),
                nn.ReLU(inplace=True)
            ))
            self.down_pools.append(nn.MaxPool3d(kernel_size=2))
            self.up_interpolates.append(nn.Upsample(scale_factor=2, mode='nearest'))
            # merge convolutions
            self.merge_convs.append(nn.Sequential(
                nn.Conv3d(self.num_fmaps * self.fmap_inc_rule(i) * 2, self.num_fmaps * self.fmap_inc_rule(i), kernel_size=1, bias=False),
                nn.BatchNorm3d(self.num_fmaps * self.fmap_inc_rule(i)),
                nn.ReLU(inplace=True)
            ))
            # output convolutions
            if i == depth - 1:
                self.out_convs.append(nn.Conv3d(self.num_fmaps * self.fmap_inc_rule(i), out_channels, kernel_size=1))
            else:
                self.out_convs.append(nn.Sequential(
                    nn.Conv3d(self.num_fmaps * self.fmap_inc_rule(i), self.num_fmaps * self.fmap_inc_rule(i), kernel_size=1),
                    nn.BatchNorm3d(self.num_fmaps * self.fmap_inc_rule(i)),
                    nn.ReLU(inplace=True)
                ))

    def forward(self, x):
        down_features = []
        for i in range(self.depth):
            x = self.down_convs[i](x)
            down_features.append(x)
            x = self.down_pools[i](x)
        for i in range(self.depth):
            x = self.up_interpolates[i](x)
            x = torch.cat([x, down_features[self.depth - i - 1]], dim=1)
            x = self.merge_convs[i](x)
            x = self.out_convs[i](x)
        return x