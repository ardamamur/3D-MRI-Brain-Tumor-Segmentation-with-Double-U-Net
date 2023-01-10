"""
author : ardamamur
date : 03.01.2023
"""
import torch
import torch.nn as nn

"""
This implementation consists of a series of DoubleConv blocks, Down blocks, and Up blocks. The DoubleConv block consists
of two 3D convolutional layers followed by batch normalization and ReLU activation. The Down block applies max pooling 
to reduce the spatial dimensions of the input and then applies a DoubleConv block. The Up block applies upsampling to 
the input and then concatenates it with another input before applying a DoubleConv block. The final layer is a 
1x1 convolutional layer followed by a sigmoid activation to produce a segmentation map
"""

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_z = x2.size()[4] - x1.size()[4]
        x1 = torch.nn.functional.pad(x1, (diff_z // 2, diff_z - diff_z // 2,
                                          diff_y // 2, diff_y - diff_y // 2,
                                          diff_x // 2, diff_x - diff_x // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out = nn.Sequential(
            nn.Conv3d(64, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


# model = UNet3D(in_channels=1, out_channels=1)