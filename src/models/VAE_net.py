"""
This code implements: 3D MRI brain tumor segmentation using
autoencoder regularization, A. Myronenko

Building Blocks etc. taken from (and adapted):
https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/ResNet3D_VAE.py
https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/models/unet.py

"""

import torch
import torch.nn as nn

class GreenBlock(nn.Module):
    """
    Green Blocks from the paper.
    They are always into the right direction.
    Up and Down are separate.
    """
    def __init__(self, in_channels, n_groups=8):
        super().__init__()
        self.layer = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(n_groups, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )

    
    def forward(self, x):
        # note the skip connection
        return x + self.layer(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                              stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """
    note that this block uses 1-convolutions before to reduce the channels
    """
    # use bilinear as in paper
    # but check if nearest may perform better
    def __init__(self, in_channels, out_channels, mode="nearest"):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1, 1, 1), stride=1),
            nn.Upsample(scale_factor=2, mode=mode))
    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    def __init__(self, modalities, start_channels) -> None:
        self.start_channels = start_channels
        super().__init__()
        self.l1 = \
            nn.Sequential(
                nn.Conv3d(in_channels=modalities, out_channels=start_channels,
                        kernel_size=(3, 3, 3),
                        stride=1, padding=1),
                nn.Dropout3d(0.2),
                GreenBlock(start_channels)
            )
        self.l2 = \
            nn.Sequential(
                GreenBlock(start_channels*2, start_channels*2),
                GreenBlock(start_channels*2, start_channels*2),
            )
        self.l3 = \
            nn.Sequential(                
                GreenBlock(start_channels*4, start_channels*4),
                GreenBlock(start_channels*4, start_channels*4),
            )
        self.fourblocks = \
            nn.Sequential(
                GreenBlock(start_channels*8),
                GreenBlock(start_channels*8),
                GreenBlock(start_channels*8),
                GreenBlock(start_channels*8),
            )
        self.down1 = Down(self.start_channels, self.start_channels*2)
        self.down2 = Down(self.start_channels*2, self.start_channels*4)
        self.down3 = Down(self.start_channels*4, self.start_channels*8)

    def forward(self, x):
        x1 = self.l1(x)
        x1down = self.down1(x1)

        x2 = self.l2(x1down)
        x2down = self.down2(x2)

        x3 = self.l3(x2down)
        xout = self.down3(x3)

        return xout, x1, x2, x3
    

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes=3) -> None:
        super().__init__()
        self.up1 = Up(in_channels, in_channels // 2)
        self.green1 = GreenBlock(in_channels // 2, in_channels//2)
        
        self.up2 = Up(in_channels//2, in_channels//4)
        self.green2 = GreenBlock(in_channels//4, in_channels//4)

        self.up3 = Up(in_channels//4, in_channels//8)
        self.green3 = GreenBlock(in_channels//8, in_channels//8)

        self.final_conv = nn.Conv3d(in_channels//8, num_classes, kernel_size=1)
    
    def forward(self, x, x1, x2, x3):
        x = self.up1(x) + x3
        x = self.green1(x)
        
        x = self.up2(x) + x2
        x = self.green2(x)

        x = self.up3(x) + x1
        x = self.green3(x)

        x = self.final_conv(x)

        return x



class VariationalDecoder(nn.Module):
    def __init__(self, in_channels, in_vol_dim, out_channels=4, samplingmode="nearest") -> None:
        super().__init__()
        self.in_channels = in_channels
        in_volume_after_down = (in_vol_dim[0]//16) * (in_vol_dim[1]//16) * (in_vol_dim[2]//16)
        self.param_layer = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels=16, kernel_size=3, stride=2, padding=(1,1,1)),
            nn.Flatten(1, -1),
            nn.Linear(16*in_volume_after_down, in_channels)
        )

        self.reprojection_layer = nn.Sequential(
            nn.Linear(in_channels // 2, 16*in_volume_after_down),
            nn.ReLU(),
            nn.Unflatten(1, (16, in_vol_dim[0]//16, in_vol_dim[1]//16, in_vol_dim[2]//16)),
            nn.Conv3d(16, in_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode=samplingmode),
        )

        self.up1 = Up(in_channels, in_channels // 2)
        self.green1 = GreenBlock(in_channels // 2)

        self.up2 = Up(in_channels // 2, in_channels // 4)
        self.green2 = GreenBlock(in_channels // 4)

        self.up3 = Up(in_channels // 4, in_channels//8)
        self.green3 = GreenBlock(in_channels // 8)

        self.last_conv = nn.Conv3d(in_channels // 8, 4, kernel_size=(1, 1, 1))
    
    def _reparametrize(self, mu, logvar):
        # since we only allow for positive variance, we need to exponentiate this value/vector
        # 0.5 because std = sqrt(var) for the next step
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        z_params = self.param_layer(x)
        mu = z_params[:, :self.in_channels//2]
        logvar = z_params[:, self.in_channels//2:]
        z_sampled = self._reparametrize(mu, logvar)

        x_new = self.reprojection_layer(z_sampled)
        x_new = self.up1(x_new)
        x_new = self.green1(x_new)

        x_new = self.up2(x_new)
        x_new = self.green2(x_new)

        x_new = self.up3(x_new)
        x_new = self.green3(x_new)

        x_reconstructed = self.last_conv(x_new)

        return x_reconstructed, mu, logvar