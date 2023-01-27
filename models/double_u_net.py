
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore")
from models import resnet as rn
from models import UNet3D_v1 as un

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            
        self.conv = un.DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None, x3 = None):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        if x2 == None:
            pass
        elif x3 == None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)


class Encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = rn.pre_trained_resnet18()

    def forward(self, x):
        return self.model(x)
    
class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = un.Up(512+128,128)
        self.dec2 = un.Up(128 + 64,64)
        self.dec3 = un.Up(2*64,64)
        self.dec4 = un.Up(65,32)
        self.out = un.Out(32,1)
    
    def forward(self, x, skip_layers):
        x1 = self.dec1(x,skip_layers[3])
        x2 = self.dec2(x1,skip_layers[2])
        x3 = self.dec3(x2,skip_layers[1])
        x4 = self.dec4(x3,skip_layers[0])
        out = self.out(x4)
        return out
        
class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = un.DoubleConv(1, 16)
        self.enc1 = un.Down(16, 64)
        self.enc2 = un.Down(64, 128)
        self.enc3 = un.Down(128, 256)
        self.enc4 = un.Down(256, 512)
    
    def forward(self, x):
        skip_layers = []
        x1 = self.conv(x)
        skip_layers.append(x1)
        x2 = self.enc1(x1)
        skip_layers.append(x2)
        x3 = self.enc2(x2)
        skip_layers.append(x3)
        x4 = self.enc3(x3)
        skip_layers.append(x4)
        x5 = self.enc4(x4)

        return x5, skip_layers


class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = Up(512+256+128,256)
        self.dec2 = Up(256+128+64,128)
        self.dec3 = Up(128+64+64,64)
        self.dec4 = Up(81,32)
        self.out = un.Out(32,16)

    def forward(self, x, skip_layers1, skip_layers2):
        x1 = self.dec1(x,skip_layers1[3],skip_layers2[3])
        x2 = self.dec2(x1,skip_layers1[2],skip_layers2[2])
        x3 = self.dec3(x2,skip_layers1[1],skip_layers2[1])
        x4 = self.dec4(x3, skip_layers1[0],skip_layers2[0])
        out = self.out(x4)
        return out

class DoubleUNet3d(nn.Module):
    def __init__(self, in_channels = 1 , n_classes = 4):
        super().__init__()
        self.enc1 = Encoder1()
        self.dec1 = Decoder1()
        self.enc2 = Encoder2()
        self.dec2 = Decoder2()
        self.conv = nn.Conv3d(16, n_classes, kernel_size = 1)
    def forward(self, x):
        x1, skip_layers1 = self.enc1(x)
        out1 = self.dec1(x1, skip_layers1)
        x3 = torch.mul(out1, x)
        x4, skip_layers2 = self.enc2(x3)
        out2 = self.dec2(x4,skip_layers1,skip_layers2)
        out = torch.mul(out2, out1)
        mask = self.conv(out)
        
        return out1, mask