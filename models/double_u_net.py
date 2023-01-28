
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
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.dec1 = un.Up(self.in_channels*(512+128), 128)
        self.dec2 = un.Up(128 + self.in_channels * 64, 64)
        self.dec3 = un.Up(64 + self.in_channels * 64, 64)
        self.dec4 = un.Up(64 + self.in_channels, 32)
        self.out = un.Out(32, 1)
    
    def forward(self, x, skip_layers):
        x1 = self.dec1(x,skip_layers[3])
        x2 = self.dec2(x1,skip_layers[2])
        x3 = self.dec3(x2,skip_layers[1])
        x4 = self.dec4(x3,skip_layers[0])
        out = self.out(x4)
        return out
        
class Encoder2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = un.DoubleConv(in_channels, 16)
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
    def __init__(self,in_channels = 4, n_classes = 3) :
        super().__init__()
        self.in_channels = in_channels
        self.dec1 = Up(512+ 256+ self.in_channels*128, 256)
        self.dec2 = Up(256+ 128+ self.in_channels*64, 128)
        self.dec3 = Up(128+ 64+ self.in_channels*64, 64)
        self.dec4 = Up(80+ self.in_channels, 32)
        self.out = un.Out(32, n_classes)

    def forward(self, x, skip_layers1, skip_layers2):
        x1 = self.dec1(x,skip_layers1[3],skip_layers2[3])
        x2 = self.dec2(x1,skip_layers1[2],skip_layers2[2])
        x3 = self.dec3(x2,skip_layers1[1],skip_layers2[1])
        x4 = self.dec4(x3, skip_layers1[0],skip_layers2[0])
        out = self.out(x4)
        return out

class DoubleUNet3d(nn.Module):
    def __init__(self, in_channels = 4 , n_classes = 3):
        super().__init__()
        self.in_channels = in_channels
        self.enc1 = Encoder1()
        self.dec1 = Decoder1(in_channels)
        self.enc2 = Encoder2(in_channels)
        self.dec2 = Decoder2(in_channels)
        self.conv = nn.Conv3d(n_classes+1, n_classes, kernel_size = 1)
    def forward(self, x):
        x1_ls = []
        skip_layers1_ls = []
        for i in range(self.in_channels):
            x1, skip_layers = self.enc1(x[:,i,:].reshape(x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]))
            x1_ls.append(x1)
            skip_layers1_ls.append(skip_layers)
        x1 = torch.concat([y for y in x1_ls], dim = 1)
        skip_layers1 = []
        for i in range(4):
            concat_tensor = []
            for j in range(self.in_channels):
                concat_tensor.append(skip_layers1_ls[j][i])
            skip_layers1.append(torch.cat(concat_tensor, dim =1))
        out1 = self.dec1(x1, skip_layers1)
        x3 = torch.mul(out1, x)
        x4, skip_layers2 = self.enc2(x3)
        out2 = self.dec2(x4,skip_layers1,skip_layers2)
        out = torch.cat([out2, out1],dim=1)
        mask = self.conv(out)
        
        return out1, mask