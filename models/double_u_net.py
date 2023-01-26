
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore")
import resnet as rn
import UNet3D_v1 as un

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
        self.model = rn.pre_trained_resnet18()

    def forward(self, x):
        return self.model(x)
    
class Decoder1(nn.Module):
    def __init__(self):
        self.dec1 = un.Up(512+128,128)
        self.dec2 = un.Up(128 + 64,64)
        self.dec3 = un.Up(2*64,64)
        self.out = un.Out(64,1)
    
    def forward(self, x, skip_layers):
        #shape(x) = [1, 512, 8, 12, 10] || skip_layers[2]= 1, 128, 16, 24, 20
        #shape(x1) = [1, 128, 16, 24, 20] || skip_layers[1]= 1, 64, 32, 48, 40
        #shape(x2) = [1, 64, 32, 48, 40] || skip_layers[0]= 1, 64, 64, 96, 80
        #shape(x3) = [1, 64, 64, 96, 80]
        #shape(out) = [1, 1, 128, 192, 160]
        x1 = self.dec1(x,skip_layers[2])
        x2 = self.dec2(x1,skip_layers[1])
        x3 = self.dec3(x2,skip_layers[0])
        out = self.out(x3)
        return out
        
class Encoder2(nn.Module):
    def __init__(self):
        self.conv = un.DoubleConv(1, 16)
        self.enc1 = un.Down(16, 64)
        self.enc2 = un.Down(64, 128)
        self.enc3 = un.Down(128, 256)
        self.enc4 = un.Down(256, 512)
    
    def forward(self, x):
        #shape(x) = [1, 1, 128, 192, 160]
        #shape(x1) = [1, 16, 128, 192, 160]
        #shape(x2) = [1, 64, 64, 96, 80]
        #shape(x3) = [1, 128, 32, 48, 40]
        #shape(x4) = [1, 256, 16, 24, 20]
        #shape(x5) = [1, 512, 8, 12, 10]
        skip_layers = []
        x1 = self.conv(x)
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
        self.dec1 = Up(512+256+128,256)
        self.dec2 = Up(256+128+64,128)
        self.dec3 = Up(128+64+64,64)
        self.dec4 = Up(64,32)
        self.out = un.Out(32,1)

    def forward(self, x, skip_layers1, skip_layers2):
        #shape(x) = [1, 512, 8, 12, 10] || skip_layers1[2]= 1, 128, 16, 24, 20 ||skip_layers1[2]= [1, 256, 16, 24, 20]
        #shape(x1) = [1, 256, 16, 24, 20] || skip_layers[1]= 1, 64, 32, 48, 40 ||skip_layers1[1]= [1, 128, 32, 48, 40]
        #shape(x2) = [1, 128, 32, 48, 40] || skip_layers[0]= 1, 64, 64, 96, 80 ||skip_layers1[0]= [1, 64, 64, 96, 80]
        #shape(x3) = [1, 64, 64, 96, 80]
        #shape(x4) = [1, 32, 128, 192, 160]
        #shape(out) = [1, 1, 128, 192, 160]
        x1 = self.dec1(x,skip_layers1[2],skip_layers2[2])
        x2 = self.dec2(x1,skip_layers1[1],skip_layers2[1])
        x3 = self.dec3(x2,skip_layers1[0],skip_layers2[0])
        x4 = self.dec4(x3)
        out = self.out(x4)
        return out

class DoubleUNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.enc1 = Encoder1()
        self.dec1 = Decoder1()
        self.enc2 = Encoder2()
        self.dec2 = Decoder2()
        self.conv = nn.Conv3d(2, 1, kernel_size = 1)
    def forward(self, x):
        x1, skip_layers1 = self.enc1(x)
        out1 = self.dec1(x1, skip_layers1)
        x3 = torch.mul(out1, x)
        x4, skip_layers2 = self.enc2(x3)
        out2 = self.dec2(x4,skip_layers1,skip_layers2)
        out = torch.cat([out1, out2], dim=1)
        mask = self.conv(out)
        
        return mask