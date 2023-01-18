
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore")


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

    
class Up1(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up2(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        #TODO
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class Encoder1(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        #TODO
        super().__init__()
        self.base_model = models.resnet18()
        self.base_model.load_state_dict(torch.load("../input/resnet18/resnet18.pth"))
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
    
    def forward(self, x):
        #TODO
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return layer4, [layer0 ,layer1, layer2, layer3]
    
class Decoder1(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.dec1 = Up1(16 * n_channels, 4 * n_channels)
        self.dec2 = Up1(8 * n_channels, 2 * n_channels)
        self.dec3 = Up1(4 * n_channels, n_channels)
        self.dec4 = Up1(2 * n_channels, n_channels)
    def forward(self, x, skip_layers):
        x = self.dec1(x, skip_layers[3])
        x = self.dec2(x, skip_layers[2])
        x = self.dec3(x, skip_layers[1])
        x = self.dec4(x, skip_layers[0])

        return x

class Encoder2(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.layer0 = DoubleConv(in_channels, n_channels)
        self.layer1 = Down(n_channels, 2 * n_channels)
        self.layer2 = Down(2 * n_channels, 4 * n_channels)
        self.layer3 = Down(4 * n_channels, 8 * n_channels)
        self.layer4 = Down(8 * n_channels, 8 * n_channels)
    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return layer4, [layer0, layer1, layer2, layer3]


class Decoder2(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.dec1 = Up1(16 * n_channels, 4 * n_channels)
        self.dec2 = Up1(8 * n_channels, 2 * n_channels)
        self.dec3 = Up1(4 * n_channels, n_channels)
        self.dec4 = Up1(2 * n_channels, n_channels)
    def forward(self, x, skip_layers):
        x = self.dec1(x, skip_layers[3])
        x = self.dec2(x, skip_layers[2])
        x = self.dec3(x, skip_layers[1])
        x = self.dec4(x, skip_layers[0])
        return x

class doubleUNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.encoder1 = Encoder1()
        self.decoder1 = Decoder1()
        self.encoder2 = Encoder2()
        self.decoder2 = Decoder2()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask