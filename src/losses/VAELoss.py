import torch
import torch.nn as nn
from src.losses.KLDivergenceIndGaussian import KLDivergence
from src.losses.DiceLoss import DiceLoss

class VAELoss(nn.Module):
    def __init__(self, w_KL=0.1, w_reconstruct=0.1) -> None:
        super().__init__()
        self.kl = KLDivergence()
        self.reconstruct = nn.MSELoss()
        self.dice = DiceLoss()
        
        self.w_KL = w_KL
        self.w_reconstruct = w_reconstruct
    
    def forward(self, x_reconstruct, y_reconstruct, x_true, y_true, logvar, mu):
        kl_loss = self.kl(mu, logvar)
        reconstruct_loss = self.reconstruct(x_reconstruct, x_true)
        dice_loss = self.dice(y_reconstruct, y_true)

        total_loss = self.w_KL*kl_loss + self.w_reconstruct*reconstruct_loss + dice_loss
        return total_loss, kl_loss, reconstruct_loss, dice_loss