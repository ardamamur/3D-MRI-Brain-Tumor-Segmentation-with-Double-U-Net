import torch
import torch.nn as nn

class KLDivergence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, mu, logvar):
        # let's take the sum here instead of mean
        # depending on our batch size - which may be one - it won't differ
        # also experiment with different weights
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())