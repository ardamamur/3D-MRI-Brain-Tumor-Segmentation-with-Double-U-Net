import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, sigmoid=True, mode="sum", eps=1e-7) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.mode = mode
        self.eps = 1e-7

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.sigmoid(pred) if self.sigmoid else pred
        batches, channels = pred.shape[:2]

        if self.mode == "sum":
            intersect = pred * target
            union = pred**2 + target**2
            dice_coeff = 2*intersect.sum() / (union.sum()+self.eps)
        
        elif self.mode == "sum_channel_avg_batch":
            pred = pred.view(batches, -1)
            target = target.view(batches, -1)
            intersect = pred * target
            union = pred**2 + target**2
            dice_coeff = 2*intersect.sum(-1) / (union.sum(-1)+self.eps)
            dice_coeff = dice_coeff.mean()
        
        elif self.mode == "average_channel_batch":
            pred = pred.view(batches, channels, -1)
            target = target.view(batches, channels, -1)
            intersect = pred * target
            union = pred**2 + target**2
            dice_coeff = 2*intersect.sum(-1) / (union.sum(-1)+self.eps)
            dice_coeff = dice_coeff.mean()

        return 1-dice_coeff