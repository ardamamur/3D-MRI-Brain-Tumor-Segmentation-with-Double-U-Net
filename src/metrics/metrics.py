import torch.nn as nn
import torch.nn.functional as F
import torch

def channel_wise_dice_score(pred: torch.Tensor, truth: torch.Tensor, mode="average"):
    
    assert len(pred.shape) == 5

    pred = pred.flatten(-3)
    truth = truth.flatten(-3)

    dice_score = None
    if mode == "overall":
        intersection = pred*truth
        union = pred + truth

        union_sum = union.sum(0, -1)

        union_zero = (union_sum == 0)

        dice_score = torch.zeros(union_sum.shape, device="cuda")

        dice_score[union_zero] = 1
        dice_score[~union_zero] = 2*intersection.sum(0, -1) / union_sum

    if mode == "average":
        intersection = pred*truth
        union = pred + truth

        union_sum = union.sum(-1)

        union_zero = (union_sum == 0)

        dice_score = torch.zeros(union_sum.shape, device="cuda")
        
        dice_score[union_zero] == 1
        dice_score[~union_zero] = 2*intersection.sum(-1) / union_sum
        dice_score = dice_score.mean(0)
    return dice_score