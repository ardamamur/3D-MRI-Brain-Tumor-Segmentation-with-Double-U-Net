import torch.nn as nn
import torch.nn.functional as F
import torch

def channel_wise_dice_score(pred: torch.Tensor, truth: torch.Tensor, threshold=0.5, mode="average"):
    
    assert len(pred.shape) == 5
    
    pred[pred>=threshold] = 1

    pred = pred.flatten(-3)
    truth = truth.flatten(-3)

    dice_score = None
    if mode == "overall":
        intersection = pred*truth
        union = pred + truth
        dice_score = 2*intersection.sum(0, -1) / union.sum(0, -1)

    if mode == "average":
        intersection = pred*truth
        union = pred + truth
        dice_score = 2*intersection.sum(-1) / union.sum(-1)
        dice_score = dice_score.mean(0)
    return dice_score