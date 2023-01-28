import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore")

def dice_coef_metric(probabilities: torch.Tensor,
                    truth: torch.Tensor,
                    treshold: float = 0.5,
                    eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection) / (union + eps))
    #return np.mean(scores)
    return scores

def jaccard_coef_metric(probabilities: torch.Tensor,
            truth: torch.Tensor,
            treshold: float = 0.5,
            eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection) / (union + eps))
    #return np.mean(scores)
    return scores

class Meter:
    '''factory for storing and updating iou and dice scores.'''
    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        #self.dice_scores: list = []
        self.dice_WT: list  = []
        self.dice_TC: list = []
        self.dice_ET: list = []
        self.iou_WT: list = []
        self.iou_TC: list = []
        self.iou_ET: list = []
        #self.iou_scores: list = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        
        self.dice_WT.append(dice[0])
        self.dice_TC.append(dice[1])
        self.dice_ET.append(dice[2])

        self.iou_WT.append(iou[0])
        self.iou_TC.append(iou[1])
        self.iou_ET.append(iou[2])

        #self.dice_scores.append(dice)
        #self.iou_scores.append(iou)
    
    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice_score_WT = np.mean(self.dice_WT)
        dice_score_TC = np.mean(self.dice_TC)
        dice_score_ET = np.mean(self.dice_ET)

        iou_score_WT = np.mean(self.iou_WT)
        iou_score_TC = np.mean(self.iou_TC)
        iou_score_ET = np.mean(self.dice_ET)

        metrics = {
            "dice_WT" : dice_score_WT,
            "dice_TC" : dice_score_TC,
            "dice_ET" : dice_score_ET,
            "iou_WT"  : iou_score_WT,
            "iou_TC" : iou_score_TC,
            "iou_ET" : iou_score_ET
        }

        return metrics

class DiceLoss(nn.Module):
    """Calculate dice loss."""
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability.shape == targets.shape)
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score


class DiceLossV2(nn.Module):
    def __init__(self, sigmoid=True, mode="average_channel_batch", eps=1e-7) -> None:
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
        
        
class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return bce_loss + dice_loss
    
# helper functions for testing.  
def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Dice score for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
                
    return scores


def jaccard_coef_metric_per_classes(probabilities: np.ndarray,
            truth: np.ndarray,
            treshold: float = 0.5,
            eps: float = 1e-9,
            classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Jaccard index for data batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with jaccard scores for each class."
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert(predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores