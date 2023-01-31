import torch
import numpy as np
import logging

from tqdm import tqdm
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, experiment_name: str, device, apply_sigmoid=False) -> None:
    base = "/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/"
    logging.basicConfig(filename=base+experiment_name+".log", format='%(asctime)s %(message)s',  encoding='utf-8', level=logging.INFO)
    seg_channels = ["WT", "TC", "ET"]
    dice_scores = []
    hausdorff_distances = []
    model.eval()
    with torch.no_grad():
        for train_features, train_labels in tqdm(test_loader):
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            y_hat = model(train_features)
            if apply_sigmoid:
                y_hat = (torch.sigmoid(y_hat) > 0.5).float()
            dice_scores.append(compute_dice(y_hat, train_labels, ignore_empty=False).cpu())
            hausdorff_distances.append(compute_hausdorff_distance(y_hat, train_labels, include_background=True, percentile=95).cpu())

        dice_scores = torch.cat(dice_scores).mean(0)
        hausdorff_distances = torch.cat(hausdorff_distances)
        hd = []
        for i in range(len(seg_channels)):
            hd.append(hausdorff_distances[hausdorff_distances[:, i].isfinite(), i].mean())
            
    logging.info(f"Dice_Scores:")
    for i in range(len(seg_channels)):
        logging.info(f"{seg_channels[i]}: {dice_scores[i]}")
    logging.info(f"Hausdorff:")
    for i in range(len(seg_channels)):
        logging.info(f"{seg_channels[i]}: {hd[i]}")
    
    
