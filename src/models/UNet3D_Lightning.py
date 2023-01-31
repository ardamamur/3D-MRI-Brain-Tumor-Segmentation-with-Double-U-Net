from typing import Dict, Sequence
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.models.UNet3D_v1 import *
from src.losses.UNet3D_Loss import BCEDiceLoss
from solver import PolyLR

from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

class UNet3D_Lightning(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        """
        hparams = {
            "volume_shape" : 160,192,128
            "modalities" : 4
            "start_channels" : 16
            "num_classes" : 3"
            "learning_rate" : 1e-4
            "weight_decay" : 1e-5
        }
        """
        # volume_shape, modalities=4, start_channels=16, num_classes=3
        # start_channels : init_channels = 16
        self.hparams = hparams
        self.model = UNet3d(in_channels=hparams['modalities'],
                            n_classes=hparams['num_classes'], 
                            n_channels=hparams['start_channels'])
        self.bce_dice_loss = BCEDiceLoss()

        def forward(self, x):
            pred = self.model(x)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            return pred

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(),
                                        lr=self.hparams['learning_rate'], 
                                        weight_decay=self.hparams['weight_decay'])

            return {"optimizer": optimizer, "lr_scheduler": PolyLR(optimizer, max_epoch=300)}

        def training_step(self, batch):
            x, y = batch
            y_hat = self.model(x)
            bce_dice_loss =  self.bce_dice_loss()
            



