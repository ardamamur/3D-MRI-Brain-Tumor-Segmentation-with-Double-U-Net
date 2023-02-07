from typing import Dict, Sequence
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.models.UNet3D_v1 import *
from src.models.double_u_net import *
from losses.BCEDiceLoss import BCEDiceLoss

from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

class UNet3D_Lightning(pl.LightningModule):
    def __init__(self, modalities=4, num_classes=3,
                 total_iterations=300, learning_rate=1e-4, weight_decay=1e-5) -> None:
        super().__init__()
        self.model_type = "double_unet"
        self.model =  DoubleUNet3d(in_channels=modalities,
                                    n_classes=num_classes)
        self.model.enc1.freeze()


        self.bce_dice_loss = BCEDiceLoss()
        self.total_iterations = total_iterations
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        out1, pred = self.model(x)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.total_iterations, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch):
        x, y = batch
        y_hat1, y_hat2 = self.model(x)
        loss1 =  self.bce_dice_loss(y_hat1, y[:,0].unsqueeze(1))
        loss2 =  self.bce_dice_loss(y_hat2, y)
        total_loss = loss1.cpu() + loss2.cpu()
        return {"loss": total_loss.cpu()}        

    def training_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> Dict:
        # assert outputs[0]["kl_loss"].requires_grad == False
        avg_total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {"train_total_loss": avg_total_loss}
        tensorboard_logs["step"] = self.current_epoch
        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)