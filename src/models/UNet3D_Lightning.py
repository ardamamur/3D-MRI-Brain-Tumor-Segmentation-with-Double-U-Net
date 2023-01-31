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
    def __init__(self, hparams, model_name) -> None:
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
        if model_name == "3dunet":
            self.model = UNet3d(in_channels=hparams['modalities'],
                                n_classes=hparams['num_classes'], 
                                n_channels=hparams['start_channels'])
        else:
            self.model =  DoubleUNet3d(in_channels=hparams['modalities'],
                                        n_classes=hparams['num_classes'])

        self.bce_dice_loss = BCEDiceLoss()
        self.channel_to_class = {0: "WT", 1: "TC", 2: "ET"}

        def forward(self, x):
            pred = self.model(x)
            """
            for inference
            """
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            return pred

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.hparams['learning_rate'], 
                                        weight_decay=self.hparams['weight_decay'])

            return {"optimizer": optimizer, "lr_scheduler": PolyLR(optimizer, max_epoch=300)}

        def training_step(self, batch):
            x, y = batch
            y_hat = self.model(x)
            total_loss =  self.bce_dice_loss()
            return {"loss": total_loss.cpu()}

        def training_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> Dict:
            # assert outputs[0]["kl_loss"].requires_grad == False
            avg_total_loss = torch.stack([x['loss'] for x in outputs]).mean()
            tensorboard_logs = {"train_total_loss": avg_total_loss}
            tensorboard_logs["step"] = self.current_epoch
            self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)

        def validation_step(self, batch, batch_idx):
            x, y = batch
            pred = self.forward(x)
            
            # already averaged over batch (different methods available)
            dice_coeff = compute_dice(pred, y, ignore_empty=False).mean(0)

            # average over batch
            hausdorff = compute_hausdorff_distance(pred, y, include_background=True, percentile=95).mean(0)

            return {"dice_coeff": dice_coeff.cpu(), "hausdorff": hausdorff.cpu()}

        def validation_epoch_end(self, outputs):
            num_classes = len(self.channel_to_class)

            avg_dice_coeff = torch.stack([x['dice_coeff'] for x in outputs]).mean(0)
            avg_hausdorff_distance = torch.stack([x['hausdorff'] for x in outputs]).mean(0)

            dice = {"val_dice_coeff": {self.channel_to_class[i]: avg_dice_coeff[i] for i in range(num_classes)}}
            hausdorff = {"val_hausdorff": {self.channel_to_class[i]: avg_hausdorff_distance[i] for i in range(num_classes)}}
            avg_overall_dice = {"val_avg_overall_dice": avg_dice_coeff.mean()}
            dice.update(hausdorff)
            dice.update(avg_overall_dice)
            tensorboard_logs = dice

            tensorboard_logs["step"] = self.current_epoch

            self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)
            self.log("val_avg_overall_dice", avg_dice_coeff.mean())


        def test_step(self, batch, batch_idx):
            x, y = batch
            pred = self.forward(x)
            
            # already averaged over batch (different methods available)
            dice_coeff = compute_dice(pred, y, ignore_empty=False).mean(0)

            # average over batch
            hausdorff = compute_hausdorff_distance(pred, y, include_background=True, percentile=95).mean(0)

            return {"dice_coeff": dice_coeff, "hausdorff": hausdorff}
            
        def test_step_end(self, outputs):
            num_classes = len(self.channel_to_class)

            avg_dice_coeff = torch.stack([x['dice_coeff'] for x in outputs]).mean(0)
            avg_hausdorff_distance = torch.stack([x['hausdorff'] for x in outputs]).mean(0)

            dice = {"val_dice_coeff": {self.channel_to_class[i]: avg_dice_coeff[i] for i in range(num_classes)}}
            hausdorff = {"val_hausdorff": {self.channel_to_class[i]: avg_hausdorff_distance[i] for i in range(num_classes)}}
            avg_overall_dice = {"val_avg_overall_dice": avg_dice_coeff.mean()}
            dice.update(hausdorff)
            dice.update(avg_overall_dice)
            tensorboard_logs = dice

            tensorboard_logs["step"] = self.current_epoch

            self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)
            self.log("val_avg_overall_dice", avg_dice_coeff.mean())


