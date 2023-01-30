from typing import Dict, Sequence
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.models.VAE_net import Decoder, Encoder, VariationalDecoder
from src.losses.VAELoss import VAELoss
from src.losses.DiceLoss import DiceLoss
from solver import PolyLR

from metrics.metrics import channel_wise_dice_score
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

from torchmetrics import Dice#
from monai.metrics.meandice import compute_dice

class VAELightning(pl.LightningModule):
    def __init__(self, volume_shape, modalities=4, start_channels=32, num_classes=3) -> None:
        super().__init__()
        self.encoder = Encoder(modalities=modalities, start_channels=start_channels)
        self.decoder = Decoder(in_channels=32*8, num_classes=num_classes)
        self.vae_decoder = VariationalDecoder(32*8, volume_shape, num_classes)
        
        self.vae_loss = VAELoss()
        self.dice_loss = DiceLoss()

        self.channel_to_class = {0: "WT", 1: "TC", 2: "ET"}

        # self.dice_metric = Dice(zero_division=1, num_classes=3, threshold=0.5, average="samples", mdmc_average="global")

    def forward(self, x):
        enc, x1, x2, x3 = self.encoder(x)
        pred = self.decoder(enc, x1, x2, x3)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": PolyLR(optimizer, max_epoch=300)}

    def training_step(self, batch):
        x, y = batch
        latent, x1, x2, x3 = self.encoder(x)
        y_hat = self.decoder(latent, x1, x2, x3)
        x_hat, mu, logvar = self.vae_decoder(latent)
        total_loss, kl_loss, reconstruct_loss, dice_loss = self.vae_loss(x_hat,
                                                                         y_hat,
                                                                         x,
                                                                         y,
                                                                         mu,
                                                                         logvar)
        return {"loss": total_loss.cpu(), "kl_loss": kl_loss.cpu(), "reconstruct_loss": reconstruct_loss.cpu(),
                "dice_loss": dice_loss.cpu()}

    def training_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> Dict:
        # assert outputs[0]["kl_loss"].requires_grad == False
        avg_kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        avg_total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_reconstruction_loss = torch.stack([x['reconstruct_loss'] for x in outputs]).mean()
        avg_dice_loss = torch.stack([x['dice_loss'] for x in outputs]).mean()
        tensorboard_logs = {"train_kl_loss": avg_kl_loss, "train_total_loss": avg_total_loss,
                            "train_reconstruction_loss": avg_reconstruction_loss,
                            "train_dice_loss": avg_dice_loss}
        tensorboard_logs["step"] = self.current_epoch
        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        
        # already averaged over batch (different methods available)
        dice_coeff = channel_wise_dice_score(pred, y)

        # average over batch
        hausdorff = compute_hausdorff_distance(pred, y, include_background=True, percentile=95).mean(0)

        return {"dice_coeff": dice_coeff.cpu(), "hausdorff": hausdorff.cpu()}


    def validation_epoch_end(self, outputs) -> Dict:
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
        num_classes = len(self.channel_to_class)
        dice_coeff = self.dice_metric(pred, y)

        # average over batch
        hausdorff = compute_hausdorff_distance(pred, y, include_background=True, percentile=95).mean(0)
        
        dice = {"val_dice_coeff": {self.channel_to_class[i]: dice_coeff[i] for i in range(num_classes)}}
        hausdorff = {"val_hausdorff": {self.channel_to_class[i]: hausdorff[i] for i in range(num_classes)}}
        return {"dice_coeff": dice, "hausdorff": hausdorff}
