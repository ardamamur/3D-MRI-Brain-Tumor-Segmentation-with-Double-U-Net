from typing import Dict, Sequence
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

from VAE_net import Decoder, Encoder, VariationalDecoder
from losses.VAELoss import VAELoss
from losses.DiceLoss import DiceLoss

from metrics.metrics import channel_wise_dice_score
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

class VAELightning(pl.LightningModule):
    def __init__(self, volume_shape, channel_to_class, modalities=4, start_channels=32, num_classes=3) -> None:
        super().__init__()
        self.encoder = Encoder(modalities=modalities, start_channels=start_channels)
        self.decoder = Decoder(in_channels=32*8, num_classes=num_classes)
        self.vae_decoder = VariationalDecoder(32*8, volume_shape, num_classes)
        
        self.vae_loss = VAELoss()
        self.dice_loss = DiceLoss()

        self.channel_to_class = channel_to_class

        self.epoch = 0

    def _pred(self):
        pass

    def forward(self, x):
        enc = self.encoder(x)
        pred = self.decoder(enc)
        return torch.sigmoid(pred)

    def training_step(self, batch):
        x, y = batch
        latent = self.encoder(x)
        y_hat = self.decoder(latent)
        x_hat, mu, logvar = self.vae_decoder(latent)
        total_loss, kl_loss, reconstruct_loss, dice_loss = self.vae_loss(x_hat,
                                                                         y_hat,
                                                                         x,
                                                                         y,
                                                                         mu,
                                                                         logvar)
        return {"loss": total_loss, "kl_loss": kl_loss, "reconstruct_loss": reconstruct_loss,
                "dice_loss": dice_loss}

    def on_train_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> Dict:
        assert outputs[0]["kl_loss"].requires_grad == False
        avg_kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        avg_total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_reconstruction_loss = torch.stack([x['reconstruct_loss'] for x in outputs]).mean()
        avg_dice_loss = torch.stack([x['dice_loss'] for x in outputs]).mean()
        tensorboard_logs = {"train_kl_loss": avg_kl_loss, "train_total_loss": avg_total_loss,
                            "train_reconstruction_loss": avg_reconstruction_loss,
                            "train_dice_loss": avg_dice_loss}
        tensorboard_logs["step"] = self.current_epoch
        return {"log": tensorboard_logs}

    def validation_step(self, batch):
        x, y = batch
        pred = self.forward(x)
        
        dice_coeff = channel_wise_dice_score(pred, y)
        hausdorff = compute_hausdorff_distance(pred, y)

        return {"dice_coeff": dice_coeff, "hausdorff": hausdorff}

    
    def validation_epoch_end(self, outputs) -> Dict:
        num_classes = len(self.channel_to_class)

        avg_dice_coeff = torch.stack([x['dice_coeff'] for x in outputs]).mean(0)
        avg_hausdorff_distance = torch.stack([x['hausdorff'] for x in outputs]).mean(0)

        dice = {"val_dice_coeff": {self.channel_to_dice(i): avg_dice_coeff[i] for i in num_classes}}
        hausdorff = {"val_hausdorff": {self.channel_to_dice(i): avg_hausdorff_distance[i] for i in num_classes}}

        tensorboard_logs = dice | hausdorff

        tensorboard_logs["step"] = self.current_epoch

        return {"log": tensorboard_logs}