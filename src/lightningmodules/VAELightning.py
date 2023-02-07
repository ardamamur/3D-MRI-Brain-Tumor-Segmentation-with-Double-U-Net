import torch
from typing import Dict, Sequence

from lightningmodules.BraTSLightning import BraTSLightning
from src.models.VAE_net import Decoder, Encoder, VariationalDecoder
from src.losses.VAELoss import VAELoss
from src.losses.DiceLoss import DiceLoss

class VAELightning(BraTSLightning):
    def __init__(self, volume_shape, modalities=4, start_channels=32, num_classes=3, total_iterations=300,
                learning_rate=1e-4, weight_decay=1e-5) -> None:
        super().__init__()
        self.encoder = Encoder(modalities=modalities, start_channels=start_channels)
        self.decoder = Decoder(in_channels=start_channels*8, num_classes=num_classes)
        self.vae_decoder = VariationalDecoder(start_channels*8, volume_shape, num_classes)
        
        self.vae_loss = VAELoss()
        self.dice_loss = DiceLoss()

        self.total_iterations = total_iterations
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        enc, x1, x2, x3 = self.encoder(x)
        pred = self.decoder(enc, x1, x2, x3)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.total_iterations, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        avg_kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        avg_total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_reconstruction_loss = torch.stack([x['reconstruct_loss'] for x in outputs]).mean()
        avg_dice_loss = torch.stack([x['dice_loss'] for x in outputs]).mean()
        tensorboard_logs = {"train_kl_loss": avg_kl_loss, "train_total_loss": avg_total_loss,
                            "train_reconstruction_loss": avg_reconstruction_loss,
                            "train_dice_loss": avg_dice_loss}
        tensorboard_logs["step"] = self.current_epoch
        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)