from typing import Dict, Sequence
import torch

from src.models.UNet3D_v1 import *
from src.models.double_u_net import *
from losses.BCEDiceLoss import BCEDiceLoss
from lightningmodules.BraTSLightning import BraTSLightning

from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

class UnetLightning(BraTSLightning):
    def __init__(self, modalities: int = 4, start_channels: int = 16,
                 num_classes: int = 3,
                 total_iterations: int = 300, 
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5) -> None:
        """3D Unet standard model

        Args:
            modalities (int, optional): Number of MRI modes. Defaults to 4.
            start_channels (int, optional): Num Channels after first conv. Defaults to 16.
            num_classes (int, optional): Output (tumor) classes. Defaults to 3.
            total_iterations (int, optional): max epochs. Defaults to 300.
            learning_rate (float, optional): t0 learning rate. Defaults to 1e-4.
            weight_decay (float, optional): parameter regularization. Defaults to 1e-5.
        """        
        super().__init__()
        self.model = UNet3d(in_channels=modalities,
                            n_classes=num_classes, 
                            n_channels=start_channels)
        self.bce_dice_loss = BCEDiceLoss()
        self.total_iterations = total_iterations
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()



    def forward(self, x):
        pred = self.model(x)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.total_iterations, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        total_loss =  self.bce_dice_loss(y_hat, y)
        return {"loss": total_loss.cpu()}
 
    def training_epoch_end(self, outputs: Sequence[Dict[str, torch.Tensor]]) -> Dict:
        avg_total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {"train_total_loss": avg_total_loss}
        tensorboard_logs["step"] = self.current_epoch
        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)

