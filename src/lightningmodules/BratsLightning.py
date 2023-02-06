from typing import Dict, Sequence
import pytorch_lightning as pl
import torch

from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

class BratsLightning(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.channel_to_class = {0: "WT", 1: "TC", 2: "ET"}

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

    def inference_with_metrics(self, batch: Sequence[torch.Tensor]) -> Dict:
        x, y = batch
        pred = self.forward(x)
        
        # already averaged over batch (different methods available)
        dice_coeff = compute_dice(pred, y, ignore_empty=False)

        # average over batch
        hausdorff = compute_hausdorff_distance(pred, y, include_background=True, percentile=95)

        return {"dice_coeff": dice_coeff.cpu(), "hausdorff": hausdorff.cpu()}

    def aggregate_metrics(self, outputs: Dict) -> Dict:
        num_classes = len(self.channel_to_class)

        avg_dice_coeff = torch.cat([x['dice_coeff'] for x in outputs]).mean(0)
        avg_hausdorff_distance = torch.cat([x['hausdorff'] for x in outputs])

        # throw away nan and inf values:
        # some true masks may lack one or more of the 3 classes
        # this leads to overall nan in the average
        hd = []
        for i in range(num_classes):
            hd.append(avg_hausdorff_distance[avg_hausdorff_distance[:, i].isfinite(), i].mean())

        dice = {"val_dice_coeff": {self.channel_to_class[i]: avg_dice_coeff[i] for i in range(num_classes)}}
        hausdorff = {"val_hausdorff": {self.channel_to_class[i]: hd[i] for i in range(num_classes)}}
        avg_overall_dice = {"val_avg_overall_dice": avg_dice_coeff.mean()}
        dice.update(hausdorff)
        dice.update(avg_overall_dice)
        return dice

    def validation_step(self, batch, batch_idx):
        metrics = self.inference_with_metrics(batch)
        return metrics

    def validation_epoch_end(self, outputs):
        tensorboard_logs = self.aggregate_metrics(outputs)
        avg_overall_dice = tensorboard_logs["val_avg_overall_dice"]

        tensorboard_logs["step"] = self.current_epoch

        self.logger.log_metrics(tensorboard_logs, step=self.current_epoch)
        self.log("val_avg_overall_dice", avg_overall_dice)

    def test_step(self, batch, batch_idx):
        metrics = self.inference_with_metrics(batch)
        return metrics

    def test_step_end(self, outputs):
        # use some file writing in future
        metrics =  self.aggregate_metrics(outputs)
        print(f"Dice_Scores:")
        for i in range(len(self.channel_to_class)):
            print(f"{self.channel_to_class[i]}: {metrics['val_dice_coeff'][i]}")
        print(f"Hausdorff Distances:")
        for i in range(len(self.channel_to_class)):
            print(f"{self.channel_to_class[i]}: {metrics['val_hausdorff'][i]}")