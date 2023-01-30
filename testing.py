import torch

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from src.dataset.BraTSDataset import BraTSDataset
from src.models.VAELightning import VAELightning

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    test_path = "/cluster/51/emre/project/data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    base = "/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/runs/VAE/"
    train_path = "/cluster/51/emre/project/data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    data = BraTSDataset(train_path, training=True)
    gen = torch.Generator()
    gen.manual_seed(0)
    train, val = torch.utils.data.random_split(
        data, [0.9, 0.1],
        generator=gen
    )
    model = VAELightning.load_from_checkpoint(
        checkpoint_path="/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/runs/VAE/Dice_AVG_Second/best_models/name=0_epoch=92_val_avg_overall_dice=0.83.ckpt",
    )

    # init trainer with whatever options
    trainer = Trainer(accelerator="gpu", devices=1)

    # test (pass in the model)
    trainer.test(model, val)

if __name__ == "__main__":
    main()