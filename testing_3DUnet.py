from src.models.UNet3D_Lightning import UNet3D_Lightning
import torch

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from src.dataset.BraTSDataset import BraTSDataset
from src.models.VAELightning import VAELightning

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

from testing_legacy import test

def main():
    # test_path = "/home/ek/Desktop/BraTSData/RSNA_ASNR_MICCAI_BraTS2021_ValidationData"
    # base = "/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/runs/VAE/"
    train_path = "/home/ek/Desktop/BraTSData/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    data = BraTSDataset(train_path, training=False)
    gen = torch.Generator()
    gen.manual_seed(0)
    train, val = torch.utils.data.random_split(
        data, [0.9, 0.1],
        generator=gen
    )

    val_loader = DataLoader(val, batch_size=1, num_workers=8, pin_memory=False, shuffle=False)
    model = UNet3D_Lightning.load_from_checkpoint("/home/ek/Desktop/ML3d/OurProject/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/name=3dunet_epoch=36_val_avg_overall_dice=0.84.ckpt", 
                                              model_name="3dunet",
                                              volume_shape=data.crop_size,
                                              modalities=4, start_channels=16, num_classes=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test(model, val_loader, experiment_name="test3dUnet", device=device)

    # init trainer with whatever options
    # trainer = Trainer(accelerator="cpu", devices=1)

    # test (pass in the model)
    #
    #trainer.test(model, val_loader, ckpt_path="/home/ek/Desktop/ML3d/OurProject/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/name=0_epoch=92_val_avg_overall_dice=0.83.ckpt")

if __name__ == "__main__":
    main()