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
    train_path = "/cluster/51/emre/project/data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    data = BraTSDataset(train_path, training=False)
    gen = torch.Generator()
    gen.manual_seed(0)
    train, val = torch.utils.data.random_split(
        data, [0.9, 0.1],
        generator=gen
    )

    val_loader = DataLoader(val, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)
    model = VAELightning.load_from_checkpoint("/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/runs/VAE/Dice_AVG_Second/best_models/name=0_epoch=92_val_avg_overall_dice=0.83.ckpt", 
                                              volume_shape=data.crop_size, 
                                              modalities=4, start_channels=32, num_classes=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test(model, val_loader, experiment_name="val_with_augmentation", device=device)

    # init trainer with whatever options
    # trainer = Trainer(accelerator="cpu", devices=1)

    # test (pass in the model)
    #
    #trainer.test(model, val_loader, ckpt_path="/home/ek/Desktop/ML3d/OurProject/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/name=0_epoch=92_val_avg_overall_dice=0.83.ckpt")

if __name__ == "__main__":
    main()