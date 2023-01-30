import torch

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from src.dataset.BraTSDataset import BraTSDataset
from src.models.VAELightning import VAELightning

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

def main():
    test_path = "/home/ek/Desktop/BraTSData/RSNA_ASNR_MICCAI_BraTS2021_ValidationData"
    base = "/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/runs/VAE/"
    train_path = "/home/ek/Desktop/BraTSData/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    data = BraTSDataset(train_path, training=True)
    gen = torch.Generator()
    gen.manual_seed(0)
    train, val = torch.utils.data.random_split(
        data, [0.9, 0.1],
        generator=gen
    )

    val_loader = DataLoader(val, batch_size=1, num_workers=1, pin_memory=False, shuffle=False)
    model = VAELightning.load_from_checkpoint("/home/ek/Desktop/ML3d/OurProject/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/name=0_epoch=92_val_avg_overall_dice=0.83.ckpt", 
                                              volume_shape=data.crop_size, 
                                              modalities=4, start_channels=32, num_classes=3)

    i = 0
    for train_features, train_labels in val_loader:
        y_hat = model(train_features)
        print(y_hat.dtype)
        print(y_hat.shape)
        print(train_labels.shape)
        print(compute_dice(y_hat, train_labels))
        print(compute_hausdorff_distance(y_hat, train_labels, include_background=True, percentile=95))
        i += 1
        if i > 15:
            break

    # init trainer with whatever options
    # trainer = Trainer(accelerator="cpu", devices=1)

    # test (pass in the model)
    #
    #trainer.test(model, val_loader, ckpt_path="/home/ek/Desktop/ML3d/OurProject/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/name=0_epoch=92_val_avg_overall_dice=0.83.ckpt")

if __name__ == "__main__":
    main()