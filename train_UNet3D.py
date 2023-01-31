import torch

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from src.dataset.BraTSDataset import BraTSDataset
from src.models.VAELightning import VAELightning

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    base = "/cluster/51/arda/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/runs/UNet3D/"
    train_path = "/cluster/51/arda/dataset/train"
    data = BraTSDataset(train_path, training=True)
    
    gen = torch.Generator()
    gen.manual_seed(0)
    train, val = torch.utils.data.random_split(
        data, [0.9, 0.1],
        generator=gen
    )

    train_loader = DataLoader(train, batch_size=1, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, num_workers=16, pin_memory=True)
    experiment = base
    logger = TensorBoardLogger(experiment+"/logs/")
    model = VAELightning(data.crop_size)

    checkpoint_best = ModelCheckpoint(
        dirpath=experiment+"/best_models/",
        filename='{name}_{epoch}_{val_avg_overall_dice:.2f}',
        save_top_k=5,
        monitor='val_avg_overall_dice',
        mode='max')

    checkpoint_last = ModelCheckpoint(
        dirpath=experiment+'/last_models/',
        filename='{name}_{epoch}',
        save_top_k=5,
        save_last=True,
        mode="max",
        monitor="step")


    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=20, check_val_every_n_epoch=1,
                    callbacks=[checkpoint_best, checkpoint_last], logger=logger,
    )
    
    model = model.cuda()
    # start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()