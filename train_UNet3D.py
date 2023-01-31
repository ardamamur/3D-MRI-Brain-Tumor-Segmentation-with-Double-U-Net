import torch

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from src.dataset.BraTSDataset_Unet import BraTSDataset_Unet
from src.models.UNet3D_Lightning import UNet3D_Lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main(model_name):
    base = "/cluster/51/arda/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/runs/"
    train_path = "/cluster/51/arda/dataset/train"
    data = BraTSDataset_Unet(train_path, training=True)
    print(len(data))
    proportions = [0.9, 0.1]
    lengths = [int(p * len(data)) for p in proportions]
    lengths[-1] = len(data) - sum(lengths[:-1])
    print(lengths[0])
    print(lengths[1])
    gen = torch.Generator()
    gen.manual_seed(0)
    train, val = torch.utils.data.random_split(
        data, lengths,
        generator=gen
    )

    train_loader = DataLoader(train, batch_size=1, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, num_workers=16, pin_memory=True)
    name = model_name
    experiment = base+name
    logger = TensorBoardLogger(experiment+"/logs/")
    hparams = {
        "volume_shape" : data.crop_size,
        "modalities" : 4,
        "start_channels" : 16,
        "num_classes" : 3,
        "learning_rate" : 1e-4,
        "weight_decay" : 1e-5
    }

    model = UNet3D_Lightning(model_name, data.crop_size)

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


    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=50, check_val_every_n_epoch=1,
                    callbacks=[checkpoint_best, checkpoint_last], logger=logger,
    )
    
    model = model.cuda()
    # start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    model_name = "double_unet"
    main(model_name)

