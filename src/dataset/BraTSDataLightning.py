from typing import Sequence
import pytorch_lightning as pl
import json
from src.dataset.BraTSDataset import BraTSDataset
from torch.utils.data import DataLoader

class BraTSDataLightning(pl.LightningDataModule):
    def __init__(self, split: int, data_dir: str = "/cluster/51/emre/project/data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021",
                 split_file: str = "/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/splits/datasplit.json",
                 batch_size: int = 1,
                 crop_size: Sequence[int] = (160, 192, 128)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split = split
        self.split_file = split_file
        self.crop_size = crop_size
        assert split in range(-1, 5)

    def setup(self, stage: str):
        f = open(self.split_file)
        data = json.load(f)

        self.test_patients = data["test"]
        
        if self.split == -1:
            self.train_patients = data["train"]
            # just some random val, since this case happens only when
            # training on all train data in order to do real inference
            # or real evaluation on the final test split or on the
            # synapse platform
            self.val_patients = data["split_0"]["val"]
        else:
            self.train_patients = data[f"split_{self.split}"]["train"]
            self.val_patients = data[f"split_{self.split}"]["val"]

    def train_dataloader(self):
        traindata = BraTSDataset(self.data_dir, crop_size=self.crop_size,
                                 training=True,
                                 patientsdir=self.train_patients)
        return DataLoader(traindata, batch_size=self.batch_size,
                          shuffle=True, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        valdata = BraTSDataset(self.data_dir, crop_size=self.crop_size,
                                 training=True,
                                 patientsdir=self.val_patients)
        return DataLoader(valdata, batch_size=self.batch_size,
                          shuffle=False, num_workers=16, pin_memory=True)

    def test_dataloader(self):
        testdata = BraTSDataset(self.data_dir, crop_size=self.crop_size,
                                training=False,
                                patientsdir=self.val_patients)
        return DataLoader(testdata, batch_size=self.batch_size,
                          shuffle=False, num_workers=16, pin_memory=True)