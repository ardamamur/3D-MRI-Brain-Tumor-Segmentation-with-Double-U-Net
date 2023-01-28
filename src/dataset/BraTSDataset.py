# based on: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/data.py

import random
import torch
import numpy as np
import os
import nibabel as nib
import torchvision
import torchio as tio
from monai import transforms as montransforms

from pathlib import Path

from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, path, crop_size=(160, 192, 128), training=True) -> None:
        super().__init__()
        self.path = Path(path)
        self.patient_root_dirs = os.listdir(self.path)

        # careful: order matters. Last one needs to be seg
        self.modes = ["t1", "t2", "t1ce", "flair", "seg"]
        self.training = training
        self.crop_size = crop_size

        self.transformations = self.get_transformations()

    def __len__(self):
        return len(self.patient_root_dirs)

    def __getitem__(self, index):
        patient_dir = self.patient_root_dirs[index]
        im_volumes = []
        seg_volume = None
        rescale = tio.RescaleIntensity((0, 1))
        for m in self.modes:
            vol = nib.load(str(self.path / patient_dir / patient_dir) + "_" + m + ".nii.gz").get_data()
            if not m == "seg":
                im_volumes.append(vol)
            else:
                seg_volume = vol
        im_volumes = np.stack(im_volumes)

        # todo: try standardization
        im_volumes = rescale(im_volumes)

        # get one hot labels for each class per channel
        seg_volume = self.get_segmentation_labels(seg_volume)

        volumes = np.concatenate([im_volumes, seg_volume])

        if self.training:
            volumes = self.transformations(volumes)


        return volumes[:-3], volumes[-3:]

    def get_segmentation_labels(self, seg_volume):
        # https://arxiv.org/pdf/1811.02629.pdf
        # 
        # whole tumor
        wt_volume = seg_volume > 0

        # Union of labels 1, 3, and 4, but 3 was removed and merged
        # with 1
        tc_volume = np.logical_or(seg_volume == 4, seg_volume == 1)

        #enhanacing tumor
        et_volume = (seg_volume == 4)
        seg_volume = [wt_volume, tc_volume, et_volume]
        seg_volume = np.stack(seg_volume, axis=0).astype("float32")
        return seg_volume

    def get_transformations(self):
        """
        todo: maybe add also channel wise intensity shift
        """
        transformations = tio.Compose([montransforms.RandSpatialCrop(roi_size=self.crop_size, random_center=True, random_size=False),
                                       tio.RandomFlip(axes=(0, 1, 2)),
                                       tio.RandomAffine(scales=0.1, isotropic=True)
                                       ])
        return transformations


    def random_crop(self,x,y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg