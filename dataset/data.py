import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import skimage.transform as skTrans
import configparser
import random
import warnings
warnings.filterwarnings('ignore')

class Brats2021(Dataset):

    def __init__(self, patients_dir, crop_size, modes, dataset:str=None, model_name:str=None):
        self.patients_dir = patients_dir
        self.modes = modes # scan_types
        self.dataset = dataset
        self.crop_size = crop_size
        self.model_name = model_name

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        if not self.dataset=="test":
            modes = list(self.modes) + ['seg'] # scan_types + seg
        else:
            modes = list(self.modes)
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + "_" + mode + ".nii.gz")
            volume = nib.load(volume_path).get_data()
            if self.dataset == "test":
                volume = skTrans.resize(volume, self.crop_size, order=1, preserve_range=True)

            if not mode=="seg":  #apply normalization to the input images 
                volume = self.normalize(volume)
            volumes.append(volume)
        
        if not self.dataset=="test":
            seg_volume = volumes[-1] # last element is seg mask
            volumes = volumes[:-1]
            volume, seg_volume = self.aug_sample(volumes, seg_volume)
            """
            1) "wt_volume" which is equal to 1 (True) wherever the "seg_volume" is greater than 0. 
                This is likely selecting all voxels that correspond to necrotic (dead) tissue or unenhanced tumor regions.

            2) "tc_volume" which is equal to 1 (True) wherever "seg_volume" is equal to 4 or 1. 
                This is likely selecting all voxels that correspond to the enhancing tumor core or active tumor regions.

            3) The last line creates a new variable "et_volume" which is equal to 1 (True) wherever "seg_volume" is equal to 4. 
                This is likely selecting all voxels that correspond to the enhancing tumor core specifically.
            """
            
            wt_volume = seg_volume > 0 
            tc_volume = np.logical_or(seg_volume == 4, seg_volume == 1)
            et_volume = (seg_volume == 4)
            seg_volume = [wt_volume, tc_volume, et_volume] # seg.shape = [3 h w d]
            seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

            input_data = torch.tensor(volume.copy(), dtype=torch.float)
            mask_data = torch.tensor(seg_volume.copy(), dtype=torch.float)

            return (input_data, mask_data)

        else:
            # resize test data
            x = np.stack(volumes, axis=0)  
            input_data = torch.tensor(x.copy(), dtype=torch.float)
            return (input_data)

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]
        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.dataset=="train":
            # crop volume
            x, y = self.random_crop(x,y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)

        elif self.dataset=="val":
            x, y = self.center_crop(x,y)

        return x,y

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
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normalize(self, x):
        return (x - x.min()) / (x.max()- x.min())


def split_dataset(data_root):
    patients_dir = glob.glob(os.path.join(data_root, "BraTS2021*"))
    n_patients = len(patients_dir)
    print(f"total patients: {n_patients}")
    train_patients_list, val_patients_list = train_test_split(patients_dir, test_size=0.20, random_state=42)
    print(f"train patients: {len(train_patients_list)}, val patients: {len(val_patients_list)}")
    return train_patients_list, val_patients_list

def make_data_loaders(mode:str, model_name:str):
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['params']
    modes = params['modes'].split(",")
    shapes = params['input_shape'].split(",")
    input_shape = (int(shapes[0]), int(shapes[1]), int(shapes[2]))
    loaders = {}
    if not mode == "test":
        data_root = params['train_dataset']
        # get patients dir from the csv files 
        train_df = pd.read_csv("dataset/train_data.csv")
        train_list = train_df['train'].values.tolist()
        val_df = pd.read_csv("dataset/val_data.csv")
        val_list = val_df['val'].values.tolist()
        #print("train len:", len(train_list))
        # create dataset object
        train_ds = Brats2021(train_list, crop_size=input_shape, modes=modes, dataset="train", model_name=model_name)
        val_ds = Brats2021(val_list, crop_size=input_shape, modes=modes, dataset="val", model_name=model_name)
        # create dataloaders
        loaders = {}
        loaders['train'] = DataLoader(train_ds, batch_size=int(params['batch_size']),
                                    num_workers=int(params['num_workers']),
                                    pin_memory=True,
                                    shuffle=True)
        loaders['val'] = DataLoader(val_ds, batch_size=int(params['batch_size']),
                                    num_workers=int(params['num_workers']),
                                    pin_memory=True,
                                    shuffle=False)
    else:
        data_root = params['test_dataset']
        test_list = glob.glob(os.path.join(data_root, "BraTS2021*"))
        n_patients = len(test_list)
        print(f"test patients: {n_patients}")
        test_ds = Brats2021(test_list, crop_size=input_shape, modes=modes, dataset="test", model=model)
        loaders = {}
        loaders['test'] = DataLoader(test_ds, batch_size=int(params['batch_size']),
                                    num_workers=int(params['num_workers']),
                                    pin_memory=True,
                                    shuffle=True)

    return loaders

def create_train_val_set(data_root):
    train_list, val_list = split_dataset(data_root=data_root)
    dict_train = {'train': train_list}
    dict_val = {'val': val_list }
    df_train = pd.DataFrame(dict_train) 
    df_train.to_csv("dataset/train_data.csv")
    df_val = pd.DataFrame(dict_val) 
    df_val.to_csv("dataset/val_data.csv")
    print("train and validation sets are saved in csv file format")

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['params']
    train_dataset = params['train_dataset']
    #create_train_val_set(train_dataset)

    #modes = params['modes'].split(",")
    #shapes = params['input_shape'].split(",")
    #input_shape = (int(shapes[0]), int(shapes[1]), int(shapes[2]))
    #data_root = params['data_root']

    #train_list, val_list = split_dataset(data_root=data_root)
    #train_ds = Brats2021(train_list, crop_size=input_shape, modes=modes, train=True)
    #val_ds = Brats2021(val_list, crop_size=input_shape, modes=modes, train=False)
    
    loaders = make_data_loaders(mode="train", model_name="double_unet")
    train_loader = loaders['train']
    print(len(train_loader))
    input_image, mask_image = next(iter(train_loader))
    #print(np.unique(mask))
    print(input_image.shape)
    print(mask_image.shape)
    #train_list, val_list = split_dataset(cfg.DATASET.DATA_ROOT, cfg.DATASET.NUM_FOLDS, cfg.DATASET.SELECT_FOLD)
    #train_ds = Brats2018(train_list, crop_size=cfg.DATASET.INPUT_SHAPE, modes=cfg.DATASET.USE_MODES, train=True)
    #val_ds = Brats2018(val_list, crop_size=cfg.DATASET.INPUT_SHAPE, modes=cfg.DATASET.USE_MODES, train=False)

#main()