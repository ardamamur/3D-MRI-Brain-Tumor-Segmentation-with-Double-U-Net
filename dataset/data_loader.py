# ABOUT DATASET
"""
All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe 
a) native (T1)
b) post-contrast T1-weighted (T1ce), 
c) T2-weighted (T2)
d) T2 Fluid Attenuated Inversion Recovery (FLAIR) volumes.

Annotations comprise the GD-enhancing tumor (ET — label 4), 
the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1).
"""
# TO-DO
"""
Each pixel must be labeled “1” if it is part of one of the classes (NCR/NET — label 1, ED — label 2, ET — label 4),
and “0” if not.
"""

# IMPORT LIBRARIES
from tqdm import tqdm
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
import nibabel as nib
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations import Compose, HorizontalFlip
import warnings
warnings.simplefilter("ignore")

def get_augmentations(phase):
    list_transforms = []
    
    list_trfms = Compose(list_transforms)
    return list_trfms

class BratsDataset(Dataset):
    def __init__(self, root, patient_ids, phase, is_resize: bool=True):
        self.root = root
        self.patient_ids = patient_ids
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.scan_types = ['flair', 't1', 't1ce', 't2']
        self.is_resize = is_resize
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        id_ = patient_id
        # load all modalities
        images = []
        for scan_type in self.scan_types:
            img_path = f'{self.root}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_{scan_type}.nii.gz'
            img = self.load_img(img_path) #.transpose(2, 0, 1)
            
            if self.is_resize:
                img = self.resize(img)
    
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        
        if self.phase != "test":
            mask_path =  f'{self.root}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_seg.nii.gz'
            mask = self.load_img(mask_path)
            
            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)
    
            augmented = self.augmentations(image=img.astype(np.float32), 
                                           mask=mask.astype(np.float32))
            
            img = augmented['image']
            mask = augmented['mask']
        
            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }
        
        return {
            "Id": id_,
            "image": img,
        }
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data
    
    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask

def get_dataloader(
    dataset: torch.utils.data.Dataset,
    root: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
):

    ids = [d.split("_")[-1] for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    train_ids, eval_ids = train_test_split(ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(eval_ids, test_size=0.1, random_state=42)
    
    if phase == "train":
        patient_ids = train_ids
    elif phase == "val":
        patient_ids = val_ids
    else:
        patient_ids = test_ids
    
    dataset = dataset(root, patient_ids, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    return dataloader

def test_data_loader():
    root = "/home/ardamamur/TUM/ML3D/dataset/train"
    dataloader = get_dataloader(dataset=BratsDataset, root=root, phase='train', fold=0)
    print("len:", len(dataloader))

    data = next(iter(dataloader))
    print("patient_id:", data['Id'])
    print("scan_image:",data['image'].shape)
    print("segmentation:", data['mask'].shape)

    img_tensor = data['image'].squeeze()[0].cpu().detach().numpy() 
    mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

test_data_loader()