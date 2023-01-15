import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
import numpy as np
import nibabel as nib # to load and save neuroimaging data
from sklearn.model_selection import train_test_split

class ImageReader:
    """
    ImageReader class. The load_patient_scan method of the ImageReader class reads in all the scan types 
    specified in scan_types and concatenates them along the channel dimension before returning them.
    
    Note:
        A common approach is to train a model using all the modalities, 
        concatenating the different modalities along the channel dimension of the input tensor. 
        This allows the model to take advantage of the information provided by each modality.
    """
    def __init__(self, root:str, img_size:int=256, normalize:bool=False, single_class:bool=False, scan_types:list=['flair', 't1', 't1ce', 't2']):
        self.scan_types = scan_types
        self.pad_size = 256 if img_size > 256 else 224
        self.resize = A.Compose(
            [
                A.PadIfNeeded(min_height=self.pad_size, min_width=self.pad_size, value=0),
                A.Resize(img_size, img_size)
            ]
        )
        self.normalize=normalize
        self.single_class=single_class
        self.root=root
        
    def read_file(self, path:str) -> dict:
        raw_image = nib.load(path).get_fdata()
        raw_mask = nib.load(path.replace(path.split('_')[-1], 'seg.nii.gz')).get_fdata()
        processed_frames, processed_masks = [], []
        for frame_idx in range(raw_image.shape[2]):
            frame = raw_image[:, :, frame_idx]
            mask = raw_mask[:, :, frame_idx]
            if self.normalize:
                if frame.max() > 0:
                    frame = frame/frame.max()
                frame = frame.astype(np.float32)
            else:
                frame = frame.astype(np.uint8)
            resized = self.resize(image=frame, mask=mask)
            processed_frames.append(resized['image'])
            processed_masks.append(1*(resized['mask'] > 0) if self.single_class else resized['mask'])
        return {
            'scan': np.stack(processed_frames, 0),
            'segmentation': np.stack(processed_masks, 0),
            'orig_shape': raw_image.shape
        }
    
    def load_patient_scan(self, idx:int, segmentation:np.ndarray) -> dict:
        patient_id = str(idx).zfill(5)
        scan_list = []
        for scan_type in self.scan_types:
            scan_filename = f'{self.root}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_{scan_type}.nii.gz'
            scan_list.append(nib.load(scan_filename).get_fdata())
        return {
            'scan': np.concatenate(scan_list, axis=0),
            'segmentation': segmentation,
            'orig_shape': scan_list[0].shape
        }

class BratsDataLoader(Dataset):
    """
    BratsDataLoader class takes the same arguments as the previous version, 
    but it also takes an additional argument scan_types, which is passed to the ImageReader class. 
    The load_patient_scan method of the ImageReader class reads in all the scan types specified in scan_types 
    and concatenates them along the channel dimension before returning them.
    """
    def __init__(self, root:str, img_size:int=256, normalize:bool=False, single_class:bool=False, scan_types:list=['flair', 't1', 't1gd', 't2'], patient_ids=None):
        self.image_reader = ImageReader(root, img_size, normalize, single_class, scan_types)
        self.root = root
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        scan_filename = f'{self.root}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_seg.nii.gz'
        segmentation = nib.load(scan_filename).get_fdata()
        #print(segmentation)
        patient_data = self.image_reader.load_patient_scan(patient_id, segmentation)
        scan = patient_data['scan']
        #print(scan)
        segmentation = patient_data['segmentation']
        return {'scan': torch.from_numpy(scan), 'segmentation': torch.from_numpy(segmentation)}



def test_data_loader(dataset):

    patient_ids = [d.split("_")[-1] for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
    train_ids, eval_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(eval_ids, test_size=0.1, random_state=42)


    data_loader = DataLoader(BratsDataLoader(root=dataset, img_size=256, normalize=True, single_class=True, scan_types=['flair', 't1', 't1ce', 't2'], patient_ids=train_ids),
                         batch_size=16, shuffle=True, num_workers=0)

    data = next(iter(data_loader))

    # Check the shape of the data
    scan = data['scan']
    segmentation = data['segmentation']
    print("Shape of scan: ", scan.shape)
    print("Shape of segmentation: ", segmentation.shape)

    # Check the data type of the data
    print("Data type of scan: ", scan.dtype)
    print("Data type of segmentation: ", segmentation.dtype)


root = "/home/ardamamur/TUM/ML3D/dataset/"
dataset = root + "train"
test_data_loader(dataset)
