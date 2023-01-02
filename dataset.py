import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.preprocessing import minmax_scale

class BraTS2021Dataset(Dataset):
    def __init__(self, data_dir, data_split, transform=None):
        self.data_dir = data_dir
        self.data_split = data_split
        self.transform = transform
        self.data_dir = os.path.join(data_dir, data_split)
        self.filenames = [f for f in os.listdir(self.data_dir) if f.endswith(".nii.gz")]
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.data_dir, filename)
        image = nib.load(filepath)
        image_data = image.get_fdata()
        image_data = minmax_scale(image_data, feature_range=(0, 1), axis=(0, 1, 2))
        image_tensor = torch.from_numpy(image_data)
        if "seg" in filename:
            label_tensor = image_tensor
        else:
            label_tensor = None
        if self.transform:
            image_tensor = self.transform(image_tensor)
        sample = {"image": image_tensor, "label": label_tensor}
        return sample

# dataset = BraTS2020Dataset(data_dir, data_split="train", transform=transforms.ToTensor())
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)