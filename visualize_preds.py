import torch

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split

from src.dataset.BraTSDataset import BraTSDataset
from src.models.VAELightning import VAELightning
from src.models.UNet3D_Lightning import UNet3D_Lightning

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from tqdm import tqdm

#from testing_legacy import test
model_name = "3dunet"

train_path = "/home/ardamamur/TUM/ML3D/dataset/train"
data = BraTSDataset(train_path, training=False)
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
best_model_path = "runs/3dunet/best_models/name=0_epoch=46_val_avg_overall_dice=0.85.ckpt"
val_loader = DataLoader(val, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)
model = UNet3D_Lightning.load_from_checkpoint(best_model_path, model_name=model_name, volume_shape=data.crop_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions = []
model.to(device)
model.eval()
with torch.no_grad():
    for train_features, train_labels in tqdm(val_loader):
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        y_hat = model(train_features)
        predictions.append(y_hat)

print(predictions.shape)





