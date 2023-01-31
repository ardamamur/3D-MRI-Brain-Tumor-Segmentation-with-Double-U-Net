from src.dataset.BraTSDataset import BraTSDataset
import torch
import configparser

from src.models.UNet3D_v1 import UNet3d
from src.models.double_u_net import DoubleUNet3d
from testing_legacy import test
from torch.utils.data import DataLoader

import pandas as pd

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['params']

    train_path = "/cluster/51/emre/project/data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"

    val = get_data_from_csv(train_path)
    val_loader = DataLoader(val, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)

    experiment_name = "unet"
    checkpoint = "/cluster/51/emre/project/best_model_3dunet.pth"

    hyperparams = {"init_channels" : int(params['init_channels']),
                    "in_channels" : 4}
    
    model = select_model("unet", hyperparams)
    model.load_state_dict(torch.load(checkpoint))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)


    test(model, val_loader, experiment_name=experiment_name, device=device, apply_sigmoid=True)


def get_data_from_csv(train_path):
    df = pd.read_csv("/cluster/51/emre/project/val_data.csv")
    df = df["val"].str.split("/")
    patients = []
    for d in df:
        patients.append(d[-1])
    
    # take validation split from train data
    val = BraTSDataset(train_path, training=True, patientsdir=patients)
    return val

def select_model(m: str, hyper_parameters: dict) -> torch.nn.Module:
    if m == "unet":
        return UNet3d(in_channels=hyper_parameters["in_channels"], n_classes=3, n_channels=hyper_parameters['init_channels'])
    if m == "double":
        return DoubleUNet3d(in_channels=hyper_parameters["in_channels"], n_classes=3)
if __name__ == "__main__":
    main()