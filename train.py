import configparser
from models.UNet3D_v1 import UNet3d
from solver import PolyLR
from dataset.data import *
#from models.UNet3D_V2 import *
#from models.double_u_net import *
#from models.VAE_net import * 
from models.loss import *
from models.utils import summarize_model
import os
import torch
import nibabel as nib
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(params, model, loaders, optimizer, scheduler, losses, metrics=None):
    n_epohcs = int(params['num_epochs'])
    end = time.time()
    best_dice = 0.0
    #for epoch in range(n_epohcs):


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['params']
    
    print("device:",device)
    modes = params['modes'].split(",")
    shapes = params['input_shape'].split(",")
    hyper_parameters = {
        "lr" : float(params['learning_rate']),
        "weight_decay" : float(params['weight_decay']),
        "dropout" : float(params['dropout']),
        "init_channels" : int(params['init_channels']), # inital output channel of first conv block
        "modes" : modes,
        "in_channels" : len(modes),
        "shapes" : shapes,
        "num_epochs" : int(params['num_epochs']),
        "input_shape" : (int(shapes[0]), int(shapes[1]), int(shapes[2]))
    
    }
    print("hyper parameters:", hyper_parameters)

    model = UNet3d(in_channels=hyper_parameters["in_channels"], n_classes=3, n_channels=hyper_parameters['init_channels'])
    #print(summarize_model(model))
    model.to(device)
    loaders = make_data_loaders(params)
    optimizer = torch.Adam(model.parameters(), lr=hyper_parameters['lr'], weight_decay=hyper_parameters['weight_decay'])
    scheduler = PolyLR(optimizer, max_epoch=hyper_parameters['num_epochs'], power=0.9)
    losses = 
    #train()

main()
