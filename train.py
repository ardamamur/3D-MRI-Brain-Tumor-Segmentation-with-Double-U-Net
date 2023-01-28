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
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train():
    def __init__(self, hyper_parameters, model_name):
        self.model_name = model_name
        if self.model_name == "3dunet":
            self.model =  UNet3d(in_channels=hyper_parameters["in_channels"], n_classes=3, n_channels=hyper_parameters['init_channels'])
        else:
            self.model =  DoubleUNet3d(in_channels=hyper_parameters["in_channels"], n_classes=3)
        self.model =  UNet3d(in_channels=hyper_parameters["in_channels"], n_classes=3, n_channels=hyper_parameters['init_channels'])
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyper_parameters['lr'], weight_decay=hyper_parameters['weight_decay'])
        self.scheduler = PolyLR(self.optimizer, max_epoch=hyper_parameters['num_epochs'], power=0.9)
        self.criterion = BCEDiceLoss()
        self.phases = ["train", "val"]
        self.num_epochs = hyper_parameters["num_epochs"]
        self.accumulation_steps = hyper_parameters['accumulation_steps']
        self.loaders = make_data_loaders(mode="train", model_name=self.model_name)
        self.best_loss = float("inf")
        self.n_steps = hyper_parameters['n_steps']
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores_WT = {phase: [] for phase in self.phases}
        self.dice_scores_TC = {phase: [] for phase in self.phases}
        self.dice_scores_ET = {phase: [] for phase in self.phases}
        self.jaccard_scores_WT = {phase: [] for phase in self.phases}
        self.jaccard_scores_TC = {phase: [] for phase in self.phases}
        self.jaccard_scores_ET = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(device)
        targets = targets.to(device)
        logits = None
        loss = None
        if self.model_name == "3dunet":
            logits = self.model(images)
            loss = self.criterion(logits, targets)
        else:
            logit1, logit2 = self.model(images)
            loss1 = self.criterion(logit1, targets[:,0].unsqueeze(1))
            loss2 = self.criterion(logit2, targets)
            loss = loss1+ loss2
            logits = logit2
        return loss, logits

    def _do_epoch(self, epoch:int, phase:str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.model.train() if phase == "train" else self.model.eval()
        meter = Meter()
        loader = self.loaders[phase]
        total_batches = len(loader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, (images, targets) in enumerate(loader):
            #print(images.shape)
            #print(targets.shape)
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(),targets.detach().cpu())
            if (itr + 1) % self.n_steps == 0:
                print("BCEDice loss: ", loss.item())
        
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        metrics = meter.get_metrics()
        print("###################################")
        print("accumulated loss: ", epoch_loss)

        print("WT dice: " , metrics['dice_WT'], 
            " TC dice: " , metrics['dice_TC'] ,
            " ET dice: ", metrics['dice_ET'])

        print("WT jaccard: ", metrics['iou_WT'],
            " TC jaccard: ", metrics['iou_TC'],
            " ET jaccard: ", metrics['iou_ET'])

        print("###################################")
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores_WT[phase].append(metrics['dice_WT'])
        self.dice_scores_TC[phase].append(metrics['dice_TC'])
        self.dice_scores_ET[phase].append(metrics['dice_ET'])

        self.jaccard_scores_WT[phase].append(metrics['iou_WT'])
        self.jaccard_scores_TC[phase].append(metrics['iou_TC'])
        self.jaccard_scores_ET[phase].append(metrics['iou_ET'])
        
        return epoch_loss


    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()
            if val_loss < self.best_loss:
                print("Saved new checkpoint")
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
            print()
        self._save_train_history()       


    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(self.model.state_dict(),
                f"last_epoch_model.pth")

        logs_ = [self.losses,
                self.dice_scores_WT, self.dice_scores_TC, self.dice_scores_ET,
                self.jaccard_scores_WT, self.jaccard_scores_TC, self.jaccard_scores_ET]

        log_names_ = ["_loss", "_dice_WT", "_dice_TC", "dice_ET",
                    "_jaccard_WT", "_jaccard_TC", "_jaccard_ET"]

        logs = [logs_[i][key] for i in list(range(len(logs_)))
                        for key in logs_[i]]
        log_names = [key+log_names_[i] 
                    for i in list(range(len(logs_))) 
                    for key in logs_[i]
                    ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv("train_log.csv", index=False)

    def load_predtrain_model(self, state_path: str):
        self.model.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")


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
        "input_shape" : (int(shapes[0]), int(shapes[1]), int(shapes[2])),
        "accumulation_steps" : 4 / int(params['batch_size']),
        "n_steps" : 5
    
    }
    print("hyper parameters:", hyper_parameters)
    
    trainer = Train(hyper_parameters=hyper_parameters, model_name="double_unet")
    
    if params['pre_trained'] == "True":
        trainer.load_predtrain_model(params['pre_trained_path'])
        
        # if need - load the logs.      
        train_logs = pd.read_csv("train_log.csv")
        trainer.losses["train"] =  train_logs.loc[:, "train_loss"].to_list()
        trainer.losses["val"] =  train_logs.loc[:, "val_loss"].to_list()


        trainer.dice_scores_WT["train"] = train_logs.loc[:, "train_dice_WT"].to_list()
        trainer.dice_scores_TC["train"] = train_logs.loc[:, "train_dice_TC"].to_list()
        trainer.dice_scores_ET["train"] = train_logs.loc[:, "train_dice_ET"].to_list()

        trainer.dice_scores_WT["val"] = train_logs.loc[:, "val_dice_WT"].to_list()
        trainer.dice_scores_TC["val"] = train_logs.loc[:, "val_dice_TC"].to_list()
        trainer.dice_scores_ET["val"] = train_logs.loc[:, "val_dice_ET"].to_list()


        trainer.jaccard_scores_WT["train"] = train_logs.loc[:, "train_jaccard_WT"].to_list()
        trainer.jaccard_scores_TC["train"] = train_logs.loc[:, "train_jaccard_TC"].to_list()
        trainer.jaccard_scores_ET["train"] = train_logs.loc[:, "train_jaccard_ET"].to_list()

        trainer.jaccard_scores_TC["val"] = train_logs.loc[:, "val_jaccard"].to_list()
        trainer.jaccard_scores_TC["val"] = train_logs.loc[:, "val_jaccard"].to_list()
        trainer.jaccard_scores_ET["val"] = train_logs.loc[:, "val_jaccard"].to_list()

    print("START TRAINING")         
    trainer.run()
    print("TRAINING DONE. LOGS ARE SAVED..")


main()
