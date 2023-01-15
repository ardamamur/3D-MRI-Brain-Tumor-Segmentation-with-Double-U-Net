from dataset.data_loader2 import *
from models.loss import *
from models.UNet import *
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau




def train(model, train_dataloader, val_dataloader, device, config):
    loss_criterion = BCEDiceLoss()
    loss_criterion.to(device)
    accumulation_steps = config['accumulation_steps'] // config['batch_size']
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
    meter = Meter()
    model.train()

    best_loss_val = np.inf

    training_loss_running = 0.
    
    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            images, targets = batch['image'], batch['mask']
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            

            preds = model(images)
            loss_total = loss_criterion(preds, targets)
            loss = loss / accumulation_steps
            loss_total.backward()
            optimizer.step()

            training_loss_running += loss.item()
            meter.update(preds.detach().cpu(),
                         targets.detach().cpu()
                        )
            iteration = epoch * len(train_dataloader) + batch_idx
            
            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                train_loss_running = (training_loss_running * accumulation_steps) / len(train_dataloader)
                dice_loss, iou = meter.get_metrics()
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
                print(f'[{epoch:03d}/{batch_idx:05d}] dice_loss: {dice_loss / config["print_every_n"]:.6f}')
                print(f'[{epoch:03d}/{batch_idx:05d}] iou: {iou / config["print_every_n"]:.6f}')
                train_loss_running = 0.

            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                model.eval()
                loss_val = 0.
                for batch_val in val_dataloader:
                    images, targets = batch_val['image'], batch_val['mask']
                    images = images.to(device)
                    targets = targets.to(device)

                    with torch.no_grad():
                        preds = model(images)
                        meter.update(preds.detach().cpu(),
                         targets.detach().cpu()
                        )
                        loss_val += loss_criterion(preds, targets).item()

                loss_val = (loss_val * accumulation_steps) / len(val_dataloader)
                dice_loss, iou = meter.get_metrics()
                if loss_val < best_loss_val:
                    torch.save(model.state_dict(), f'model_best.ckpt')
                    best_loss_val = loss_val

                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')
                print(f'[{epoch:03d}/{batch_idx:05d}] dice_loss: {dice_loss / config["print_every_n"]:.6f}')
                print(f'[{epoch:03d}/{batch_idx:05d}] iou: {iou / config["print_every_n"]:.6f}')

                model.train()
        scheduler.step()
                        

def main(config):
    train_dataloader, val_dataloader, test_data_loader = get_dataloader(dataset=BratsDataset, root=config['root'], fold=0)
    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    model = UNet3d(in_channels=4, n_classes=3, n_channels=24)

    train(model, train_dataloader, val_dataloader, device, config)


config = {
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'batch_size': 4,
    'accumulation_steps' : 4,
    'learning_rate':5e-4,
    'max_epochs': 3,
    'print_every_n': 50,
    'validate_every_n' : 25,
    'root' : "/home/ardamamur/TUM/ML3D/dataset/train"
}

main(config)