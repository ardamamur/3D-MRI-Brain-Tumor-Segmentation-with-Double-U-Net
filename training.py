from models.UNet import UNet3D, NoNewNet
from models.utils import summarize_model
from dataset.data_loader import *
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from loss_functions.crossentropy import *
from loss_functions.deep_supervision import *
from loss_functions.dice_loss import *
from loss_functions.focal_loss import *
import datetime
import pytz
from datetime import timedelta
today = datetime.datetime.now(pytz.timezone("Europe/Berlin"))
today = today.strftime("%m-%d-%Y-%H%M%S")



def load_model(model_name):
    """
    args:
        model_name : name of the model class
    return:
        torch.model
    """

    if model_name == "unet3d":
        model = UNet3D(in_channels=960, out_channels=1)
    elif model_name == "nonewnet":
        model = NoNewNet(in_channels=960, out_channels=1)
    else:
        raise NotImplementedError
    print(summarize_model(model))
    return model

def train(model_name, data_path, num_epochs, batch_size, lr):

    model = load_model(model_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SoftDiceLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create datasets
    train_dataset = BratsDataLoader(root_dir=data_path, split='train')
    val_dataset = BratsDataLoader(root_dir=data_path, split='val')

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize best validation loss to a large value
    best_val_loss = float('inf')
    best_model_state_dict = None
    print("########## training starting ############")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            scans = data['scan'].to(device)
            labels = data['segmentation'].to(device)

            # Forward pass
            outputs = model(scans)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                scans = data['scan'].to(device)
                labels = data['segmentation'].to(device)

                # Forward pass
                outputs = model(scans)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            best_model_path = "trained_models/" + today + "_best_model.pth"
            torch.save(best_model_state_dict,best_model_path)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    model.load_state_dict(best_model_state_dict)



if __name__ == "__main__":

    model_name = "unet3d"
    data_path = "/home/ardamamur/TUM/ML3D/dataset/train"
    num_epochs = 5
    batch_size = 8
    lr = 1e-4
    train(model_name=model_name, data_path=data_path, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
    print("########## training done ################")