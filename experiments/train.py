from pytorch_lightning.cli import LightningCLI
from src.lightningmodules.BraTSLightning import BraTSLightning
from src.lightningmodules.DoubleUnetLightning import DoubleUnetLightning
from src.lightningmodules.VAELightning import VAELightning
from src.lightningmodules.UnetLightning import UnetLightning
from src.dataset.BraTSDataLightning import BraTSDataLightning

from pytorch_lightning.callbacks import ModelCheckpoint

def cli_main():

    cli = LightningCLI(datamodule_class=BraTSDataLightning)

if __name__ == "__main__":
    cli_main()