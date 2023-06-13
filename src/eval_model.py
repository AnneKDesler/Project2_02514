import argparse
import pytorch_lightning as pl
import torch
import os
from src.load_data import get_dataloaders_DRIVE
from src.load_data import get_dataloaders_PH2
from src.model import Model, DilatedNet


def eval(model_src):
    if not os.path.isfile(model_src):
        model_src = os.path.join("models", model_src)

    try:
        model = DilatedNet.load_from_checkpoint(checkpoint_path=model_src, 
                                           #target_mask_supplied=True,
                                            loss="focal"
        )
    except:
        model = DilatedNet.load_from_checkpoint(
            checkpoint_path=model_src, 
            #target_mask_supplied=True,
            loss="focal"
        )

    #trainloader, valloader, testloader = get_dataloaders_DRIVE(batch_size=8, data_path="data/DRIVE/training")
    trainloader, valloader, testloader = get_dataloaders_PH2(batch_size=8, data_path="data/PH2_Dataset_images")


    if torch.cuda.is_available():
        trainer = pl.Trainer(
            default_root_dir="",
            accelerator="gpu",
            devices=[0],
            #strategy="ddp",
        )
    else:
        trainer = pl.Trainer(default_root_dir="")
    
    #results = trainer.test(model=model, dataloaders=trainloader, verbose=True)

    #print(results)

    results = trainer.test(model=model, dataloaders=valloader, verbose=True)

    print(results)

    results = trainer.test(model=model, dataloaders=testloader, verbose=True)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="UNet_Drive/best.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    eval(args.path)
