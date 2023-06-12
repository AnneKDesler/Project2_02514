import argparse
import pytorch_lightning as pl
import torch
import os
from src.load_data import get_dataloaders
from src.model import Model


def eval(model_src):
    if not os.path.isfile(model_src):
        model_src = os.path.join("models", model_src)

    try:
        model = Model.load_from_checkpoint(checkpoint_path=model_src)
    except:
        model = Model.load_from_checkpoint(
            checkpoint_path=model_src, batch_normalization=True
        )

    trainloader, valloader, testloader = get_dataloaders(batch_size=8)

    if torch.cuda.is_available():
        trainer = pl.Trainer(
            default_root_dir="",
            accelerator="gpu",
            devices=[0],
            #strategy="ddp",
        )
    else:
        trainer = pl.Trainer(default_root_dir="")
    
    results = trainer.test(model=model, dataloaders=trainloader, verbose=True)

    print(results)

    results = trainer.test(model=model, dataloaders=valloader, verbose=True)

    print(results)

    results = trainer.test(model=model, dataloaders=testloader, verbose=True)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="epoch=19-step=2560.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    eval(args.path)
