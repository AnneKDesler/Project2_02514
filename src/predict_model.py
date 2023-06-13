import argparse
import pytorch_lightning as pl
import torch
import os
from src.load_data import get_dataloaders_DRIVE
from src.load_data import get_dataloaders_PH2
from src.model import Model, DilatedNet
import matplotlib.pyplot as plt

def predict(model_src):
    if not os.path.isfile(model_src):
        model_src = os.path.join("models", model_src)
    print(model_src)
    try:
        model = Model.load_from_checkpoint(checkpoint_path=model_src, 
                                           target_mask_supplied=True,
                                            #loss="focal"
        )
    except:
        model = Model.load_from_checkpoint(
            checkpoint_path=model_src, 
            target_mask_supplied=True,
            #loss="focal"
        )
    model.to("cuda")

    trainloader, valloader, testloader = get_dataloaders_DRIVE(batch_size=1, data_path="data/DRIVE/training")
    #_, _, testloader = get_dataloaders_PH2(batch_size=1, data_path="data/PH2_Dataset_images")

    # visualize all segmented images for test dataset and save to file

    output = model(list(testloader)[0][0].to("cuda"))

    output = torch.sigmoid(output)

    output = output.detach().cpu().numpy()

    output = output.squeeze()

    output = output > 0.5

    output = output.astype(int)

    

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(output, cmap="gray")
    plt.axis("off")
    fig.savefig("33.png", bbox_inches="tight", pad_inches=0)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="UNet_Drive/best.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    predict(args.path)
