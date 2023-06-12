import wandb
import yaml
from src.train_UNet_PH2 import train_UNet


if __name__ == "__main__":
    with open("src/config/sweep_config_UNet_PH2.yaml", "r") as yamlfile:
        sweep_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="project2_02514",
        entity="chrillebon",
    )
    wandb.agent(sweep_id, train_UNet, count=20)
