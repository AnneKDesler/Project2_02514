import wandb
import yaml
from src.train import train


if __name__ == "__main__":
    with open("src/config/sweep_config.yaml", "r") as yamlfile:
        sweep_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="project1_02514",
        entity="chrillebon",
    )
    wandb.agent(sweep_id, train, count=20)
