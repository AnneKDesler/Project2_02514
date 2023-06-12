import wandb
import yaml
from src.train_CNN import train


if __name__ == "__main__":
    with open("src/config/sweep_config_DilatedNet.yaml", "r") as yamlfile:
        sweep_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="project2_02514",
        entity="chrillebon",
    )
    wandb.agent(sweep_id, train, count=20)
