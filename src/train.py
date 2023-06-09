import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from src.load_data import get_dataloaders
from src.model import Model
from src.model import DilatedNet


def train(config=None, checkpoint_callbacks=None):
    with wandb.init(config=config, 
                    project="project1_02514",
                    entity="chrillebon",):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        num_blocks = int(wandb.config.num_blocks)
        num_features = int(wandb.config.num_features)
        lr = wandb.config.lr
        weight_decay = wandb.config.weight_decay
        epochs = wandb.config.epochs
        batch_size = wandb.config.batch_size
        batch_normalization = wandb.config.batch_normalization
        optimizer = wandb.config.optimizer

        device = 0
        """
        model = Model(
            num_blocks=num_blocks,
            num_features=num_features,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            batch_normalization=batch_normalization,
            optimizer=optimizer,
        )
        """
        model = DilatedNet(
            num_features=num_features,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            optimizer=optimizer,
        )

        wandb.watch(model, log_freq=1)
        logger = pl.loggers.WandbLogger(project="project1_02514", entity="chrillebon")

        trainloader, valloader, _ = get_dataloaders(batch_size=batch_size)

        # make sure no models are saved if no checkpoints are given
        if checkpoint_callbacks is None:
            checkpoint_callbacks = [
                ModelCheckpoint(monitor=False, save_last=False, save_top_k=0)
            ]

        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir="",
            callbacks=checkpoint_callbacks,
            accelerator="gpu",
            devices=[device],
            strategy="ddp",
            logger=logger,
        )

        trainer.fit(
            model=model,
            train_dataloaders=trainloader,
            val_dataloaders=valloader,
        )

        print("Done!")


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(dirpath="models/Name_of_model", filename="best")
    train(
        config="src/config/default_params.yaml",
        checkpoint_callbacks=[checkpoint_callback],
    )
