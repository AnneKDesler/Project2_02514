import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from src.load_data import get_dataloaders_PH2, get_dataloaders_DRIVE
from src.model import DilatedNet


def train(config=None, checkpoint_callbacks=None):
    with wandb.init(config=config, 
                    project="project2_02514",
                    entity="chrillebon",):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        lr = wandb.config.lr
        weight_decay = wandb.config.weight_decay
        epochs = wandb.config.epochs
        batch_size = wandb.config.batch_size
        optimizer = wandb.config.optimizer

        device = 0
        model = DilatedNet(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            optimizer=optimizer,
            target_mask_supplied=True,
            loss="focal"
        )

        wandb.watch(model, log_freq=1)
        logger = pl.loggers.WandbLogger(project="project2_02514", entity="chrillebon")

        #trainloader, valloader, _ = get_dataloaders_PH2(batch_size=batch_size, data_path="data/PH2_Dataset_images")
        trainloader, valloader,_ = get_dataloaders_DRIVE(batch_size=batch_size, data_path="data/DRIVE/training")
        
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
            logger=logger,
            log_every_n_steps=1,

        )

        trainer.fit(
            model=model,
            train_dataloaders=trainloader,
            val_dataloaders=valloader,
        )

        print("Done!")


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(dirpath="models/CNN_Drive_focal", filename="best")
    train(
        config="src/config/default_params_DilatedNet.yaml",
        checkpoint_callbacks=[checkpoint_callback],
    )
