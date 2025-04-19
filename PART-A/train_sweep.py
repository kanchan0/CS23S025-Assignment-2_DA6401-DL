import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.fabric.utilities.seed import seed_everything
from pathlib import Path
from convolutional import convNet
from dataset import iNaturalistDataModule

class wandbconfig:

    def set_configs(args):

        wandb_configs = {
            "epochs": args.epochs,
            "img_size": args.img_size,
            "dataug": args.dataug,
            "batchnorm": args.batchnorm,
            "num_filters": args.num_filters,
            "filter_size": args.filter_size,
            "filter_org": args.filter_org,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "usedropout": args.usedropout,
            "optimizer": args.optimizer,
            "dense_neurons": args.dense_neurons,
            "activation": args.activation,
            "stride": args.stride,
            "padding": args.padding,
        }

        return wandb_configs

    def run_name(wandb_configs):

        run_name = "nf_{}_fsz_{}_fo_{}_a_{}_e_{}_b_{}_dn_{}_da_{}".format(
            wandb_configs["num_filters"],
            wandb_configs["filter_size"],
            wandb_configs["filter_org"],
            wandb_configs["activation"],
            wandb_configs["epochs"],
            wandb_configs["batch_size"],
            wandb_configs["dense_neurons"],
            wandb_configs["dataug"],
        )

        return run_name

    def sweep_name(wandb_configs):

        sweep_name = "nf_{}_fsz_{}_fo_{}_a_{}_e_{}_b_{}_dn_{}".format(
            wandb_configs.num_filters,
            wandb_configs.filter_size,
            wandb_configs.filter_org,
            wandb_configs.activation,
            wandb_configs.epochs,
            wandb_configs.batch_size,
            wandb_configs.dense_neurons,
        )

        return sweep_name

    def tuner_set_configs(args):

        wandb_configs = {
            "epochs": args.epochs,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
        }

        return wandb_configs


device = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(42)

sweep_configs = {
    "name": "Local Machine Larger Model Sweep",
    "metric": {"name": "validation_accuracy", "goal": "maximize"},
    "method": "grid",
    "early_terminate": {"type": "hyperband", "min_iter": 2, "eta": 2},
    "parameters": {
        "img_size": {"values": [64, 128,256]},
        "num_filters": {"values": [32,64,128]},
        "filter_size": {"values": [3, 5]},
        "filter_org": {"values": [2]},
        "activation": {"values": ["Mish","relu","gelu"]},
        "optimizer": {"values": ["adam","nadam","rmsprop"]},
        "epochs": {"values": [20]},
        "batch_size": {"values": [32,64,128]},
        "learning_rate": {"values": [0.001]},
        "dataug": {"values": ["N"]},
        "dense_neurons": {"values": [64]},
        "batchnorm": {"values": ["Y"]},
        "stride": {"values": [1]},
        "padding": {"values": [1, 2]},
        "dropout": {"values": [0.1,0.01]},
        "usedropout": {"values": ["Y"]},
    },
}

hyperparameter_defaults = dict()

def train():

    # Initialise wandb
    wandb.init(
        config=hyperparameter_defaults, project="CS23S025-Assignment-2-DL", 
        entity="cs23s025-indian-institute-of-technology-madras"
    )
    wandb_logger = WandbLogger(project="CS23S025-Assignment-2-DL")
    wandb_configs = wandb.config

    sweep_name = wandbconfig.sweep_name(wandb_configs)
    wandb.run.name = sweep_name

    model = convNet(
        img_size=wandb_configs.img_size,
        activation=wandb_configs.activation,
        num_filters=wandb_configs.num_filters,
        filter_size=wandb_configs.filter_size,
        filter_org=wandb_configs.filter_org,
        stride=wandb_configs.stride,
        padding=wandb_configs.padding,
        dense_neurons=wandb_configs.dense_neurons,
        learning_rate=wandb_configs.learning_rate,
        optimizer=wandb_configs.optimizer,
        dropout=wandb_configs.dropout,
        usedropout=wandb_configs.usedropout,
        batchnorm=wandb_configs.batchnorm,
    )

    data = iNaturalistDataModule(
        data_dir=Path(""),
        batch_size=wandb_configs.batch_size,
        num_workers=2,
        img_size=wandb_configs.img_size,
        data_augmentation=wandb_configs.dataug,
    )
    trainer = Trainer(
        accelerator=device,
        min_epochs=1,
        max_epochs=wandb_configs.epochs,
        logger=wandb_logger,
        log_every_n_steps=50,
    )
    trainer.fit(model, data)
    trainer.validate(model, data)
    # trainer.test(model, data)
    wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_configs,
                           entity="cs23s025-indian-institute-of-technology-madras",
                           project="CS23S025-Assignment-2-DL")
    wandb.agent(sweep_id, function=train)
