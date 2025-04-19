import torch
import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.fabric.utilities.seed import seed_everything
from pathlib import Path

from efficientNet_v2 import efficientNet
from dataset import iNaturalistDataModule



device = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(42)

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


def train(wandb_configs, args):

    # Initialise wandb, here i am using freeze all except the last classification layer
    run_name = "EfficientNetV2_Freeze_All"
    wandb.init(
        config=wandb_configs,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
    )
    wandb_logger = WandbLogger(project=args.wandb_project)
    wandb_configs = wandb.config

    model = efficientNet(0.2, 0.001)

    data = iNaturalistDataModule(
        data_dir=Path(args.path),
        batch_size=wandb_configs.batch_size,
        num_workers=2,
        img_size=wandb_configs.img_size,
        data_augmentation="N",
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

    # Create an argument parser
    parser = argparse.ArgumentParser(description="train_partb.py")

    # Tunable parameters as external arguments
    parser.add_argument(
        "-wp",
        "--wandb_project",
        default="CS23S025-Assignment-2-DL",
        help="Project name used to track experiments in Weights & Biases dashboard",
    )
    parser.add_argument(
        "-we",
        "--wandb_entity",
        default="cs23s025-indian-institute-of-technology-madras",
        help="Wandb Entity used to track experiments in the Weights & Biases dashboard",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="/home/kanchan/Downloads/nature_12K",
        help="Path of the dataset",
    )
    parser.add_argument(
        "-e", "--epochs", default=10, help="Number of epochs to train neural network"
    )
    parser.add_argument(
        "-i", "--img_size", default=224, help="Height and Width of the resized image"
    )
    parser.add_argument(
        "-b", "--batch_size", default=32, help="Batch size used to train neural network"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        help="Learning rate used to optimize model parameters",
    )
    parser.add_argument("-dr", "--dropout", default=0.1, help="Value of dropout")

    # Parse the input arguments
    args = parser.parse_args()

    wandb_configs = wandbconfig.tuner_set_configs(args)

    train(wandb_configs, args)
