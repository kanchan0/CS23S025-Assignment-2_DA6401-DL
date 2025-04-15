import torch
import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.fabric.utilities.seed import seed_everything
from pathlib import Path

from efficientNet_v2 import efficientNet
from dataset import iNaturalistDataModule
from wandbConfigs import wandbconfig

device = "gpu" if torch.cuda.is_available() else "cpu"


seed_everything(42)


def train(wandb_configs, args):

    # Initialise wandb
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
