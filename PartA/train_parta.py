import torch
import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.fabric.utilities.seed import seed_everything
from pathlib import Path

from convolutional import convNet
from dataset import iNaturalistDataModule
from wandbConfigs import wandbconfig

device = "gpu" if torch.cuda.is_available() else "cpu"

seed_everything(42)


def train(wandb_configs, args):

    # Initialise wandb
    run_name = wandbconfig.run_name(wandb_configs)
    wandb.init(
        config=wandb_configs,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
    )
    wandb_logger = WandbLogger(project=args.wandb_project)
    wandb_configs = wandb.config

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

    model_path = Path(args.bestpath)
    model_name = "Best_Model.pth"
    model_save = model_path / model_name

    if args.load == "Y":

        model.load_state_dict(torch.load(f=model_save))

    data = iNaturalistDataModule(
        data_dir=Path(args.path),
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

    if args.run_test == "Y":

        trainer.test(model, data)

    wandb.finish()

    print(f"Saving model to: {model_save}")
    torch.save(obj=model.state_dict(), f=model_save)


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description="train_parta.py")

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
        "-i", "--img_size", default=128, help="Height and Width of the resized image"
    )
    parser.add_argument(
        "-nf", "--num_filters", default=64, help="Number of filters in each layer"
    )
    parser.add_argument(
        "-fz", "--filter_size", default=3, help="Size of filter in each layer"
    )
    parser.add_argument(
        "-fo",
        "--filter_org",
        default=2,
        help="Multiplier used for filter in each layer",
    )
    parser.add_argument("-d", "--dataug", default="N", help="Data Augmentation: Y or N")
    parser.add_argument("-bn", "--batchnorm", default="Y", help="Batch Norm: Y or N")
    parser.add_argument(
        "-b", "--batch_size", default=32, help="Batch size used to train neural network"
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        default="adam",
        help="Optmizer choices: [sgd, rmsprop, adam, nadam]",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        help="Learning rate used to optimize model parameters",
    )
    parser.add_argument("-t", "--run_test", default="Y", help="Run test data Y or N")
    parser.add_argument(
        "-n", "--dense_neurons", default=64, help="Number of neurons in the dense layer"
    )
    parser.add_argument(
        "-a",
        "--activation",
        default="Mish",
        help="Activation function choices: [ReLU, GELU, SiLU, Mish]",
    )
    parser.add_argument("-s", "--stride", default=1, help="Stride to use")
    parser.add_argument("-pa", "--padding", default=1, help="Padding to use")
    parser.add_argument("-dr", "--dropout", default=0.1, help="Value of dropout")
    parser.add_argument("-ud", "--usedropout", default="N", help="Drop Out: Y or N")
    parser.add_argument(
        "-ld", "--load", default="N", help="Load Best_Model.pth: Y or N"
    )
    parser.add_argument(
        "-bp",
        "--bestpath",
        default="./",
        help="Path of Best_Model.pth",
    )

    # Parse the input arguments
    args = parser.parse_args()

    wandb_configs = wandbconfig.set_configs(args)

    train(wandb_configs, args)
