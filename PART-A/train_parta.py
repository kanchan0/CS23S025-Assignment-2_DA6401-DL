import torch
import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.fabric.utilities.seed import seed_everything
from pathlib import Path
from convolutional import convNet
from dataset import iNaturalistDataModule

""""
WANDB configs
"""
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


def train(wandb_configs, args):
    # unique name for  W&B run based on sweep
    run_name = wandbconfig.run_name(wandb_configs)
    wandb.init(
        config=wandb_configs,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
    )

   #logger to PyTorch Lightning
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

    # Optionally load a pre-trained model if flag is set
    if args.load == "Y":
        model.load_state_dict(torch.load(model_save))

    # Prepare dataset using custom DataModule
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

    # training loop
    trainer.fit(model, datamodule=data)

    # validation after training
    trainer.validate(model, datamodule=data)

    # Optionally run testing phase if enabled
    if args.run_test == "Y":
        trainer.test(model, datamodule=data)
        
    wandb.finish()
    
    print(f"Saving model to: {model_save}")
    torch.save(model.state_dict(), model_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train_parta.py")

    # Define CLI arguments to pass in various training parameters
    parser.add_argument(
        "-wp", "--wandb_project", default="CS23S025-Assignment-2-DL",
        help="W&B project to log this experiment under"
    )
    parser.add_argument(
        "-we", "--wandb_entity", default="cs23s025-indian-institute-of-technology-madras",
        help="W&B entity or team name"
    )
    parser.add_argument(
        "-p", "--path", default="/home/kanchan/Downloads/nature_12K",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "-e", "--epochs", default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "-i", "--img_size", default=256,
        help="Image resolution (height x width)"
    )
    parser.add_argument(
        "-nf", "--num_filters", default=64,
        help="Base number of convolutional filters"
    )
    parser.add_argument(
        "-fz", "--filter_size", default=3,
        help="Kernel size for conv layers"
    )
    parser.add_argument(
        "-fo", "--filter_org", default=2,
        help="Multiplicative factor to scale filters in deeper layers"
    )
    parser.add_argument(
        "-d", "--dataug", default="N",
        help="Whether to apply data augmentation: Y or N"
    )
    parser.add_argument(
        "-bn", "--batchnorm", default="Y",
        help="Enable batch normalization: Y or N"
    )
    parser.add_argument(
        "-b", "--batch_size", default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "-o", "--optimizer", default="adam",
        help="Choose optimizer: sgd, rmsprop, adam, nadam"
    )
    parser.add_argument(
        "-lr", "--learning_rate", default=0.001,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "-t", "--run_test", default="Y",
        help="Test the model after training: Y or N"
    )
    parser.add_argument(
        "-n", "--dense_neurons", default=64,
        help="Number of units in the dense layer"
    )
    parser.add_argument(
        "-a", "--activation", default="GELU",
        help="Activation function: ReLU, GELU, SiLU, Mish"
    )
    parser.add_argument(
        "-s", "--stride", default=1,
        help="Stride value for convolutions"
    )
    parser.add_argument(
        "-pa", "--padding", default=1,
        help="Padding value for convolutions"
    )
    parser.add_argument(
        "-dr", "--dropout", default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "-ud", "--usedropout", default="N",
        help="Enable dropout: Y or N"
    )
    parser.add_argument(
        "-ld", "--load", default="N",
        help="Load pretrained weights: Y or N"
    )
    parser.add_argument(
        "-bp", "--bestpath", default="./",
        help="Path to save/load Best_Model.pth"
    )

    args = parser.parse_args()
    wandb_configs = wandbconfig.set_configs(args)
    train(wandb_configs, args)
