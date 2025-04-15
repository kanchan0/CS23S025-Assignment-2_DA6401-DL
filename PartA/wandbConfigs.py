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
