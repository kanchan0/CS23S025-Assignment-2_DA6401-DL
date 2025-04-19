# DA6401-Assignment-2 (CS23S025)
Convolution Neural Networks. The project is divided in two parts. PART-A is to train a CNN model from scratch and PART-B deals with a pre-trained model just as one would in real world application.

### Part A

- **train_parta.py:** This file contains the training script to train the model. We can pass various hyperparameters as input to this file as needed.
- **train_sweep.py**: This training script is dedicated to run and log sweeps in WandB
- **convolutional.py:** This file creates the convolutional neural network. According to conventions in PyTorch Lightning it also contains the training step, validation step, test step and predict step.
- **Best_Model.pth:** This file contains the parameters of the best model obtained during the sweep. Load this model to perform evaluation.

### Part B
- **train_partb.py:** This file contains the training script to train the model. We can pass various hyperparameters as input to this file as needed.
- **efficientNet_v2.py**: This file downloads the efficientNet_V2_S model for training. The freezing of all layers is also done here. This file is analogous to **convolutional.py** in Part A folder.


### Common files
- The following files are common to both Part A and Part B folders.
- **dataset.py:** Loads the dataset

### Instructions to Run
- Install the following libraries if not present in your system using the command `pip install library_name`
	- PyTorch (library_name is `torch`)
	- PyTorch Lightning(library_name is `lightning`)
	- wandb
- After downloading the repository navigate to the folder which we want to run. Then the training process can be started by executing the following command in the terminal:
	- `python train_parta.py` - For Part A
	- `python train_partb.py` - For Part B
- The file will be run with the best parameters.
- The file **train_parta.py** takes a number of parameters enabling us to configure the convolutional neural network based on our requirements. The below table lists the possible configurations.

|           Name           |  Default Value   | Description                                                               |
| :----------------------: | :--------------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | Convolutional NN | Project name used to track experiments in Weights & Biases dashboard      |
| `-we`, `--wandb_entity`  |     cs23s011     | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|      `-p`, `--path`      |       path       | Path of the dataset                                                       |
|     `-e`, `--epochs`     |        20        | Number of epochs to train neural network.                                 |
|    `-i`, `--img_size`    |       256        | Height and Width of the resized image                                     |
|  `-nf`, `--num_filters`  |        64        | Number of filters in each layer                                           |
|  `-fz`, `--filter_size`  |        3         | Size of filter in each layer                                              |
|  `-fo`, `--filter_org`   |        2         | Multiplier used for filter in each layer                                  |
|     `-d`, `--dataug`     |        N         | Data Augmentation: Y or N                                                 |
|   `-bn`, `--batchnorm`   |        Y         | Batch Norm: Y or N                                                        |
|   `-b`, `--batch_size`   |        64        | Batch size used to train neural network                                   |
|   `-o`, `--optimizer`    |       adam       | Optmizer choices: [sgd, rmsprop, adam, nadam]                             |
| `-lr`, `--learning_rate` |      0.001       | Learning rate used to optimize model parameters                           |
|    `-t`, `--run_test`    |        Y         | Run test data Y or N                                                      |
| `-n`, `--dense_neurons`  |        64        | Number of neurons in the dense layer                                      |
|     `-s`, `--stride`     |        1         | Stride to use                                                             |
|    `-pa`, `--padding`    |        1         | Padding to use                                                            |
|   `-a`, `--activation`   |       GeLU       | choices:  ["identity", "sigmoid", "tanh", "ReLU"]                         |
|    `-dr`, `--dropout`    |       0.1        | Value of dropout                                                          |
|   `-ud`,`--usedropout`   |        N         | Drop Out: Y or N                                                          |
|     `-ld`, `--load`      |        N         | Load Best_Model.pth: Y or N                                               |
|   `-bp`, `--bestpath`    |    best path     | Path of Best_Model.pth                                                    |
- Suppose we want to execute train.py with 'ReLU' activation function we can choose any of the below two commands
	- `python train.py -a ReLU`
	- `python train.py --activation ReLU`
- The default values are set to the parameter values which give the best validation accuracy

- Similarly **train_partb.py** takes the following inputs:

|           Name           |  Default Value   | Description                                                               |
| :----------------------: | :--------------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | CS23S025-Assignment-2-DL | Project name used to track experiments in Weights & Biases dashboard      |
| `-we`, `--wandb_entity`  |   cs23s025-indian-institute-of-technology-madras      | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|      `-p`, `--path`      |       path       | Path of the dataset                                                       |
|     `-e`, `--epochs`     |        10        | Number of epochs to train neural network.                                 |
|    `-i`, `--img_size`    |       224        | Height and Width of the resized image                                     |
|   `-b`, `--batch_size`   |        32        | Batch size used to train neural network                                   |
| `-lr`,`--learning_rate`  |      0.001       | Learning rate used to optimize model parameters                           |
|    `-dr`, `--dropout`    |       0.1        | Value of dropout                                                          |

#### Evaluating the model:

- The Best_Model.pth file is present in Part A folder.
- Run `train_parta.py` with the following command: 
	- `python train_parta.py -t Y -ld Y -bp path_of_best_model_here`