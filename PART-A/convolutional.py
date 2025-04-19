import torch
from torch import nn, optim
import lightning.pytorch as L


class convNet(L.LightningModule):
    def __init__(
        self,
        img_size,
        activation,          # Activation function name: relu, gelu, mish, silu
        num_filters,         # Number of filters in the first conv layer
        filter_size,          # Kernel size for all conv layers
        filter_org,         # Multiplicative factor for filter count per laye
        stride,
        padding,
        dense_neurons,      # Number of neurons in the dense (fc1) layer
        learning_rate,
        optimizer,
        dropout,
        usedropout,
        batchnorm,
    ):
        
        super().__init__()
        # self.img_size = img_size
        self.activation = activation
        self.num_filters = num_filters
        self.kernel_size = filter_size
        self.stride = stride
        self.padding = padding
        self.dense_neurons = dense_neurons
        self.lr = learning_rate
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.usedropout = usedropout
        self.batchnorm = batchnorm

        # Define convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3,  out_channels=self.num_filters,
            kernel_size=self.kernel_size,  stride=self.stride,
            padding=self.padding,
        )
        self.bn1 = nn.BatchNorm2d(num_features=self.num_filters)

        self.conv2 = nn.Conv2d(
            in_channels=self.num_filters, out_channels=self.num_filters * filter_org,
            kernel_size=self.kernel_size,   stride=self.stride,
            padding=self.padding,
        )
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2.out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels, out_channels=self.conv2.out_channels * filter_org,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding,
        )
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3.out_channels)

        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels, out_channels=self.num_filters * filter_org,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding,
        )
        self.bn4 = nn.BatchNorm2d(num_features=self.conv4.out_channels)

        self.conv5 = nn.Conv2d(
            in_channels=self.conv4.out_channels, out_channels=self.num_filters * filter_org,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding,
        )
        self.bn5 = nn.BatchNorm2d(num_features=self.conv5.out_channels)

        # Define activation function based on user input
        if self.activation.lower() == "relu":
            self.activation_layer = nn.ReLU()
        elif self.activation.lower() == "gelu":
            self.activation_layer = nn.GELU()
        elif self.activation.lower() == "silu":
            self.activation_layer = nn.SiLU()
        elif self.activation.lower() == "mish":
            self.activation_layer = nn.Mish()
        else:
            raise ValueError(
                "Invalid activation function. Choose from 'relu', 'gelu', 'mish' or 'silu'"
            )

        # Define max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.flatten = nn.Flatten()

        c1s = int((((img_size - self.kernel_size + (2 * self.padding)) / self.stride) + 1) / 2 )

        c2s = int((((c1s - self.kernel_size + (2 * self.padding)) / self.stride) + 1) / 2)

        c3s = int((((c2s - self.kernel_size + (2 * self.padding)) / self.stride) + 1) / 2)

        c4s = int((((c3s - self.kernel_size + (2 * self.padding)) / self.stride) + 1) / 2)

        c5s = int(
            (((c4s - self.kernel_size + (2 * self.padding)) / self.stride) + 1) / 2
        )

        # print(f"c1s_{c1s}_c2s_{c2s}_c3s_{c3s}_c4s_{c4s}_c5s_{c5s}_out_{self.conv5.out_channels}")
        flatten_neurons = int(c5s * c5s * self.conv5.out_channels)
        # print(f"multi_{c5s*c5s*self.conv5.out_channels}_flat_{flatten_neurons}")
        # Define fully connected layers
        self.fc1 = nn.Linear(flatten_neurons, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, 10)
        self.softmax_layer = nn.Softmax()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        if self.batchnorm == "Y":
            x = self.bn1(x)
        x = self.activation_layer(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.batchnorm == "Y":
            x = self.bn2(x)
        x = self.activation_layer(x)
        x = self.pool(x)

        if self.usedropout == "Y":
            x = self.dropout_layer(x)

        x = self.conv3(x)
        if self.batchnorm == "Y":
            x = self.bn3(x)
        x = self.activation_layer(x)
        x = self.pool(x)

        if self.usedropout == "Y":
            x = self.dropout_layer(x)

        x = self.conv4(x)
        if self.batchnorm == "Y":
            x = self.bn4(x)
        x = self.activation_layer(x)
        x = self.pool(x)

        if self.usedropout == "Y":
            x = self.dropout_layer(x)

        x = self.conv5(x)
        if self.batchnorm == "Y":
            x = self.bn5(x)
        x = self.activation_layer(x)
        x = self.pool(x)

        # Flatten
        x = self.flatten(x)

        if self.usedropout == "Y":
            x = self.dropout_layer(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.activation_layer(x)
        x = self.fc2(x)
        # x = self.softmax_layer(x)

        return x

    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":

            return optim.Adam(self.parameters(), lr=self.lr)

        elif self.optimizer.lower() == "sgd":

            return optim.SGD(self.parameters(), lr=self.lr)

        elif self.optimizer.lower() == "nadam":

            return optim.NAdam(self.parameters(), lr=self.lr)

        elif self.optimizer.lower() == "rmsprop":

            return optim.RMSprop(self.parameters(), lr=self.lr)

    
    def training_step(self, batch, batch_idx):
        loss, scores, y, accuracy = self._common_step(batch, batch_idx)

        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, _, _, accuracy = self._common_step(batch, batch_idx)
        self.log_dict(
            {"validation_loss": loss, "validation_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y, accuracy = self._common_step(batch, batch_idx)

        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch

        accuracy = 0

        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        y_pred = torch.argmax(torch.softmax(scores, dim=1), dim=1)
        accuracy += (y_pred == y).sum().item()
        accuracy = accuracy / len(scores)
        return loss, scores, y, accuracy

    def predict_step(self, batch, batch_idx):
        x, _ = batch

        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    