import torch
from torch import nn, optim
import lightning.pytorch as L
import torchvision.models

class efficientNet(L.LightningModule):

    def __init__(self, dropout, learning_rate):
        super().__init__()

        self.dropout = dropout
        self.learning_rate = learning_rate
        
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT #We obtain the best available weights for this model
        model = torchvision.models.efficientnet_v2_s(weights=weights) #Initializing the model

        #We use the approach of overwriting the classifier layer, so that output classes is 10.
        model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=10, #Here initially it was 1000
                        bias=True))

        #We are freezing all the features layers. This does not include freezing of the classifier layer.
        for param in model.features.parameters():
            param.requires_grad = False

        self.efNet_tuned = model

        self.loss_fn = nn.CrossEntropyLoss()

    
    def forward(self, x):
        x = self.efNet_tuned(x)
        return x


    def training_step(self, batch, batch_idx):
        loss, scores, y, accuracy = self._common_step(batch, batch_idx)
 
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y, accuracy = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "validation_loss": loss,
                "validation_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _, accuracy = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy
            },
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
        accuracy = accuracy/len(scores)
        return loss, scores, y, accuracy

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
         return optim.Adam(self.parameters(), lr=self.learning_rate)