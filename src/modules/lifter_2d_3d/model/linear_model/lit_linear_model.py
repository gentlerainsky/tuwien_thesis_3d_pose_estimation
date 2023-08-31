import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from src.modules.lifter_2d_3d.model.linear_model.linear_model import BaselineModel
import numpy as np


class LitSimpleBaselineLinear(pl.LightningModule):
    def __init__(self, exclude_ankle=False, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = BaselineModel(exclude_ankle=exclude_ankle)
        self.learning_rate = learning_rate
        self.val_loss_log = []
        self.val_print_count = 0
        self.train_loss_log = []
        self.test_loss_log = []

    def forward(self, x, batch_idx):
        # use forward for inference/predictions        
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).float().to(self.device)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.train_loss_log.append(torch.sqrt(loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).float().to(self.device)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.val_loss_log.append(torch.sqrt(loss).item())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).to(self.device)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.test_loss_log.append(torch.sqrt(loss).item())
        return loss

    def on_validation_epoch_end(self):
        print(f'check #{self.val_print_count}')
        if len(self.train_loss_log) > 0:
            print(f'training loss from {len(self.train_loss_log)} batches: {np.mean(self.train_loss_log) * 1000}')
        val_loss = np.mean(self.val_loss_log)
        print(f"val loss from: {len(self.val_loss_log)} batches : {val_loss * 1000}")
        self.log("val_loss", val_loss.item())
        self.train_loss_log = []
        self.val_print_count += 1
        self.val_loss_log = []

    def on_test_epoch_end(self):
        test_loss = np.mean(self.test_loss_log)
        self.log("test_loss", test_loss)
        print(f'test loss from {len(self.test_loss_log)} batches: {test_loss * 1000}')

    # def configure_optimizers(self):
    #     # self.hparams available because we called self.save_hyperparameters()
    #     # return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 1, gamma = 0.96
        )
        #learning rate scheduler
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
            }
        }
