import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from src.modules.lifter_2d_3d.model.linear_model.linear_model import BaselineModel
import numpy as np
from src.modules.lifter_2d_3d.utils.evaluation import Evaluator


class LitSimpleBaselineLinear(pl.LightningModule):
    def __init__(
        self, exclude_ankle=False, learning_rate=1e-3, exclude_hip=False, all_activities=[]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = BaselineModel(exclude_ankle=exclude_ankle, exclude_hip=exclude_hip)
        self.learning_rate = learning_rate
        self.val_loss_log = []
        self.val_print_count = 0
        self.train_loss_log = []
        self.test_loss_log = []
        self.evaluator = Evaluator(all_activities=all_activities)

    def forward(self, x, batch_idx):
        # use forward for inference/predictions
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        img_id, arr_id, x, y, valid, activities = batch
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).float().to(self.device)
        y_hat = self.model(x)
        # print(y_hat[valid].shape, y[valid].shape)
        loss = F.mse_loss(
            y_hat.reshape(y_hat.shape[0], -1, 3)[valid],
            y.reshape(y.shape[0], -1, 3)[valid],
        )
        self.train_loss_log.append(torch.sqrt(loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        img_id, arr_id, x, y, valid, activities = batch
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).float().to(self.device)
        y_hat = self.model(x)
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            activities
        )

    def test_step(self, batch, batch_idx):
        img_id, arr_id, x, y, valid, activities = batch
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).to(self.device)
        y_hat = self.model(x)
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            activities
        )

    def on_validation_epoch_end(self):
        print(f"check #{self.val_print_count}")
        if len(self.train_loss_log) > 0:
            print(
                f"training loss from {len(self.train_loss_log)} batches: {np.mean(self.train_loss_log) * 1000}"
            )
        pjpe, mpjpe, activities_mpjpe = self.evaluator.get_result()
        print(f"val MPJPE from: {len(self.val_loss_log)} batches : {mpjpe}")
        self.log("val_loss", mpjpe)
        self.train_loss_log = []
        self.val_print_count += 1
        self.evaluator.reset()

    def on_test_epoch_end(self):
        pjpe, mpjpe, activities_mpjpe = self.evaluator.get_result()
        print('MPJPE:', mpjpe)
        print(f'PJPE\n{pjpe}')
        print(f'activities_mpjpe:\n{activities_mpjpe}')
        self.log("mpjpe", mpjpe)
        print(f"test mpjpe: {mpjpe}")

    # def configure_optimizers(self):
    #     # self.hparams available because we called self.save_hyperparameters()
    #     # return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
        # learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            },
        }
