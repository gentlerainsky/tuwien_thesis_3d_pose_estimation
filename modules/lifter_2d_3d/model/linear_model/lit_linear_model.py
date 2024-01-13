import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import numpy as np
from modules.lifter_2d_3d.model.linear_model.network.linear_model import BaselineModel
from modules.lifter_2d_3d.utils.evaluation import Evaluator


class LitSimpleBaselineLinear(pl.LightningModule):
    def __init__(
        self,
        exclude_ankle=False,
        learning_rate=1e-3,
        exclude_knee=False,
        all_activities=[],
        is_silence=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = BaselineModel(
            exclude_ankle=exclude_ankle, exclude_knee=exclude_knee
        )
        self.learning_rate = learning_rate
        self.val_loss_log = []
        self.val_history = []
        self.test_history = []
        self.val_print_count = 0
        self.train_loss_log = []
        self.test_loss_log = []
        self.evaluator = Evaluator(all_activities=all_activities)
        self.is_silence = is_silence

    def forward(self, x):
        # use forward for inference/predictions
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x = batch["keypoints_2d"]
        y = batch["keypoints_3d"]
        valid = batch["valid"]
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).float().to(self.device)
        y_hat = self.model(x)
        # y_hat.shape => batch x flatten(keypoinys_3d)
        # TODO: Fix the calculation to consider only "valid"
        loss = F.mse_loss(
            y_hat.reshape(y_hat.shape[0], -1, 3),
            y.reshape(y.shape[0], -1, 3),
            reduction="none",
        )
        # mask out invalid batch
        loss = loss.sum(axis=2) * (valid).float()
        # Mean square error
        loss = loss.mean()
        self.train_loss_log.append(torch.sqrt(loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["keypoints_2d"]
        y = batch["keypoints_3d"]
        valid = batch["valid"]
        activities = None
        if "activities" in batch:
            activities = batch["activities"]
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).float().to(self.device)
        valid = torch.flatten(valid, start_dim=1)
        y_hat = self.model(x)
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
            activities
        )

    def test_step(self, batch, batch_idx):
        x = batch["keypoints_2d"]
        y = batch["keypoints_3d"]
        valid = batch["valid"]
        activities = None
        if "activities" in batch:
            activities = batch["activities"]
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).to(self.device)
        valid = torch.flatten(valid, start_dim=1)
        y_hat = self.model(x)
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
            activities
        )

    def on_validation_epoch_end(self):
        # if not self.is_silence:
        print(f"check #{self.val_print_count}")
        if len(self.train_loss_log) > 0:
            print(
                f"training loss from {len(self.train_loss_log)} batches: {np.mean(self.train_loss_log) * 1000}"
            )
        pjpe, mpjpe, activities_mpjpe = self.evaluator.get_result()
        # if not self.is_silence:
        print(f"val MPJPE from: {len(self.evaluator.mpjpe)} samples : {mpjpe}")
        self.log("val_loss", mpjpe)
        self.train_loss_log = []
        self.val_print_count += 1
        self.evaluator.reset()
        self.val_history.append(
            {"pjpe": pjpe, "mpjpe": mpjpe, "activities_mpjpe": activities_mpjpe}
        )

    def on_test_epoch_end(self):
        pjpe, mpjpe, activities_mpjpe = self.evaluator.get_result()
        if not self.is_silence:
            print("MPJPE:", mpjpe)
            print(f"PJPE\n{pjpe}")
            print(f"activities_mpjpe:\n{activities_mpjpe}")
            print(f"test mpjpe: {mpjpe}")
        self.log("mpjpe", mpjpe)
        self.test_history.append(
            {"pjpe": pjpe, "mpjpe": mpjpe, "activities_mpjpe": activities_mpjpe}
        )

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
