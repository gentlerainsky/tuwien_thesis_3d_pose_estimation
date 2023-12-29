import torch
import numpy as np
import pytorch_lightning as pl
from torch.nn import functional as F
from src.modules.lifter_2d_3d.model.graph_mlp.graphmlp import Model as GraphMLP
# from src.modules.lifter_2d_3d.model.semgcn.utils.skeleton import Skeleton
# from src.modules.lifter_2d_3d.model.semgcn.utils.graph_utils import adj_mx_from_edges
from src.modules.lifter_2d_3d.utils.evaluation import Evaluator


class LitGraphMLP(pl.LightningModule):
    def __init__(
            self,
            exclude_ankle=False,
            learning_rate=1e-3,
            exclude_hip=False,
            all_activities=[]
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphMLP(
            frames=1,
            depth=3,
            d_hid=1024,
            token_dim=256,
            channel=512,
            n_joints=13
        )
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
        x = batch['keypoints_2d']
        y = batch['keypoints_3d']
        valid = batch['valid']
        x = x.float().unsqueeze(1).to(self.device)
        y = y.float().to(self.device)
        y_hat = self.model(x).squeeze()
        loss = F.mse_loss(y_hat, y, reduction='none')
        # mask out invalid batch
        loss = loss.sum(axis=2) * (valid).float()
        # Mean square error
        loss = loss.mean()
        self.train_loss_log.append(torch.sqrt(loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['keypoints_2d']
        y = batch['keypoints_3d']
        activities = None
        if 'activities' in batch:
            activities = batch['activities']
        x = x.float().unsqueeze(1).to(self.device)
        y = y.float().to(self.device)
        y_hat = self.model(x).squeeze()
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            activities
        )
        # loss = F.mse_loss(y_hat, y)
        # self.val_loss_log.append(torch.sqrt(loss).item())
        # return loss

    def test_step(self, batch, batch_idx):
        x = batch['keypoints_2d']
        y = batch['keypoints_3d']
        activities = None
        if 'activities' in batch:
            activities = batch['activities']

        x = x.float().unsqueeze(1).to(self.device)
        y = y.float().to(self.device)
        y_hat = self.model(x).squeeze()
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            activities
        )
        # loss = F.mse_loss(y_hat, y)
        # self.test_loss_log.append(torch.sqrt(loss).item())
        # return loss

    def on_validation_epoch_end(self):
        # print(f'check #{self.val_print_count}')
        # if len(self.train_loss_log) > 0:
        #     print(f'training loss from {len(self.train_loss_log)} batches: {np.mean(self.train_loss_log) * 1000}')
        # val_loss = np.mean(self.val_loss_log)
        # print(f"val loss from: {len(self.val_loss_log)} batches : {val_loss * 1000}")
        # self.log("val_loss", val_loss.item())
        # self.train_loss_log = []
        # self.val_print_count += 1
        # self.val_loss_log = []
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
        # test_loss = np.mean(self.test_loss_log)
        # self.log("test_loss", test_loss)
        # print(f't loss from {len(self.test_loss_log)} batches: {np.mean(self.test_loss_log) * 1000}')
        pjpe, mpjpe, activities_mpjpe = self.evaluator.get_result()
        print('MPJPE:', mpjpe)
        print(f'PJPE\n{pjpe}')
        print(f'activities_mpjpe:\n{activities_mpjpe}')
        self.log("mpjpe", mpjpe)
        print(f"test mpjpe: {mpjpe}")

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
