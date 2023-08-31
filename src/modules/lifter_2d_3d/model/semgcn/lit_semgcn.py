import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from src.modules.lifter_2d_3d.model.semgcn.sem_gcn import SemGCN
import numpy as np
from src.modules.lifter_2d_3d.model.semgcn.utils.skeleton import Skeleton
from src.modules.lifter_2d_3d.model.semgcn.utils.graph_utils import adj_mx_from_edges

connections = [
    (0, 1, 'nose_left_eye'), # nose & left_eye
    (0, 2, 'nose_right_eye'), # nose & right_eye
    (1, 2, 'left_right_eye'), # left & right eyes
    (1, 3, 'left_eye_left_ear'), # left eye & ear
    (2, 4, 'right_eye_right_ear'), # right eye & ear
    (0, 5, 'nose_left_shoulder'), # nose & left shoulder
    (0, 6, 'nose_right_shoulder'), # nose & right shoulder
    (3, 5, 'left_ear_shoulder'), # left ear & shoulder
    (4, 6, 'right_ear_shoulder'), # right ear & shoulder
    (5, 6, 'left_shoulder_right_sholder'), # left & right shoulder
    (5, 7, 'left_sholder_left_elbow'), # left shoulder & elbow
    (5, 11, 'left_shoulder_left_hip'), # left shoulder & hip
    (6, 8, 'right_shoulder_right_elbow'), # right shoulder & elbow
    (6, 12, 'right_shoulder_right_hip'), # right shoulder & hip
    (7, 9, 'left_elbow_left_wrist'), # left elbow & wrist
    (8, 10, 'right_elbow_right_wrist'), # right elbow & wrist
    (11, 12, 'left_hip_right_hip'), # left & right hip
    (11, 13, 'left_hip_left_knee'), # left hip & knee
    (12, 14, 'right_hip_right_knee'), # right hip & knee
    (13, 15, 'left_knee_left_ankle'), # left knee & ankle
    (14, 16, 'right_knee_right_ankle') # right knee & ankle
]

parents = [
    -1,
    0,
    0,
    1,
    2,
    0,
    0,
    5,
    6,
    7,
    8,
    5,
    6,
    11,
    12,
    # 13,
    # 14
]

node_groups = [
    [1, 3], # left-right ear
    [2, 4], # left-right eye
    [5, 6], # left-right sholder
    [7, 9], # left elbow-wrist
    [8, 10], # right elbow-wrist
    [11, 12], # left-right hip
]

connections = np.array(connections)[:,:2].astype(int).tolist()


class LitSemGCN(pl.LightningModule):
    def __init__(self, exclude_ankle=False, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        num_pts = 17
        _connections=connections
        if exclude_ankle:
            num_pts = 15
            _connections = connections[:-2]
        adj = adj_mx_from_edges(num_pts=num_pts, edges=_connections).to_dense()

        self.model = SemGCN(
            adj=adj,
            hid_dim=128,
            coords_dim=(2, 3),
            num_layers=4,
            nodes_group=None,
            p_dropout=None,
        )
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
        x = x.float().squeeze(2).to(self.device)
        y = y.float().squeeze(2).to(self.device)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.train_loss_log.append(torch.sqrt(loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float().squeeze(2).to(self.device)
        y = y.float().squeeze(2).to(self.device)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.val_loss_log.append(torch.sqrt(loss).item())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float().squeeze(2).to(self.device)
        y = y.float().squeeze(2).to(self.device)
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
        print(f't loss from {len(self.test_loss_log)} batches: {np.mean(self.test_loss_log) * 1000}')

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
