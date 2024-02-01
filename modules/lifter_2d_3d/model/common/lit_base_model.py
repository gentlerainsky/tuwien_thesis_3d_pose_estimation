import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import numpy as np
from modules.lifter_2d_3d.utils.evaluation import Evaluator
from modules.lifter_2d_3d.model.semgcn.network.utils.graph_utils import connections


class LitBaseModel(pl.LightningModule):
    def __init__(
        self,
        exclude_ankle=False,
        learning_rate=1e-3,
        exclude_knee=False,
        all_activities=[],
        is_silence=False,
        joint_weights=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self.learning_rate = learning_rate
        self.train_loss_log = []
        self.train_history = []
        self.val_history = []
        self.test_history = []
        self.val_print_count = 0
        self.evaluator = Evaluator(all_activities=all_activities)
        self.is_silence = is_silence
        self.exclude_ankle = exclude_ankle
        self.exclude_knee = exclude_knee
        self.connections = connections
        self.joint_weights = joint_weights

    def set_all_activities(self, all_activities):
        self.evaluator = Evaluator(all_activities=all_activities)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def preprocess_x(self, x):
        raise NotImplementedError()

    def preprocess_input(self, x, y, valid, activity):
        raise NotImplementedError()

    def preprocess_batch(self, batch):
        x = batch['keypoints_2d']
        y = batch['keypoints_3d']
        valid = None
        activity = None
        if 'valid' in batch:
            valid = batch['valid']
        if 'activity' in batch:
            activity = batch['activity']
        return x, y, valid, activity

    def training_step(self, batch, batch_idx):
        x, y, valid, activity = self.preprocess_input(*self.preprocess_batch(batch))
        y_hat = self.forward(x)
        loss = F.mse_loss(
            y_hat.reshape(y_hat.shape[0], -1, 3),
            y.reshape(y.shape[0], -1, 3),
            reduction='none',
        )
        # mask out invalid batch
        loss = loss.sum(axis=2) * (valid).float()
        # Mean square error
        loss = loss.mean()
        self.train_loss_log.append(torch.sqrt(loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, valid, activities = self.preprocess_input(*self.preprocess_batch(batch))
        y_hat = self.forward(x)
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
            activities
        )

    def test_step(self, batch, batch_idx):
        x, y, valid, activities = self.preprocess_input(*self.preprocess_batch(batch))
        y_hat = self.forward(x)
        self.evaluator.add_result(
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
            activities
        )

    def on_validation_epoch_end(self):
        if not self.is_silence:
            print(f'check #{self.val_print_count}')
            if len(self.train_loss_log) > 0:
                print(
                    f'training loss from {len(self.train_loss_log)} batches: {np.mean(self.train_loss_log) * 1000}'
                )
        pjpe, mpjpe, activities_mpjpe, activity_macro_mpjpe = self.evaluator.get_result()
        if not self.is_silence:
            print(f'val MPJPE from: {len(self.evaluator.mpjpe)} samples : {mpjpe}')
            if activity_macro_mpjpe is not None:
                print('activity_macro_mpjpe', activity_macro_mpjpe)
        self.log('mpjpe', mpjpe)
        if activity_macro_mpjpe is not None:
            self.log('activity_macro_mpjpe', activity_macro_mpjpe)
        if len(self.train_loss_log) > 0:
            self.train_history.append(np.mean(self.train_loss_log) * 1000)
            self.train_loss_log = []
        self.val_print_count += 1
        self.evaluator.reset()
        self.val_history.append({
            'pjpe': pjpe,
            'mpjpe': mpjpe,
            'activities_mpjpe': activities_mpjpe,
            'activity_macro_mpjpe': activity_macro_mpjpe
        })

    def on_test_epoch_end(self):
        pjpe, mpjpe, activities_mpjpe, activity_macro_mpjpe = self.evaluator.get_result()
        self.log('mpjpe', mpjpe)
        if activity_macro_mpjpe is not None:
            self.log('activity_macro_mpjpe', activity_macro_mpjpe)
            if not self.is_silence:
                print('activity_macro_mpjpe', activity_macro_mpjpe)
        self.test_history.append({
            'pjpe': pjpe,
            'mpjpe': mpjpe,
            'activities_mpjpe': activities_mpjpe,
            'activity_macro_mpjpe': activity_macro_mpjpe
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
        # learning rate scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': sch,
                'monitor': 'train_loss',
            },
        }
