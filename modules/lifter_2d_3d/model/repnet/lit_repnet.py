import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from modules.lifter_2d_3d.model.linear_model.network.linear_model import BaselineModel
from modules.lifter_2d_3d.model.repnet.network.discriminator import DiscriminatorModel
from modules.lifter_2d_3d.model.repnet.network.camera_net import CameraNet
from modules.lifter_2d_3d.model.repnet.network.repnet import RepNet
from modules.lifter_2d_3d.utils.evaluation import Evaluator
from modules.lifter_2d_3d.model.repnet.network.utils import (
    camera_loss,
    weighted_pose_2d_loss,
)
from collections import OrderedDict

import numpy as np
from torch.utils.data import DataLoader


class LitRepNet(pl.LightningModule):

    def __init__(
        self,
        lifter_2D_3D,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        batch_size: int = 64,
        all_activities=[],
        is_silence=False,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=["lifter_2D_3D"])
        self.automatic_optimization = False
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.is_silence = is_silence
        # networks
        self.lifter_2D_3D = lifter_2D_3D
        self.num_keypoints = 13
        self.input_dim = self.num_keypoints
        self.camera_net = CameraNet(input_dim=self.input_dim * 2)
        self.generator = RepNet(
            self.lifter_2D_3D, self.camera_net, input_dim=self.input_dim * 2
        )
        self.discriminator = DiscriminatorModel(input_size=self.input_dim * 3)
        self.evaluator = Evaluator(all_activities=all_activities)
        self.procrusted_evaluator = Evaluator(
            all_activities=all_activities, is_procrustes=True
        )
        self.val_loss_log = []
        self.val_print_count = 0
        self.total_g_loss_log = []
        self.g_loss_log = []
        self.pose_2d_loss_log = []
        self.c_loss_log = []
        self.d_loss_log = []
        self.test_loss_log = []
        self.val_history = []
        self.test_history = []

    def forward(self, x):
        y_hat = self.generator.lifter2D_3D(x)
        return y_hat

    def preprocess_x(self, x):
        # add batch dimension if there is none.
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 2)
        x = torch.flatten(x, start_dim=1).float()
        return x

    def preprocess_input(self, x, y, valid, activity):
        x = self.preprocess_x(x)
        y = torch.flatten(y, start_dim=1).float()
        valid = torch.flatten(valid, start_dim=1)
        return x, y, valid, activity

    def preprocess_batch(self, batch):
        x = batch["keypoints_2d"]
        y = batch["keypoints_3d"]
        valid = None
        activity = None
        if "valid" in batch:
            valid = batch["valid"]
        if "activity" in batch:
            activity = batch["activity"]
        return x, y, valid, activity

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(
            self.device
        )
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        [g_opt, d_opt] = self.optimizers()
        x, y, valid, activity = self.preprocess_input(*self.preprocess_batch(batch))
        # input_2d = torch.flatten(x, start_dim=1).float().to(self.device)
        # real_pose_3d = torch.flatten(y, start_dim=1).float().to(self.device)
        input_2d = x
        real_pose_3d = y
        lambda_gp = 10

        # train generator
        # if optimizer_idx == 0:
        for i in range(1):
            # generate images
            camera_out, gen_pose_3d, reprojected_2d = self.generator(input_2d)
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            # valid = torch.ones(input_2d.size(0), 1)
            # valid = valid.type_as(input_2d)

            g_loss = -torch.mean(self.discriminator(gen_pose_3d))
            c_loss = camera_loss(camera_out)
            pose_2d_loss = weighted_pose_2d_loss(
                input_2d, reprojected_2d, self.num_keypoints
            )
            total_g_loss = g_loss + c_loss + pose_2d_loss
            g_opt.zero_grad()
            self.manual_backward(total_g_loss)
            # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.)
            g_opt.step()
        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        for i in range(1):
            camera_out, gen_pose_3d, reprojected_2d = self.generator(input_2d)

            # Real images
            real_validity = self.discriminator(real_pose_3d)
            # Fake images
            fake_validity = self.discriminator(gen_pose_3d)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(
                real_pose_3d.data, gen_pose_3d.data
            )
            # Adversarial loss
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + lambda_gp * gradient_penalty
            )
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.)
            d_opt.step()
        self.log_dict({"g_loss": total_g_loss, "d_loss": d_loss}, prog_bar=True)
        self.total_g_loss_log.append(total_g_loss.item())
        self.g_loss_log.append(g_loss.item())
        self.pose_2d_loss_log.append(pose_2d_loss.item())
        self.c_loss_log.append(c_loss.item())
        self.d_loss_log.append(d_loss.item())

    def on_train_epoch_end(self):

        if self.current_epoch > 30:
            [g_scheduler, d_scheduler] = self.lr_schedulers()
            g_scheduler.step()
            d_scheduler.step()
            print(
                "current learning rate:",
                g_scheduler.get_last_lr(),
                d_scheduler.get_last_lr(),
            )

    def validation_step(self, batch, batch_idx):
        # Validation is the normal MPJPE calculation
        x, y, valid, activities = self.preprocess_input(*self.preprocess_batch(batch))
        y_hat = self.forward(x)
        result = (
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
            activities,
        )
        self.evaluator.add_result(*result)
        self.procrusted_evaluator.add_result(*result)

    def on_validation_epoch_end(self):
        if not self.is_silence:
            print(f"check #{self.val_print_count}")
            if len(self.total_g_loss_log) > 0:
                print(
                    f"training loss from {len(self.g_loss_log)} "
                    + f"batches:\nd_loss = {np.mean(self.d_loss_log)}\n"
                    + f"g_loss = {np.mean(self.g_loss_log)}\n"
                    + f"c_loss = {np.mean(self.c_loss_log)}\n"
                    + f"pose_2d_loss = {np.mean(self.pose_2d_loss_log)}\n"
                    + f"total_g_loss = {np.mean(self.total_g_loss_log)}"
                )
        self.total_g_loss_log = []
        self.g_loss_log = []
        self.pose_2d_loss_log = []
        self.c_loss_log = []
        self.d_loss_log = []

        pjpe, mpjpe, activities_mpjpe, activity_macro_mpjpe = (
            self.evaluator.get_result()
        )
        p_pjpe, p_mpjpe, p_activities_mpjpe, p_activity_macro_mpjpe = (
            self.procrusted_evaluator.get_result()
        )
        if not self.is_silence:
            print(f"val MPJPE from: {len(self.evaluator.mpjpe)} samples : {mpjpe}")
            print(
                f"val P-MPJPE from: {len(self.procrusted_evaluator.mpjpe)} samples : {p_mpjpe}"
            )
            if activity_macro_mpjpe is not None:
                print("activity_macro_mpjpe", activity_macro_mpjpe)
            if p_activity_macro_mpjpe is not None:
                print("activity_macro_procrusted_mpjpe", p_activity_macro_mpjpe)
        self.log("mpjpe", mpjpe)
        self.log("p_mpjpe", p_mpjpe)

        if activity_macro_mpjpe is not None:
            self.log("activity_macro_mpjpe", activity_macro_mpjpe)
        if p_activity_macro_mpjpe is not None:
            self.log("p_activity_macro_mpjpe", p_activity_macro_mpjpe)

        self.evaluator.reset()
        self.procrusted_evaluator.reset()
        self.val_history.append(
            {
                "pjpe": pjpe,
                "mpjpe": mpjpe,
                "activities_mpjpe": activities_mpjpe,
                "activity_macro_mpjpe": activity_macro_mpjpe,
                "p_pjpe": p_pjpe,
                "p_mpjpe": p_mpjpe,
                "p_activities_mpjpe": p_activities_mpjpe,
                "p_activity_macro_mpjpe": p_activity_macro_mpjpe,
            }
        )

        self.val_print_count += 1
        
    # def configure_optimizers(self):
    #     g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
    #     d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
    #     return g_opt, d_opt

    def test_step(self, batch, batch_idx):
        # Validation is the normal MPJPE calculation
        x, y, valid, activities = self.preprocess_input(*self.preprocess_batch(batch))
        y_hat = self.forward(x)
        result = (
            y_hat.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
            activities,
        )
        self.evaluator.add_result(*result)
        self.procrusted_evaluator.add_result(*result)

    def on_test_epoch_end(self):
        pjpe, mpjpe, activities_mpjpe, activity_macro_mpjpe = (
            self.evaluator.get_result()
        )
        p_pjpe, p_mpjpe, p_activities_mpjpe, p_activity_macro_mpjpe = (
            self.procrusted_evaluator.get_result()
        )
        self.log("mpjpe", mpjpe)
        self.log("p_mpjpe", p_mpjpe)

        if activity_macro_mpjpe is not None:
            self.log("activity_macro_mpjpe", activity_macro_mpjpe)
        if p_activity_macro_mpjpe is not None:
            self.log("p_activity_macro_mpjpe", p_activity_macro_mpjpe)

        self.evaluator.reset()
        self.procrusted_evaluator.reset()
        self.test_history.append(
            {
                "pjpe": pjpe,
                "mpjpe": mpjpe,
                "activities_mpjpe": activities_mpjpe,
                "activity_macro_mpjpe": activity_macro_mpjpe,
                "p_pjpe": p_pjpe,
                "p_mpjpe": p_mpjpe,
                "p_activities_mpjpe": p_activities_mpjpe,
                "p_activity_macro_mpjpe": p_activity_macro_mpjpe,
            }
        )



    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        g_scheduler = StepLR(g_opt, step_size=1, gamma=0.95)
        d_scheduler = StepLR(d_opt, step_size=1, gamma=0.95)

        return [g_opt, d_opt], [g_scheduler, d_scheduler]
