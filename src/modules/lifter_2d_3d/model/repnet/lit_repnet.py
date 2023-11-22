import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from src.modules.lifter_2d_3d.model.linear_model.linear_model import BaselineModel
from src.modules.lifter_2d_3d.model.repnet.discriminator import DiscriminatorModel
from src.modules.lifter_2d_3d.model.repnet.camera_net import CameraNet
from src.modules.lifter_2d_3d.model.repnet.repnet import RepNet
from src.modules.lifter_2d_3d.model.repnet.utils import (
    wasserstein_loss,
    camera_loss,
    weighted_pose_2d_loss
)
from collections import OrderedDict

import numpy as np


class LitRepNet(pl.LightningModule):

    def __init__(self,
                 lifter_2D_3D,
                 latent_dim: int = 100,
                 lr: float = 1e-4,
                 b1: float = 0.5,
                 b2: float = 0.9,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        # networks
        self.lifter_2D_3D = lifter_2D_3D
        self.num_keypoints = 13
        self.input_dim = self.num_keypoints
        self.camera_net = CameraNet(
            input_dim=self.input_dim * 2
        )
        self.generator = RepNet(
            self.lifter_2D_3D,
            self.camera_net,
            input_dim = self.input_dim * 2
        )
        self.discriminator = DiscriminatorModel(input_size=self.input_dim * 3)
        self.val_loss_log = []
        self.val_print_count = 0
        self.total_g_loss_log = []
        self.g_loss_log = []
        self.pose_2d_loss_log = []
        self.c_loss_log = []
        self.d_loss_log = []
        self.test_loss_log = []


    def forward(self, z):
        return self.generator.lifter2D_3D(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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
        g_opt, d_opt = self.optimizers()
        img_id, x, y, valid = batch
        input_2d = torch.flatten(x, start_dim=1).float().to(self.device)
        real_pose_3d = torch.flatten(y, start_dim=1).float().to(self.device)
        lambda_gp = 10

        # train generator
        # if optimizer_idx == 0:
        for i in range(1):
            # generate images
            camera_out, gen_pose_3d, reprojected_2d = self.generator(input_2d)
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(input_2d.size(0), 1)
            valid = valid.type_as(input_2d)

            g_loss = -torch.mean(self.discriminator(gen_pose_3d))
            c_loss = camera_loss(camera_out)
            pose_2d_loss = weighted_pose_2d_loss(input_2d, reprojected_2d, self.num_keypoints)
            total_g_loss = g_loss + c_loss + pose_2d_loss
            g_opt.zero_grad()
            self.manual_backward(total_g_loss)
            g_opt.step()
        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        for i in range(10):
            camera_out, gen_pose_3d, reprojected_2d = self.generator(input_2d)

            # Real images
            real_validity = self.discriminator(real_pose_3d)
            # Fake images
            fake_validity = self.discriminator(gen_pose_3d)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_pose_3d.data, gen_pose_3d.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()
        self.log_dict({"g_loss": total_g_loss, "d_loss": d_loss}, prog_bar=True)
        self.total_g_loss_log.append(total_g_loss.item())
        self.g_loss_log.append(g_loss.item())
        self.pose_2d_loss_log.append(pose_2d_loss.item())
        self.c_loss_log.append(c_loss.item())
        self.d_loss_log.append(d_loss.item())


    def validation_step(self, batch, batch_idx):
        img_id, x, y, valid = batch
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        y = torch.flatten(y, start_dim=1).float().to(self.device)
        y_hat = self.generator.lifter2D_3D(x)
        loss = F.mse_loss(y_hat.reshape(y_hat.shape[0], -1, 3)[valid], y.reshape(y.shape[0], -1, 3)[valid])
        self.log("val_loss", loss.item())
        self.val_loss_log.append(torch.sqrt(loss).item())
        return loss

    def on_validation_epoch_end(self):
        print(f'check #{self.val_print_count}')
        if len(self.total_g_loss_log) > 0:
            print(f'training loss from {len(self.g_loss_log)} ' +
                  f'batches:\nd_loss = {np.mean(self.d_loss_log)}\n' +
                  f'g_loss = {np.mean(self.g_loss_log)}\n' +
                  f'c_loss = {np.mean(self.c_loss_log)}\n' +
                  f'pose_2d_loss = {np.mean(self.pose_2d_loss_log)}\n' +
                  f'total_g_loss = {np.mean(self.total_g_loss_log)}')
        val_loss = np.mean(self.val_loss_log)
        print(f"val loss from: {len(self.val_loss_log)} batches : {val_loss * 1000}")
        self.log("val_loss", val_loss.item())
        self.total_g_loss_log = []
        self.g_loss_log = []
        self.pose_2d_loss_log = []
        self.c_loss_log = []
        self.d_loss_log = []
        self.val_print_count += 1
        self.val_loss_log = []


    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return g_opt, d_opt
