import torch
import torch.nn as nn
from src.modules.lifter_2d_3d.model.linear_model.linear_model import (
    Linear,
    init_weights
)


class CameraNet(nn.Module):
    def __init__(
            self,
            input_dim,
            linear_size=1000,
            p_dropout=0.5,
            num_stages=2,
        ):
        super(CameraNet, self).__init__()
        # input_size = 16 * 2  # Input 2d-joints.
        self.input_size = input_dim
        # number of camera parameter (weak perspective camera model)
        output_size = 6

        self.w1 = nn.Linear(self.input_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(linear_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        # initialize model weights
        self.apply(init_weights)

    def forward(self, x):
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear blocks
        y = y.squeeze(-1)
        for linear in self.linear_stages:
            y = linear(y)

        y = self.w2(y)
        # print(f'output y.shape {y.shape}')
        return y

