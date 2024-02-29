import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


connections = [
    (0, 1, 'nose_left_eye'), # nose & left_eye
    (0, 2, 'nose_right_eye'), # nose & right_eye
    # (1, 2, 'left_right_eye'), # left & right eyes
    (1, 3, 'left_eye_left_ear'), # left eye & ear
    (2, 4, 'right_eye_right_ear'), # right eye & ear
    (0, 5, 'nose_left_shoulder'), # nose & left shoulder
    (0, 6, 'nose_right_shoulder'), # nose & right shoulder
    # (3, 5, 'left_ear_shoulder'), # left ear & shoulder
    # (4, 6, 'right_ear_shoulder'), # right ear & shoulder
    # (5, 6, 'left_shoulder_right_sholder'), # left & right shoulder
    (5, 7, 'left_sholder_left_elbow'), # left shoulder & elbow
    (5, 11, 'left_shoulder_left_hip'), # left shoulder & hip
    (6, 8, 'right_shoulder_right_elbow'), # right shoulder & elbow
    (6, 12, 'right_shoulder_right_hip'), # right shoulder & hip
    (7, 9, 'left_elbow_left_wrist'), # left elbow & wrist
    (8, 10, 'right_elbow_right_wrist'), # right elbow & wrist
    # (11, 12, 'left_hip_right_hip'), # left & right hip
    # (11, 13, 'left_hip_left_knee'), # left hip & knee
    # (12, 14, 'right_hip_right_knee'), # right hip & knee
    # (13, 15, 'left_knee_left_ankle'), # left knee & ankle
    # (14, 16, 'right_knee_right_ankle') # right knee & ankle
]

connections = np.array(connections)[:,:2].astype(int).tolist()
# kcs_matrix = np.zeros((17, len(connections))).astype(float)
# for idx, c in enumerate(connections):
#     kcs_matrix[c[0], idx] = 1
#     kcs_matrix[c[1], idx] = -1

class DiscriminatorModel(nn.Module):
    def __init__(
            self,
            input_size,
            linear_dim=100,
            # p_dropout=0.5,
        ):
        super(DiscriminatorModel, self).__init__()

        self.l1 = nn.Linear(input_size, linear_dim)
        # pose path
        self.l2 = nn.Linear(linear_dim, linear_dim)
        self.l3 = nn.Linear(linear_dim, linear_dim)
        self.l4 = nn.Linear(linear_dim, linear_dim)
        # KCS path
        psi_shape = len(connections) * len(connections)
        self.kcs_l1 = nn.Linear(psi_shape, linear_dim)
        self.kcs_l2 = nn.Linear(linear_dim, linear_dim)
        self.kcs_l3 = nn.Linear(linear_dim, linear_dim)
        self.kcs_matrix = np.zeros((int(input_size / 3), len(connections))).astype(float)
        for idx, c in enumerate(connections):
            self.kcs_matrix[c[0], idx] = 1
            self.kcs_matrix[c[1], idx] = -1
        self.hidden_layer = nn.Linear(linear_dim * 2, linear_dim)
        self.predict_layer = nn.Linear(linear_dim, 1)
        # self.kcs_l3 = nn.Linear(linear_dim, linear_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def kcs_layer(self, x):
        # KCS matrix
        Ct = torch.tensor(self.kcs_matrix).float().to(self.device)
        # copy KCS matrix for every item in the batch
        C = torch.tile(Ct, (x.shape[0], 1))\
            .reshape((-1, self.kcs_matrix.shape[0], self.kcs_matrix.shape[1]))
        poses3 = torch.reshape(x, [-1, 3, self.kcs_matrix.shape[0]]).float()
        B = poses3 @ C
        Psi = B.permute([0, 2, 1]) @ B
        return Psi

    def forward(self, x):
        # pose_path
        l1_out = self.l1(x)
        l1_out = F.leaky_relu(l1_out)
        l2_out = self.l2(l1_out)
        l2_out = F.leaky_relu(l2_out)
        l3_out = self.l3(l2_out)
        l3_out = F.leaky_relu(l3_out)
        pose_path_skip = l1_out + l3_out
        pose_path_skip = F.leaky_relu(pose_path_skip)
        l4_out = self.l4(pose_path_skip)

        # kcs path
        psi = self.kcs_layer(x)
        psi_vec = torch.flatten(psi, start_dim=1)
        kcs_l1 = self.kcs_l1(psi_vec)
        kcs_l1 = F.leaky_relu(kcs_l1)
        kcs_l2 = self.kcs_l2(kcs_l1)
        kcs_l2 = F.leaky_relu(kcs_l2)
        kcs_l3 = self.kcs_l2(kcs_l2)
        kcs_l3 = F.leaky_relu(kcs_l3)
        kcs_skip = kcs_l1 + kcs_l3

        # output
        # hidden_out = torch.stack([l4_out, kcs_skip], dim=1)
        hidden_out = torch.cat([l4_out, kcs_skip], dim=1)
        hidden_out = self.hidden_layer(hidden_out)
        hidden_out = F.leaky_relu(hidden_out)
        out = self.predict_layer(hidden_out)
        return F.sigmoid(out)
