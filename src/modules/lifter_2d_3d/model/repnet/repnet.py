import torch
import torch.nn as nn

class RepNet(nn.Module):
    def __init__(
            self,
            lifter2D_3D,
            camera_net,
            input_dim,
        ):
        super(RepNet, self).__init__()
        self.lifter2D_3D = lifter2D_3D
        self.camera_net = camera_net
        self.input_dim = input_dim

    def forward(self, x):
        pose_3d = self.lifter2D_3D(x)
        camera_out = self.camera_net(x)
        camera_matrix = camera_out.reshape((-1, 2, 3))
        pose_3d_matrix = pose_3d.reshape((-1, 3, int(self.input_dim/2)))
        reprojected_x = torch.reshape(camera_matrix @ pose_3d_matrix, [-1, self.input_dim])
        return camera_out, pose_3d, reprojected_x
