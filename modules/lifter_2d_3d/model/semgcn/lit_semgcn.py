import pytorch_lightning as pl
from modules.lifter_2d_3d.model.common.lit_base_model import LitBaseModel
from modules.lifter_2d_3d.model.semgcn.network.sem_gcn import SemGCN
from modules.lifter_2d_3d.model.semgcn.network.utils.graph_utils import (
    adj_mx_from_edges
)

node_groups = [
    [1, 3],  # right-eye-ear
    [0],  # nose
    [2, 4],  # left-eye-ear
    [5, 6],  # left-right-sholder
    [7, 9],  # right-elbow-wrist
    [8, 10],  # left-elbow-wrist
    [11, 12],  # left-right hip
]


class LitSemGCN(LitBaseModel):
    def __init__(
        self,
        **args
    ):
        super().__init__(**args)
        num_pts = 17
        if self.exclude_ankle:
            num_pts -= 2
            self.connections = self.connections[:-2]
        if self.exclude_knee:
            num_pts -= 2
            self.connections = self.connections[:-2]

        adj = adj_mx_from_edges(num_pts=num_pts, edges=self.connections).to_dense()

        self.model = SemGCN(
            adj=adj,
            hid_dim=128,
            coords_dim=(2, 3),
            num_layers=4,
            nodes_group=node_groups,
            p_dropout=None,
        )

    def preprocess_x(self, x):
        # add batch dimension if there is none.
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 2)
        x = x.float().squeeze(2)
        return x

    def preprocess_input(self, x, y, valid, activity):
        x = self.preprocess_x(x)
        y = y.float().squeeze(2)
        return x, y, valid, activity
