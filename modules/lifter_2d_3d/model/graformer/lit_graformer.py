import torch
from modules.lifter_2d_3d.model.common.lit_base_model import LitBaseModel
from modules.lifter_2d_3d.model.graformer.network.GraFormer import GraFormer
from modules.lifter_2d_3d.model.semgcn.network.utils.graph_utils import adj_mx_from_edges


class LitGraformer(LitBaseModel):
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

        adj = torch.tensor(
            adj_mx_from_edges(num_pts=num_pts, edges=self.connections).to_dense()
        )
        self.register_buffer("adj", adj)
        self.register_buffer("src_mask", torch.tensor([[[True] * num_pts]]))
        self.model = GraFormer(adj=adj, hid_dim=128, n_pts=num_pts)

    def preprocess_x(self, x):
        # add batch dimension if there is none.
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 2)
        return x.float()

    def preprocess_input(self, x, y, valid, activity):
        x = self.preprocess_x(x)
        y = y.float()
        return x, y, valid, activity

    def forward(self, x):
        y_hat = self.model(x, self.src_mask).squeeze(1)
        return y_hat
