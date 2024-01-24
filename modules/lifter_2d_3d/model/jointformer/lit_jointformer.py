import torch
from torch.nn import functional as F
from modules.lifter_2d_3d.model.semgcn.network.utils.graph_utils import adj_mx_from_edges
from modules.lifter_2d_3d.model.jointformer.network.jointformer import JointTransformer
from modules.lifter_2d_3d.model.common.lit_base_model import LitBaseModel


class LitJointFormer(LitBaseModel):
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
        self.model = JointTransformer(
            num_joints_in=num_pts,
            n_layers=4,
            encoder_dropout=0,
            d_model=64,
            intermediate=True,
            spatial_encoding=False,
            pred_dropout=0.2,
            embedding_type="conv",
            adj=adj,
        )

    def preprocess_x(self, x):
        # add batch dimension if there is none.
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 2)
        return x.float()

    def preprocess_input(self, x, y, valid, activity):
        x = self.preprocess_x(x)
        y = y.float()
        return x, y, valid, activity

    def inference(self, x):
        out, enc_output, error = self.model(x)
        return out, enc_output, error

    def forward(self, x):
        out, _, _ = self.inference(x)
        return out[-1]

    def calculate_valid_loss(self, y_hat, y, valid):
        loss = F.mse_loss(y_hat, y, reduction="none")
        # mask out invalid batch
        loss = loss.sum(axis=2) * (valid).float()
        # Mean square error
        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        x, y, valid, activities = self.preprocess_input(
            *self.preprocess_batch(batch)
            )
        out, enc_output, error = self.inference(x)
        loss_3d_pos = 0
        for outputs_3d, error_3d in zip(out, error):
            true_error = torch.abs(outputs_3d.detach() - y)
            loss_3d_pos += (
                self.calculate_valid_loss(outputs_3d, y, valid)
                + self.calculate_valid_loss(error_3d, true_error, valid)
            ) / 2
        loss_3d_pos = loss_3d_pos / len(out)
        self.train_loss_log.append(torch.sqrt(loss_3d_pos).item())
        return loss_3d_pos

