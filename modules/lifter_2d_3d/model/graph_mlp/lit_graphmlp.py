from modules.lifter_2d_3d.model.common.lit_base_model import LitBaseModel
from modules.lifter_2d_3d.model.graph_mlp.network.graphmlp import Model as GraphMLP


class LitGraphMLP(LitBaseModel):
    def __init__(
        self,
        **args
    ):
        super().__init__(**args)
        self.model = GraphMLP(
            frames=1,
            depth=3,
            d_hid=1024,
            token_dim=256,
            channel=512,
            n_joints=13
        )

    def preprocess_x(self, x):
        # add batch dimension if there is none.
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 2)
        return x.float().unsqueeze(1).to(self.device)

    def preprocess_input(self, x, y, valid, activity):
        x = self.preprocess_x(x)
        y = y.float().to(self.device)
        return x, y, valid, activity

    def forward(self, x):
        y_hat = self.model(x).squeeze(1)
        return y_hat
