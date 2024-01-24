import torch
from modules.lifter_2d_3d.model.common.lit_base_model import LitBaseModel
from modules.lifter_2d_3d.model.linear_model.network.linear_model import BaselineModel


class LitSimpleBaselineLinear(LitBaseModel):
    def __init__(
        self,
        **args
    ):
        super().__init__(**args)
        self.model = BaselineModel(
            exclude_ankle=self.exclude_ankle,
            exclude_knee=self.exclude_knee
        )

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
